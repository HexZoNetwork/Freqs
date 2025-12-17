import argparse
import json
import sys
import os
import random

import chess
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from main import ChessCC
from nn_model import ChessNN
import chess_board.display


def play_game(args):
    board = chess.Board()
    # The --depth argument is ignored by the MCTS engine
    ai = ChessCC(model_path=args.model, time_limit_seconds=args.time, num_workers=args.workers)
    human_color = chess.WHITE if args.color.lower() == 'white' else chess.BLACK
    last_move = None

    while not board.is_game_over(claim_draw=True):
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"ChessCC vs. Human ({'White' if human_color == chess.WHITE else 'Black'})\n")
        chess_board.display.display_board(board, last_move=last_move)
        print(f"FEN: {board.fen()}")

        if board.turn == human_color:
            try:
                command = input("\nEnter move (e.g. e2e4), 'back', 'save file.fen', or 'load file.fen': ")
                parts = command.strip().split()
                action = parts[0].lower()

                if action == "save":
                    if len(parts) > 1:
                        filename = parts[1]
                        with open(filename, "w") as f:
                            f.write(board.fen())
                        print(f"Game state saved to {filename}")
                    else:
                        print("Please specify a filename.")
                    input("Press Enter to continue...")
                    continue
                elif action == "load":
                    if len(parts) > 1 and os.path.exists(parts[1]):
                        filename = parts[1]
                        with open(filename, "r") as f:
                            board.set_fen(f.read())
                        print(f"Game state loaded from {filename}")
                        ai.mcts_root = None # Reset AI state
                        last_move = None
                    else:
                        print("File not found.")
                    input("Press Enter to continue...")
                    continue
                
                if action in ["back", "undo", "takeback"]:
                    if len(board.move_stack) >= 2:
                        board.pop() # Pop AI's move
                        board.pop() # Pop human's move
                        # Reset the AI's internal tree to force a fresh search
                        ai.mcts_root = None 
                        # Update last_move for display, if moves are left
                        last_move = board.peek() if board.move_stack else None
                        continue # Skip to the next loop iteration
                    else:
                        print("Cannot take back any further.")
                        input("Press Enter to continue...")
                        continue

                move = chess.Move.from_uci(command)
                if move in board.legal_moves:
                    board.push(move)
                    last_move = move
                else:
                    print("Invalid move. Please try again.")
            except Exception as e:
                print(f"An error occurred: {e}")
        else:
            print("ChessCC is thinking...")
            last_move, _ = ai.find_best_move(board, visualize=args.visualize)
            board.push(last_move)

    print("\nGame Over!")
    print("Result: " + board.result(claim_draw=True))

def generate_data(args):
    print(f"Starting self-play data generation for {args.games} games...")
    ai = ChessCC(model_path=args.model, time_limit_seconds=args.time, add_exploration_noise=True, num_workers=args.workers)
    
    # Load existing replay buffer or create a new one
    if os.path.exists(args.replay_buffer):
        print(f"Loading existing replay buffer from {args.replay_buffer}")
        replay_buffer = torch.load(args.replay_buffer)
    else:
        print("Creating new replay buffer.")
        replay_buffer = []

    for i in range(args.games):
        board = chess.Board()
        game_history = [] # (board_tensor, policy_target)
        
        while not board.is_game_over(claim_draw=True) and len(game_history) < 200:
            # Use temperature to control exploration
            # High temperature for the first 30 moves, then deterministic
            move_number = len(game_history)
            temperature = 1.0 if move_number < 30 else 0.0

            board_tensor = ai.board_to_tensor(board)
            move, policy_target = ai.find_best_move(board, temperature=temperature)
            game_history.append((board_tensor, policy_target))
            board.push(move)
        
        result = board.result(claim_draw=True)
        if result == '1-0': z = 1.0
        elif result == '0-1': z = -1.0
        else: z = 0.0
        
        # Assign final game outcome to all positions in the history
        for i, (board_tensor, policy_target) in enumerate(game_history):
            # The value is from the perspective of the player to move in that position
            value_target = z if (i % 2 == 0) else -z
            
            # Add original sample
            replay_buffer.append((board_tensor, policy_target, torch.tensor([value_target], dtype=torch.float32)))

        print(f"Game {i+1}/{args.games} finished. Result: {result}. Replay buffer size: {len(replay_buffer)}")

    # Trim the replay buffer to keep it from growing indefinitely
    if len(replay_buffer) > args.buffer_size:
        print(f"Trimming replay buffer from {len(replay_buffer)} to {args.buffer_size} samples.")
        replay_buffer = replay_buffer[-args.buffer_size:]

    # Save the updated replay buffer
    torch.save(replay_buffer, args.replay_buffer)
    print(f"Data generation complete. Saved replay buffer to {args.replay_buffer}.")

def train_network(args):
    print(f"Loading training data from replay buffer: {args.replay_buffer}...")
    training_data = torch.load(args.replay_buffer)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    start_epoch = 0

    # Load checkpoint if it exists
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        print(f"Loading checkpoint from {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    elif os.path.exists(args.model):
        # Fallback to loading just the model weights if no checkpoint is provided
        print(f"Loading initial model weights from {args.model}")
        model.load_state_dict(torch.load(args.model, map_location=device))

    policy_loss_fn = torch.nn.CrossEntropyLoss()
    value_loss_fn = torch.nn.MSELoss()

    # Unpack data
    board_tensors, policy_targets, value_targets = zip(*training_data)
    dataset = TensorDataset(torch.cat(board_tensors), torch.stack(policy_targets), torch.stack(value_targets))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    print(f"Starting training for {args.epochs} epochs...")
    model.train() # Set model to training mode
    for epoch in range(start_epoch, args.epochs):
        total_loss = 0
        for boards, policies, values in dataloader:
            boards, policies, values = boards.to(device), policies.to(device), values.to(device)

            optimizer.zero_grad()
            pred_policies, pred_values = model(boards)

            loss_policy = policy_loss_fn(pred_policies, policies)
            loss_value = value_loss_fn(pred_values, values)
            loss = loss_policy + loss_value

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{args.epochs}, Average Loss: {total_loss / len(dataloader):.4f}")
        scheduler.step()

        # Save checkpoint at the end of each epoch
        if args.checkpoint_path:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, args.checkpoint_path)

    # Save the trained model
    torch.save(model.state_dict(), args.output_model)
    print(f"Training complete. Saved new model to {args.output_model}")

    if args.run_eval:
        print("\n--- Starting Evaluation ---")
        eval_args = argparse.Namespace(
            new_model=args.output_model,
            best_model=args.model,
            games=args.eval_games,
            time=1, # Use a short time limit for evaluation games
            workers=args.workers
        )
        evaluate_model(eval_args)

def evaluate_model(args):
    print(f"Evaluating new model '{args.new_model}' against best model '{args.best_model}'.")
    
    ai_new = ChessCC(model_path=args.new_model, time_limit_seconds=args.time, num_workers=args.workers)
    ai_best = ChessCC(model_path=args.best_model, time_limit_seconds=args.time, num_workers=args.workers)

    scores = {'new_wins': 0, 'best_wins': 0, 'draws': 0}

    for i in range(args.games):
        board = chess.Board()
        # Alternate who plays white
        white_player = ai_new if i % 2 == 0 else ai_best
        black_player = ai_best if i % 2 == 0 else ai_new

        while not board.is_game_over(claim_draw=True) and board.fullmove_number <= 150:
            move, _ = white_player.find_best_move(board) if board.turn == chess.WHITE else black_player.find_best_move(board)
            board.push(move)
        
        result = board.result(claim_draw=True)
        if (result == '1-0' and i % 2 == 0) or (result == '0-1' and i % 2 != 0):
            scores['new_wins'] += 1
        elif (result == '1-0' and i % 2 != 0) or (result == '0-1' and i % 2 == 0):
            scores['best_wins'] += 1
        else:
            scores['draws'] += 1
        
        print(f"Game {i+1}/{args.games} finished. Result: {result}. Score: New {scores['new_wins']} - Best {scores['best_wins']} - Draws {scores['draws']}")

    win_rate = (scores['new_wins'] + 0.5 * scores['draws']) / args.games
    print(f"\nEvaluation complete. New model win rate: {win_rate:.2%}")
    if win_rate > 0.55:
        print("New model is stronger! It will become the new best model.")
        # In a real pipeline, you would copy the new model over the best model here.
        # For example: shutil.copy(args.new_model, args.best_model)
    else:
        print("New model is not stronger. Discarding.")

def watch_game(args):
    board = chess.Board()
    ai = ChessCC(model_path=args.model, time_limit_seconds=args.time, num_workers=args.workers)

    while not board.is_game_over(claim_draw=True):
        print("\n" + str(board))
        move, _ = ai.find_best_move(board, visualize=args.visualize)
        board.push(move)
    
    print("\nGame Over!")
    print("Result: " + board.result(claim_draw=True))
    print(board)

def main():
    parser = argparse.ArgumentParser(description="ChessCC: A command-line chess AI.")
    subparsers = parser.add_subparsers(dest='command', required=True, help='Available commands')

    play_parser = subparsers.add_parser('play', help='Play against the AI.')
    play_parser.add_argument('--model', type=str, help='Path to the neural network model file.')
    play_parser.add_argument('--time', type=int, default=5, help='Time limit in seconds for AI to move.')
    play_parser.add_argument('--workers', type=int, help='Number of CPU cores to use for search.')
    play_parser.add_argument('--visualize', action='store_true', help='Print the MCTS search tree after each move.')
    play_parser.add_argument('--color', type=str, default='white', choices=['white', 'black'], help='Your color.')

    gen_parser = subparsers.add_parser('generate', help='Run self-play to generate training data.')
    gen_parser.add_argument('--model', type=str, help='Path to the model to use for self-play.')
    gen_parser.add_argument('--games', type=int, default=10, help='Number of games to simulate.')
    gen_parser.add_argument('--time', type=int, default=1, help='Time limit in seconds for AI moves during self-play.')
    gen_parser.add_argument('--workers', type=int, help='Number of CPU cores to use for data generation.')
    gen_parser.add_argument('--replay-buffer', type=str, default='replay_buffer.pt', help='Path to the replay buffer file.')
    gen_parser.add_argument('--buffer-size', type=int, default=50000, help='Maximum number of game states to keep in the replay buffer.')

    train_parser = subparsers.add_parser('train', help='Train the neural network on generated data.')
    train_parser.add_argument('--replay-buffer', type=str, required=True, help='Path to the replay buffer file to train on.')
    train_parser.add_argument('--model', type=str, required=True, help='Path to the model to load for continued training.')
    train_parser.add_argument('--output-model', type=str, required=True, help='Path to save the newly trained model.')
    train_parser.add_argument('--checkpoint-path', type=str, help='Path to save/load training checkpoints.')
    train_parser.add_argument('--run-eval', action='store_true', help='Run evaluation against the best model after training.')
    train_parser.add_argument('--eval-games', type=int, default=40, help='Number of games to play for evaluation.')
    train_parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    train_parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    train_parser.add_argument('--batch-size', type=int, default=64, help='Training batch size.')

    watch_parser = subparsers.add_parser('watch', help='Watch two AIs play against each other.')
    watch_parser.add_argument('--model', type=str, help='Path to the neural network model file.')
    watch_parser.add_argument('--time', type=int, default=5, help='Time limit in seconds for AI moves.')
    watch_parser.add_argument('--workers', type=int, help='Number of CPU cores to use for search.')
    watch_parser.add_argument('--visualize', action='store_true', help='Print the MCTS search tree after each move.')

    # UCI command
    uci_parser = subparsers.add_parser('uci', help='Run the engine in UCI mode for use with GUIs.')

    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Pit two models against each other.')
    eval_parser.add_argument('--new-model', type=str, required=True, help='Path to the new model to be evaluated.')
    eval_parser.add_argument('--best-model', type=str, required=True, help='Path to the current best model to play against.')
    eval_parser.add_argument('--games', type=int, default=40, help='Number of games to play.')
    eval_parser.add_argument('--workers', type=int, help='Number of CPU cores to use for search.')

    args = parser.parse_args()

    if args.command == 'play': play_game(args)
    elif args.command == 'generate': generate_data(args)
    elif args.command == 'train': train_network(args)
    elif args.command == 'watch': watch_game(args)
    elif args.command == 'evaluate': evaluate_model(args)
    elif args.command == 'uci':
        from uci import uci_loop
        uci_loop()

if __name__ == "__main__":
    main()