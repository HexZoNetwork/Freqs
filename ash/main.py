# chess_cc_pro.py

import chess
import random
import math
import json
import os
import chess.syzygy
import multiprocessing
import time
from collections import OrderedDict

import numpy as np
import torch
import tensorflow as tf

from nn_model import ChessNN
from move_map import UCI_TO_INDEX

class MCTSNode:
    """A node in the Monte Carlo Tree Search tree."""
    def __init__(self, board: chess.Board, parent=None, move=None):
        self.board = board
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0
        self.visits = 0
        self.policy_prior = 0.0
        # A dictionary to store policy priors for child moves
        self.child_priors = {}

    def uct_select_child(self, c_puct=1.0):
        """
        Select a child node using the PUCT (Polynomial Upper Confidence Trees) formula.
        This balances exploitation (Q value) and exploration (U value).
        """
        best_score = -float('inf')
        best_child = None
        for child in self.children:
            q_value = child.wins / child.visits if child.visits > 0 else 0
            u_value = c_puct * child.policy_prior * (math.sqrt(self.visits) / (1 + child.visits))
            score = q_value + u_value
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def add_child(self, move):
        new_board = self.board.copy()
        new_board.push(move)
        n = MCTSNode(new_board, parent=self, move=move)
        self.children.append(n)
        return n

    def update(self, result):
        """Update this node - one visit, and add result to wins."""
        self.visits += 1
        self.wins += result

class ChessCC:
    def __init__(self, model_path=None, time_limit_seconds=5, opening_book_path="opening_book.json", add_exploration_noise=False, num_workers=None):
        self.time_limit = time_limit_seconds
        self.opening_book = self.load_opening_book(opening_book_path)

        # Load the neural network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ChessNN().to(self.device)
        self.add_exploration_noise = add_exploration_noise
        self.dirichlet_alpha = 0.3
        self.exploration_fraction = 0.25
        self.tablebases = None
        self.mcts_root = None # To store the root of the MCTS tree for reuse
        self.num_workers = num_workers if num_workers is not None else (os.cpu_count() or 1)
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval() # Set the model to evaluation mode

    def load_opening_book(self, path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def set_tablebase_path(self, path):
        """Initializes the Syzygy tablebases from the given path."""
        try:
            self.tablebases = chess.syzygy.open_tablebase(path)
            print(f"Syzygy tablebases loaded successfully from {path}")
        except Exception as e:
            print(f"Failed to load tablebases: {e}")
            self.tablebases = None

    def get_policy_entropy(self, board: chess.Board) -> float:
        """
        Calculates the entropy of the neural network's policy for the current board.
        A higher entropy indicates more uncertainty/complexity from the NN's perspective.
        """
        if board.is_game_over():
            return 0.0

        board_tensor = self.board_to_tensor(board).to(self.device)
        with torch.no_grad():
            policy_logits, _ = self.model(board_tensor)

        legal_moves = list(board.legal_moves)
        policy = {move: policy_logits[0, self.move_to_nn_index(move)] for move in legal_moves}
        policy_exp = {move: math.exp(p) for move, p in policy.items()}
        sum_exp = sum(policy_exp.values())

        if sum_exp == 0:
            return 0.0

        probs = [p_exp / sum_exp for p_exp in policy_exp.values()]
        entropy = -sum(p * math.log(p + 1e-9) for p in probs) # Add epsilon for stability
        return entropy

    def board_to_tensor(self, board: chess.Board, flip: bool = False) -> torch.Tensor:
        """
        Converts a chess board state to a tensor representation for the neural network.
        The tensor has multiple channels, each representing a feature of the board.
        Shape: (C, H, W) = (18, 8, 8)
        """
        # 12 channels for pieces, 1 for color, 4 for castling, 1 for en passant
        tensor = np.zeros((18, 8, 8), dtype=np.float32)

        # Channel mapping for pieces
        piece_to_channel = {
            (chess.PAWN, chess.WHITE): 0, (chess.KNIGHT, chess.WHITE): 1, (chess.BISHOP, chess.WHITE): 2,
            (chess.ROOK, chess.WHITE): 3, (chess.QUEEN, chess.WHITE): 4, (chess.KING, chess.WHITE): 5,
            (chess.PAWN, chess.BLACK): 6, (chess.KNIGHT, chess.BLACK): 7, (chess.BISHOP, chess.BLACK): 8,
            (chess.ROOK, chess.BLACK): 9, (chess.QUEEN, chess.BLACK): 10, (chess.KING, chess.BLACK): 11,
        }

        for piece_type, color in piece_to_channel:
            channel = piece_to_channel[(piece_type, color)]
            # The board is iterated from a1 (0) to h8 (63)
            for square in board.pieces(piece_type, color):
                rank, file = chess.square_rank(square), chess.square_file(square)
                if flip:
                    file = 7 - file
                tensor[channel, rank, file] = 1

        # Channel 12: Player to move (1 for White, 0 for Black)
        if board.turn == chess.WHITE:
            tensor[12, :, :] = 1

        # Channels 13-16: Castling rights (swap kingside/queenside if flipping)
        white_ks, white_qs = (board.has_queenside_castling_rights(chess.WHITE), board.has_kingside_castling_rights(chess.WHITE)) if flip else (board.has_kingside_castling_rights(chess.WHITE), board.has_queenside_castling_rights(chess.WHITE))
        black_ks, black_qs = (board.has_queenside_castling_rights(chess.BLACK), board.has_kingside_castling_rights(chess.BLACK)) if flip else (board.has_kingside_castling_rights(chess.BLACK), board.has_queenside_castling_rights(chess.BLACK))
        
        if white_ks: tensor[13, :, :] = 1
        if white_qs: tensor[14, :, :] = 1
        if black_ks: tensor[15, :, :] = 1
        if black_qs: tensor[16, :, :] = 1

        # Channel 17: En passant square
        if board.ep_square:
            rank, file = chess.square_rank(board.ep_square), chess.square_file(board.ep_square)
            if flip:
                file = 7 - file
            tensor[17, rank, file] = 1

        return torch.from_numpy(tensor).unsqueeze(0) # Add batch dimension

    def _run_mcts_simulation(self, board_fen, time_limit, temperature):
        """
        Finds the best move using Monte Carlo Tree Search.
        During training, this also returns the policy target for the root node.
        """
        board_fen_pos = board_fen.split(' ')[0]
        if board_fen_pos in self.opening_book:
            # Opening book is not used during training/data generation
            # It's only for playing against a human/another AI
            if temperature == 0.0:
                move_uci = random.choice(self.opening_book[board_fen_pos])
                print(f"ChessCC plays from opening book: {move_uci}")
                # We don't have a policy target for book moves, so we return a dummy one
                return chess.Move.from_uci(move_uci), torch.zeros(4672)
        
        board = chess.Board(board_fen)

        # --- Tree Reuse Logic ---
        # Check if the current board position is a child of the previous root
        if self.mcts_root and self.mcts_root.children:
            root_node = None
            found_child = False
            for child in self.mcts_root.children:
                if child.board == board:
                    root_node = child
                    root_node.parent = None # The child becomes the new root
                    print("--- Ponder Hit! Reusing MCTS tree. ---")
                    found_child = True
                    break
            if not found_child:
                self.mcts_root = MCTSNode(board=board) # Discard old tree
        else:
            self.mcts_root = MCTSNode(board=board) # No tree to reuse
        
        root_node = self.mcts_root

        # --- Initial expansion of the root node ---
        if not root_node.children and not root_node.board.is_game_over():
            board_tensor = self.board_to_tensor(root_node.board).to(self.device)
            # Note: In a real parallel implementation, the model might need to be passed or reloaded.
            with torch.no_grad():
                policy_logits, _ = self.model(board_tensor)
            
            policy = {move: policy_logits[0, self.move_to_nn_index(move)] for move in root_node.board.legal_moves}
            policy_exp = {move: math.exp(p) for move, p in policy.items()}
            sum_exp = sum(policy_exp.values())
            
            # Add Dirichlet noise for exploration during training
            if self.add_exploration_noise:
                noise = None
                legal_moves = list(root_node.board.legal_moves)
                noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_moves))
            
            for i, move in enumerate(root_node.board.legal_moves):
                child = root_node.add_child(move)
                prior = policy_exp[move] / sum_exp
                if self.add_exploration_noise:
                    child.policy_prior = (1 - self.exploration_fraction) * prior + self.exploration_fraction * noise[i] # type: ignore
                else:
                    child.policy_prior = prior
        # --- End of initial expansion ---

        start_time = time.time()
        num_sims = 0
        while time.time() - start_time < time_limit:
            # 1. Selection
            node = root_node
            while node.children:
                node = node.uct_select_child()

            # 2. Expansion & 3. Simulation (using NN)
            # If the node is a leaf node, expand it.
            # The NN's value output replaces the random rollout.
            if not node.children and not node.board.is_game_over():
                # Get NN policy and value for the current node
                board_tensor = self.board_to_tensor(node.board).to(self.device)
                with torch.no_grad():
                    policy_logits, value_tensor = self.model(board_tensor)
                
                value = value_tensor.item()

                # Filter policy for legal moves and apply softmax
                policy = {move: policy_logits[0, self.move_to_nn_index(move)] for move in node.board.legal_moves}
                policy_exp = {move: math.exp(p) for move, p in policy.items()}
                sum_exp = sum(policy_exp.values())
                
                # Create children and assign policy priors
                for move, p_exp in policy_exp.items():
                    child = node.add_child(move)
                    child.policy_prior = p_exp / sum_exp

            else:
                # If the game is over at this node, the value is the actual result
                result = node.board.result()
                if result == '1-0': value = 1.0
                elif result == '0-1': value = -1.0
                else: value = 0.0

            # 4. Backpropagation
            while node is not None:
                # The value is from the perspective of the current player at the node.
                # We need to flip it for the parent.
                node.update(value)
                value = -value
                node = node.parent
            num_sims += 1

        return root_node, num_sims

    def find_best_move(self, board: chess.Board, temperature: float = 0.0, visualize: bool = False) -> tuple[chess.Move, torch.Tensor]:
        """
        Finds the best move using a parallelized Monte Carlo Tree Search.
        """
        board_fen = board.fen()
        # In UCI mode, the GUI sends 'ucinewgame', so we should reset our tree
        if board.fullmove_number <= 1 and board.ply() <= 1:
            self.mcts_root = None
        board_fen_pos = board_fen.split(' ')[0]

        # --- Tablebase Probe ---
        if self.tablebases and chess.popcount(board.occupied) <= self.tablebases.max_pieces:
            try:
                best_move = self.tablebases.probe_wdl(board)[1]
                print("--- Tablebase Hit! Playing perfect move. ---")
                return best_move, torch.zeros(4672)
            except IndexError: # This can happen if the position is a terminal draw/loss
                pass # Fall through to MCTS

        if board_fen_pos in self.opening_book and temperature == 0.0:
            move_uci = random.choice(self.opening_book[board_fen_pos])
            print(f"ChessCC plays from opening book: {move_uci}")
            return chess.Move.from_uci(move_uci), torch.zeros(4672)

        # Revert to single-threaded search to allow for correct tree reuse
        root_node, num_sims = self._run_mcts_simulation(board_fen, self.time_limit, temperature)
        print(f"Completed {num_sims} simulations.")

        # Store the tree for potential reuse on the next move
        self.mcts_root = root_node

        # Optionally visualize the search tree
        if visualize:
            self.visualize_tree(root_node)

        # Create policy target for training: a vector of visit counts
        policy_target = torch.zeros(4672)
        total_visits = sum(child.visits for child in root_node.children)
        for child in root_node.children:
            policy_target[self.move_to_nn_index(child.move)] = child.visits / total_visits if total_visits > 0 else 0
        
        # Select move based on temperature
        if temperature == 0.0:
            # Greedy selection: choose the move with the most visits
            best_move = sorted(root_node.children, key=lambda c: c.visits)[-1].move
        else:
            # Probabilistic selection based on visit counts and temperature
            visit_counts = np.array([child.visits for child in root_node.children])
            # Apply temperature to the distribution
            visit_dist = visit_counts**(1 / temperature)
            visit_dist /= np.sum(visit_dist)
            # Sample a move from the distribution
            move_index = np.random.choice(len(root_node.children), p=visit_dist)
            best_move = root_node.children[move_index].move

        return best_move, policy_target

    def move_to_nn_index(self, move: chess.Move) -> int:
        """Converts a chess.Move object to its corresponding neural network output index."""
        return UCI_TO_INDEX.get(move.uci(), -1) # Return -1 for moves not in map (e.g., standard promotions)

    def visualize_tree(self, root_node, max_depth=3, max_children=5):
        """Prints a visualization of the MCTS tree."""
        print("\n--- MCTS Tree Visualization ---")
        self._print_node_recursive(root_node, 0, max_depth, max_children)

    def _print_node_recursive(self, node, depth, max_depth, max_children):
        if depth > max_depth or not node.visits:
            return

        indent = "  " * depth
        move_str = node.move.uci() if node.move else "Root"
        win_rate = (node.wins / node.visits) * 100 if node.visits > 0 else 0
        
        # For the root node, the Q-value is from its perspective. For children, it's from the parent's perspective.
        q_value = (node.wins / node.visits) if node.visits > 0 else 0
        if node.parent:
            q_value = 1 - q_value

        print(f"{indent}{move_str: <6} | Visits: {node.visits: <5} | Win Rate: {win_rate:5.1f}% | Q: {q_value:.2f} | P: {node.policy_prior:.3f}")

        sorted_children = sorted(node.children, key=lambda c: c.visits, reverse=True)
        for i, child in enumerate(sorted_children):
            if i >= max_children: break
            self._print_node_recursive(child, depth + 1, max_depth, max_children)
