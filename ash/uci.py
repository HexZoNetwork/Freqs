import sys
import chess
from main import ChessCC

def uci_loop():
    """
    The main loop for the Universal Chess Interface.
    Listens for commands from a GUI and responds.
    """
    # Redirect stderr to a log file to avoid interfering with UCI communication
    sys.stderr = open("uci_log.txt", "w")

    # Initialize the engine
    # In UCI mode, time is controlled by the 'go' command, not a fixed limit.
    ai = ChessCC(model_path="chess_model_v2.pt") # Default model, can be configured
    board = chess.Board()

    while True:
        try:
            line = sys.stdin.readline().strip()
            if not line:
                continue

            sys.stderr.write(f"GUI -> Engine: {line}\n")
            parts = line.split()
            command = parts[0]

            if command == "uci":
                sys.stdout.write("id name ChessCC\n")
                sys.stdout.write("id author Gemini Code Assist\n")
                sys.stdout.write("option name SyzygyPath type string default <empty>\n")
                sys.stdout.write("uciok\n")
                sys.stdout.flush()

            elif command == "isready":
                sys.stdout.write("readyok\n")
                sys.stdout.flush()

            elif command == "setoption":
                if parts[1] == "name" and parts[2] == "SyzygyPath" and parts[3] == "value":
                    path = " ".join(parts[4:])
                    ai.set_tablebase_path(path)

            elif command == "ucinewgame":
                board.reset()

            elif command == "position":
                moves_start_index = -1
                if "startpos" in parts:
                    board.reset()
                    if "moves" in parts:
                        moves_start_index = parts.index("moves") + 1
                elif "fen" in parts:
                    fen_start_index = parts.index("fen") + 1
                    fen_end_index = parts.index("moves") if "moves" in parts else len(parts)
                    fen = " ".join(parts[fen_start_index:fen_end_index])
                    board.set_fen(fen)
                    if "moves" in parts:
                        moves_start_index = fen_end_index + 1
                
                if moves_start_index != -1:
                    for move_uci in parts[moves_start_index:]:
                        board.push_uci(move_uci)

            elif command == "go":
                # Parse time controls from the 'go' command
                wtime, btime, winc, binc, movestogo = -1, -1, 0, 0, 40 # Sensible defaults
                if "wtime" in parts:
                    wtime = int(parts[parts.index("wtime") + 1])
                if "btime" in parts:
                    btime = int(parts[parts.index("btime") + 1])
                if "winc" in parts:
                    winc = int(parts[parts.index("winc") + 1])
                if "binc" in parts:
                    binc = int(parts[parts.index("binc") + 1])
                if "movestogo" in parts:
                    movestogo = int(parts[parts.index("movestogo") + 1])

                # --- Complexity-Aware Time Management ---
                time_to_use = 0
                time_for_move = 0
                if board.turn == chess.WHITE:
                    if wtime != -1:
                        # Add increment and calculate time for this move
                        available_time = wtime + movestogo * winc
                        time_for_move = (available_time / movestogo) + winc
                else: # Black's turn
                    if btime != -1:
                        available_time = btime + movestogo * binc
                        time_for_move = (available_time / movestogo) + binc
                
                if time_for_move > 0:
                    # Calculate complexity factor based on policy entropy
                    entropy = ai.get_policy_entropy(board)
                    # Normalize entropy (max entropy for ~35 moves is ~3.5) and scale it
                    complexity_factor = 1.0 + min(1.5, max(0, (entropy - 1.5) / 2.0)) # Scale between ~0.75 and 2.5
                    time_to_use = time_for_move * complexity_factor

                
                if time_to_use > 0:
                    # Convert ms to seconds and leave a small safety buffer
                    ai.time_limit = int(max(1, (time_to_use / 1000) * 0.95))
                else:
                    # Fallback if no time control is given
                    ai.time_limit = 5 

                best_move, _ = ai.find_best_move(board, temperature=0.0)
                
                # Find the best response to our move to tell the GUI we are pondering it
                ponder_move = ""
                if ai.mcts_root:
                    best_child = sorted(ai.mcts_root.children, key=lambda c: c.visits)[-1]
                    ponder_move = f" ponder {best_child.move.uci()}"

                sys.stdout.write(f"bestmove {best_move.uci()}{ponder_move}\n")
                sys.stdout.flush()

            elif command == "quit":
                break

        except Exception as e:
            sys.stderr.write(f"Error processing command: {line}\n")
            sys.stderr.write(f"Exception: {e}\n")
            sys.stderr.flush()