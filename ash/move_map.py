import chess

def _create_move_maps():
    """
    Creates mappings for converting chess moves to and from the 4672-dimensional
    neural network output, based on the AlphaZero paper's 8x8x73 representation.
    """
    uci_to_index = {}
    index_to_uci = [""] * 4672

    # 1. Queen-like moves (56 planes for 8 directions * 7 distances)
    queen_directions = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
    plane = 0
    for dr, df in queen_directions:
        for dist in range(1, 8):
            for from_sq in chess.SQUARES:
                from_r, from_f = chess.square_rank(from_sq), chess.square_file(from_sq)
                to_r, to_f = from_r + dr * dist, from_f + df * dist
                if 0 <= to_r < 8 and 0 <= to_f < 8:
                    to_sq = chess.square(to_f, to_r)
                    uci = chess.Move(from_sq, to_sq).uci()
                    index = from_sq * 73 + plane
                    uci_to_index[uci] = index
                    index_to_uci[index] = uci
            plane += 1

    # 2. Knight moves (8 planes)
    knight_moves = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]
    for dr, df in knight_moves:
        for from_sq in chess.SQUARES:
            from_r, from_f = chess.square_rank(from_sq), chess.square_file(from_sq)
            to_r, to_f = from_r + dr, from_f + df
            if 0 <= to_r < 8 and 0 <= to_f < 8:
                to_sq = chess.square(to_f, to_r)
                uci = chess.Move(from_sq, to_sq).uci()
                index = from_sq * 73 + plane
                uci_to_index[uci] = index
                index_to_uci[index] = uci
        plane += 1

    # 3. Pawn underpromotions (9 planes for 3 directions * 3 pieces)
    promotion_pieces = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
    promotion_directions = [-1, 0, 1] # Capture left, straight, capture right
    for piece in promotion_pieces:
        for df in promotion_directions:
            for from_f in range(8):
                to_f = from_f + df
                if 0 <= to_f < 8:
                    # White promotion
                    uci_w = chess.Move(chess.square(from_f, 6), chess.square(to_f, 7), piece).uci()
                    index_w = chess.square(from_f, 6) * 73 + plane
                    uci_to_index[uci_w] = index_w
                    index_to_uci[index_w] = uci_w
                    # Black promotion
                    uci_b = chess.Move(chess.square(from_f, 1), chess.square(to_f, 0), piece).uci()
                    index_b = chess.square(from_f, 1) * 73 + plane
                    uci_to_index[uci_b] = index_b
                    index_to_uci[index_b] = uci_b
            plane += 1

    return uci_to_index, index_to_uci

# Pre-compute the map when the module is imported.
UCI_TO_INDEX, INDEX_TO_UCI = _create_move_maps()