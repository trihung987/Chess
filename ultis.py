import numpy as np
from chess import Board
import evaluations
import chess
import os
from tqdm import tqdm 
import chess.pgn


def board_to_matrix(board: Board):
    # 8x8 is a size of the chess board.
    matrix = np.zeros((13, 8, 8))
    piece_map = board.piece_map()

    # Populate first 12 8x8 boards (where pieces are)
    for square, piece in piece_map.items():

        row, col = divmod(square, 8)
        piece_type = piece.piece_type - 1
        piece_color = 0 if piece.color else 6
        matrix[piece_type + piece_color, row, col] = 1

    # Populate the legal moves board (13th 8x8 board)
    legal_moves = board.legal_moves
    for move in legal_moves:
        to_square = move.to_square
        row_to, col_to = divmod(to_square, 8)
        matrix[12, row_to, col_to] = 1

    return matrix

def createDataSet(pgn_folder, max_moves=1000):
    inputs, targets = [], []
    files = [os.path.join(pgn_folder, f) for f in os.listdir(pgn_folder) if f.endswith(".pgn")]
    total_moves = 0

    for file in tqdm(files):
        with open(file, 'r') as f:
            while total_moves < max_moves:
                game = chess.pgn.read_game(f)
                if not game:
                    break
                board = game.board()
                for move in game.mainline_moves():
                    if total_moves >= max_moves:
                        break
                    board.push(move)
                    vec = board_to_matrix(board)
                    score = evaluations.evaluate_board(board)
                    if score is None:
                        continue
                    inputs.append(vec)
                    targets.append(score)
                    total_moves += 1
                print(total_moves)
    np.savez("preparedata/data4",b=inputs, v=targets)
    return np.array(inputs, dtype=np.float32), np.array(targets)

def create_input_for_nn(max_moves=1000):
   return createDataSet("datas/Lichess Elite Database", max_moves)

def get_data(type):
    container = np.load("preparedata/data"+type+ ".npz")

    # Extract board positions (b) and evaluations (v) from the loaded container
    b, v = container['b'], container['v']
    return b, v

def encode_moves(moves):
    move_to_int = {move: idx for idx, move in enumerate(set(moves))}
    return np.array([move_to_int[move] for move in moves], dtype=np.float32), move_to_int


