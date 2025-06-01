import chess
import chess.pgn
import numpy as np
import os
import pickle
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def board_to_features(board):
    features = np.zeros(768 + 8, dtype=np.float32)
    piece_idx = {
        (chess.PAWN, True): 0,
        (chess.KNIGHT, True): 1,
        (chess.BISHOP, True): 2,
        (chess.ROOK, True): 3,
        (chess.QUEEN, True): 4,
        (chess.KING, True): 5,
        (chess.PAWN, False): 6,
        (chess.KNIGHT, False): 7,
        (chess.BISHOP, False): 8,
        (chess.ROOK, False): 9,
        (chess.QUEEN, False): 10,
        (chess.KING, False): 11,
    }
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            idx = piece_idx.get((piece.piece_type, piece.color), -1)
            if idx >= 0:
                features[idx * 64 + square] = 1
    features[768 + 0] = int(board.turn)
    features[768 + 1] = int(board.has_kingside_castling_rights(True))
    features[768 + 2] = int(board.has_queenside_castling_rights(True))
    features[768 + 3] = int(board.has_kingside_castling_rights(False))
    features[768 + 4] = int(board.has_queenside_castling_rights(False))
    features[768 + 5] = int(board.is_check())
    features[768 + 6] = board.halfmove_clock / 100.0
    features[768 + 7] = board.fullmove_number / 100.0
    return features

def get_game_result(game):
    result = game.headers.get("Result", "*")
    if result == "1-0":
        return 1.0
    elif result == "0-1":
        return 0.0
    elif result == "1/2-1/2":
        return 0.5
    else:
        return None

def process_pgn_file(pgn_path, max_games=None, min_moves=10):
    positions = []
    outcomes = []
    try:
        with open(pgn_path, 'r') as pgn_file:
            game_count = 0
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                if max_games is not None and game_count >= max_games:
                    break
                result = get_game_result(game)
                if result is None:
                    continue
                moves = list(game.mainline_moves())
                if len(moves) < min_moves:
                    continue
                board = game.board()
                move_count = 0
                for move in moves:
                    board.push(move)
                    move_count += 1
                    if move_count >= 10 and move_count % 3 == 0:
                        features = board_to_features(board)
                        positions.append(features)
                        adjusted_result = result if board.turn == chess.WHITE else 1.0 - result
                        outcomes.append(adjusted_result)
                game_count += 1
                if game_count % 10 == 0:
                    logger.info(f"Processed {game_count} games")
    except Exception as e:
        logger.error(f"Error processing PGN file: {e}")
    return np.array(positions), np.array(outcomes)

def process_sample_pgn():
    current_dir = os.path.dirname(os.path.dirname(__file__))
    pgn_path = os.path.join(current_dir, "data", "sample_games.pgn")
    if not os.path.exists(pgn_path):
        logger.error(f"Sample PGN file not found at {pgn_path}")
        return np.array([]), np.array([])
    logger.info(f"Processing {pgn_path}")
    positions, outcomes = process_pgn_file(pgn_path)
    if len(positions) > 0:
        logger.info(f"Extracted {len(positions)} positions from sample_games.pgn")
        return positions, outcomes
    else:
        logger.warning("No positions were extracted from the sample file.")
        return np.array([]), np.array([])

def save_processed_data(positions, outcomes, filename="chess_data"):
    current_dir = os.path.dirname(os.path.dirname(__file__))
    output_dir = os.path.join(current_dir, "data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f"{filename}_positions.npy"), positions)
    np.save(os.path.join(output_dir, f"{filename}_outcomes.npy"), outcomes)
    logger.info(f"Saved {len(positions)} processed positions to {output_dir}")

def main():
    logger.info("Starting data preprocessing")
    positions, outcomes = process_sample_pgn()
    if len(positions) > 0:
        logger.info(f"Total positions extracted: {len(positions)}")
        save_processed_data(positions, outcomes)
        logger.info("Data preprocessing completed successfully")
    else:
        logger.warning("No positions were extracted. Check your PGN file.")

if __name__ == "__main__":
    main()
