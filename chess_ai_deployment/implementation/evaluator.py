import torch
import chess
import numpy as np
import os
import logging
import time
from implementation.data_prep import board_to_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChessEvaluator:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "traced_chess_model.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        self.model = torch.jit.load(model_path)
        self.model.eval()
        logger.info(f"Loaded model from {model_path}")
        self.transposition_table = {}
        self.nodes_evaluated = 0
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }

    def evaluate_position(self, board):
        features = board_to_features(board)
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            prediction = self.model(features_tensor).item()
        return prediction

    def get_centipawn_score(self, board):
        raw_score = self.evaluate_position(board)
        centipawns = int((raw_score - 0.5) * 4000)
        return centipawns if board.turn == chess.WHITE else -centipawns

    def get_best_move(self, board, depth=2, use_advanced=False):
        self.nodes_evaluated = 0
        self.transposition_table = {}
        start_time = time.time()
        best_score = float('-inf') if board.turn == chess.WHITE else float('inf')
        best_move = None
        alpha = float('-inf')
        beta = float('inf')
        moves = list(board.legal_moves)
        if use_advanced:
            moves = self._order_moves(board, moves)
        for move in moves:
            board.push(move)
            if use_advanced:
                score = -self._negamax(board, depth-1, -beta, -alpha, -1 if board.turn else 1, use_advanced)
            else:
                if depth > 1:
                    score = self._minimax(board, depth-1, float('-inf'), float('inf'), board.turn != chess.WHITE)
                else:
                    score = self.get_centipawn_score(board)
                    if board.turn != chess.WHITE:
                        score = -score
            board.pop()
            if board.turn == chess.WHITE:
                if score > best_score:
                    best_score = score
                    best_move = move
                    alpha = max(alpha, score)
            else:
                if score < best_score:
                    best_score = score
                    best_move = move
                    beta = min(beta, score)
        elapsed = time.time() - start_time
        logger.info(f"Search completed: depth={depth}, nodes={self.nodes_evaluated}, time={elapsed:.2f}s")
        return best_move

    def _negamax(self, board, depth, alpha, beta, color, use_advanced):
        self.nodes_evaluated += 1
        if board.is_game_over():
            if board.is_checkmate():
                return -10000 * color
            return 0
        board_hash = board.fen().split(' ')[0]
        if use_advanced and board_hash in self.transposition_table and self.transposition_table[board_hash]['depth'] >= depth:
            return self.transposition_table[board_hash]['score']
        if depth <= 0:
            if use_advanced:
                return self._quiescence_search(board, alpha, beta, color)
            else:
                return color * self.get_centipawn_score(board)
        if use_advanced and depth >= 3 and not board.is_check() and self._has_major_pieces(board, board.turn):
            R = 2
            board.push(chess.Move.null())
            null_score = -self._negamax(board, depth-1-R, -beta, -beta+1, -color, use_advanced)
            board.pop()
            if null_score >= beta:
                return beta
        max_score = float('-inf')
        moves = list(board.legal_moves)
        if use_advanced:
            moves = self._order_moves(board, moves)
        for move in moves:
            board.push(move)
            score = -self._negamax(board, depth-1, -beta, -alpha, -color, use_advanced)
            board.pop()
            max_score = max(max_score, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        if use_advanced:
            self.transposition_table[board_hash] = {
                'score': max_score,
                'depth': depth
            }
            if len(self.transposition_table) > 1000000:
                keys = list(self.transposition_table.keys())
                for key in keys[:500000]:
                    del self.transposition_table[key]
        return max_score

    def _quiescence_search(self, board, alpha, beta, color):
        self.nodes_evaluated += 1
        stand_pat = color * self.get_centipawn_score(board)
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat
        for move in board.legal_moves:
            if board.is_capture(move):
                if not self._is_good_capture(board, move):
                    continue
                board.push(move)
                score = -self._quiescence_search(board, -beta, -alpha, -color)
                board.pop()
                if score >= beta:
                    return beta
                if score > alpha:
                    alpha = score
        return alpha

    def _is_good_capture(self, board, move):
        to_square = move.to_square
        captured_piece = board.piece_at(to_square)
        if not captured_piece:
            if board.is_en_passant(move):
                return True
            return False
        captured_value = self.piece_values.get(captured_piece.piece_type, 0)
        from_square = move.from_square
        capturing_piece = board.piece_at(from_square)
        capturing_value = self.piece_values.get(capturing_piece.piece_type, 0)
        if captured_value >= capturing_value:
            return True
        board.push(move)
        is_attacked = board.is_attacked_by(not board.turn, to_square)
        board.pop()
        if is_attacked:
            return False
        return True

    def _has_major_pieces(self, board, color):
        for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            if len(board.pieces(piece_type, color)) > 0:
                return True
        return False

    def _order_moves(self, board, moves):
        move_scores = []
        for move in moves:
            score = 0
            if board.is_capture(move):
                to_square = move.to_square
                from_square = move.from_square
                captured_piece = board.piece_at(to_square)
                if captured_piece:
                    captured_value = self.piece_values.get(captured_piece.piece_type, 0)
                    capturing_piece = board.piece_at(from_square)
                    capturing_value = self.piece_values.get(capturing_piece.piece_type, 0)
                    score = 10 * captured_value - capturing_value
                else:
                    score = 10 * self.piece_values[chess.PAWN]
            if move.promotion:
                score += self.piece_values.get(move.promotion, 0)
            board.push(move)
            if board.is_check():
                score += 50
            board.pop()
            move_scores.append((move, score))
        move_scores.sort(key=lambda x: x[1], reverse=True)
        return [move for move, _ in move_scores]

    def _minimax(self, board, depth, alpha, beta, maximizing):
        self.nodes_evaluated += 1
        if depth == 0 or board.is_game_over():
            return self.get_centipawn_score(board)
        if maximizing:
            max_eval = float('-inf')
            for move in board.legal_moves:
                board.push(move)
                eval = self._minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in board.legal_moves:
                board.push(move)
                eval = self._minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

def main():
    board = chess.Board()
    moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"]
    for move_uci in moves:
        board.push_uci(move_uci)
    print("Current position:")
    print(board)
    evaluator = ChessEvaluator()
    score = evaluator.get_centipawn_score(board)
    print(f"Position evaluation: {score} centipawns")
    best_move = evaluator.get_best_move(board, depth=4, use_advanced=True)
    print(f"Best move: {best_move}")

if __name__ == "__main__":
    main()
