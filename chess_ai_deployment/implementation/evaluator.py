import torch
import chess
import numpy as np
import os
import logging
import time
from implementation.data_prep import board_to_features

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChessEvaluator:
    """Chess position evaluator using a trained neural network."""
    
    def __init__(self, model_path=None):
        """Initialize the evaluator with a trained model.
        
        Args:
            model_path: Path to the trained model file
        """
        if model_path is None:
            # Use relative path for deployment
            model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "traced_chess_model.pt")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        # Load the model
        self.model = torch.jit.load(model_path)
        self.model.eval()
        logger.info(f"Loaded model from {model_path}")
        
        # Initialize transposition table
        self.transposition_table = {}
        self.nodes_evaluated = 0
        
        # Piece values for quiescence search
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
    
    def evaluate_position(self, board):
        """Evaluate a chess position.
        
        Args:
            board: A chess.Board object
            
        Returns:
            float: Evaluation score between 0 and 1 (higher is better for white)
        """
        # Convert board to features
        features = board_to_features(board)
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        
        # Get prediction from model
        with torch.no_grad():
            prediction = self.model(features_tensor).item()
        
        return prediction
    
    def get_centipawn_score(self, board):
        """Get evaluation in centipawns.
        
        Args:
            board: A chess.Board object
            
        Returns:
            int: Evaluation in centipawns (positive for white, negative for black)
        """
        raw_score = self.evaluate_position(board)
        
        # Convert from [0,1] to centipawns [-2000, 2000]
        # 0.5 is equal position (0 centipawns)
        centipawns = int((raw_score - 0.5) * 4000)
        
        # Return score from perspective of current player
        return centipawns if board.turn == chess.WHITE else -centipawns
    
    def get_best_move(self, board, depth=2, use_advanced=False):
        """Find the best move using the neural network evaluation.
        
        Args:
            board: A chess.Board object
            depth: Search depth
            use_advanced: Whether to use advanced search techniques
            
        Returns:
            chess.Move: The best move
        """
        self.nodes_evaluated = 0
        self.transposition_table = {}  # Reset transposition table
        start_time = time.time()
        
        best_score = float('-inf') if board.turn == chess.WHITE else float('inf')
        best_move = None
        alpha = float('-inf')
        beta = float('inf')
        
        # Order moves to improve alpha-beta pruning
        moves = list(board.legal_moves)
        if use_advanced:
            moves = self._order_moves(board, moves)
        
        for move in moves:
            board.push(move)
            
            if use_advanced:
                # Use advanced search with alpha-beta, null move, and quiescence
                score = -self._negamax(board, depth-1, -beta, -alpha, -1 if board.turn else 1, use_advanced)
            else:
                # Use simple minimax
                if depth > 1:
                    score = self._minimax(board, depth-1, float('-inf'), float('inf'), board.turn != chess.WHITE)
                else:
                    score = self.get_centipawn_score(board)
                    if board.turn != chess.WHITE:
                        score = -score
            
            board.pop()
            
            # Update best move
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
        """Negamax search with alpha-beta pruning and advanced techniques.
        
        Args:
            board: A chess.Board object
            depth: Remaining search depth
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            color: 1 for white, -1 for black
            use_advanced: Whether to use advanced techniques
            
        Returns:
            float: Evaluation score
        """
        self.nodes_evaluated += 1
        
        # Check for game over
        if board.is_game_over():
            if board.is_checkmate():
                return -10000 * color  # Worst possible score
            return 0  # Draw
        
        # Check transposition table
        board_hash = board.fen().split(' ')[0]  # Use position without move counters
        if use_advanced and board_hash in self.transposition_table and self.transposition_table[board_hash]['depth'] >= depth:
            return self.transposition_table[board_hash]['score']
        
        # Base case: leaf node
        if depth <= 0:
            if use_advanced:
                # Use quiescence search at leaf nodes
                return self._quiescence_search(board, alpha, beta, color)
            else:
                return color * self.get_centipawn_score(board)
        
        # Null move pruning (skip for endgames, in check, or at low depths)
        if use_advanced and depth >= 3 and not board.is_check() and self._has_major_pieces(board, board.turn):
            R = 2  # Reduction factor
            board.push(chess.Move.null())
            null_score = -self._negamax(board, depth-1-R, -beta, -beta+1, -color, use_advanced)
            board.pop()
            
            if null_score >= beta:
                return beta  # Fail-high, pruning
        
        # Regular alpha-beta search
        max_score = float('-inf')
        
        # Order moves to improve alpha-beta pruning
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
                break  # Beta cutoff
        
        # Store position in transposition table
        if use_advanced:
            self.transposition_table[board_hash] = {
                'score': max_score,
                'depth': depth
            }
            
            # Keep table size reasonable
            if len(self.transposition_table) > 1000000:  # 1M entries max
                # Clear half the table when it gets too large
                keys = list(self.transposition_table.keys())
                for key in keys[:500000]:
                    del self.transposition_table[key]
        
        return max_score
    
    def _quiescence_search(self, board, alpha, beta, color):
        """Quiescence search to handle tactical sequences.
        
        Args:
            board: A chess.Board object
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            color: 1 for white, -1 for black
            
        Returns:
            float: Evaluation score
        """
        self.nodes_evaluated += 1
        
        # Get static evaluation
        stand_pat = color * self.get_centipawn_score(board)
        
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat
        
        # Only consider captures to reduce search space
        for move in board.legal_moves:
            if board.is_capture(move):
                # Skip bad captures early
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
        """Check if a capture is likely to be good (for move ordering).
        
        Args:
            board: A chess.Board object
            move: A chess.Move object
            
        Returns:
            bool: True if the capture is likely good
        """
        # Get the captured piece value
        to_square = move.to_square
        captured_piece = board.piece_at(to_square)
        if not captured_piece:
            # En passant capture
            if board.is_en_passant(move):
                return True
            return False
            
        captured_value = self.piece_values.get(captured_piece.piece_type, 0)
        
        # Get the capturing piece value
        from_square = move.from_square
        capturing_piece = board.piece_at(from_square)
        capturing_value = self.piece_values.get(capturing_piece.piece_type, 0)
        
        # Simple MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
        # If capturing a more valuable piece, it's likely good
        if captured_value >= capturing_value:
            return True
            
        # Check if the square is defended
        board.push(move)
        is_attacked = board.is_attacked_by(not board.turn, to_square)
        board.pop()
        
        # If capturing a less valuable piece and the square is defended, likely bad
        if is_attacked:
            return False
            
        return True
    
    def _has_major_pieces(self, board, color):
        """Check if side has any major pieces (for null move pruning).
        
        Args:
            board: A chess.Board object
            color: chess.WHITE or chess.BLACK
            
        Returns:
            bool: True if the side has any major pieces
        """
        for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            if len(board.pieces(piece_type, color)) > 0:
                return True
        return False
    
    def _order_moves(self, board, moves):
        """Order moves to improve alpha-beta pruning efficiency.
        
        Args:
            board: A chess.Board object
            moves: List of legal moves
            
        Returns:
            list: Ordered list of moves
        """
        move_scores = []
        
        for move in moves:
            score = 0
            
            # Captures are likely good
            if board.is_capture(move):
                to_square = move.to_square
                from_square = move.from_square
                
                # Get captured piece value
                captured_piece = board.piece_at(to_square)
                if captured_piece:
                    captured_value = self.piece_values.get(captured_piece.piece_type, 0)
                    capturing_piece = board.piece_at(from_square)
                    capturing_value = self.piece_values.get(capturing_piece.piece_type, 0)
                    
                    # MVV-LVA scoring
                    score = 10 * captured_value - capturing_value
                else:
                    # En passant capture
                    score = 10 * self.piece_values[chess.PAWN]
            
            # Promotions are good
            if move.promotion:
                score += self.piece_values.get(move.promotion, 0)
            
            # Check if the move gives check
            board.push(move)
            if board.is_check():
                score += 50
            board.pop()
            
            move_scores.append((move, score))
        
        # Sort moves by score (descending)
        move_scores.sort(key=lambda x: x[1], reverse=True)
        return [move for move, _ in move_scores]
    
    def _minimax(self, board, depth, alpha, beta, maximizing):
        """Minimax algorithm with alpha-beta pruning.
        
        Args:
            board: A chess.Board object
            depth: Remaining search depth
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            maximizing: Whether to maximize or minimize
            
        Returns:
            float: Evaluation score
        """
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
    """Demo the chess evaluator."""
    # Create a board from the starting position
    board = chess.Board()
    
    # Make some moves
    moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"]
    for move_uci in moves:
        board.push_uci(move_uci)
    
    print("Current position:")
    print(board)
    
    # Create evaluator
    evaluator = ChessEvaluator()
    
    # Get evaluation
    score = evaluator.get_centipawn_score(board)
    print(f"Position evaluation: {score} centipawns")
    
    # Get best move
    best_move = evaluator.get_best_move(board, depth=4, use_advanced=True)
    print(f"Best move: {best_move.uci()} ({board.san(best_move)})")
    
    # Compare with standard search
    standard_move = evaluator.get_best_move(board, depth=3, use_advanced=False)
    print(f"Standard search move: {standard_move.uci()} ({board.san(standard_move)})")

if __name__ == "__main__":
    main()
