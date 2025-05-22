import torch
import chess
import numpy as np
import os
import logging
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
    
    def get_best_move(self, board, depth=2):
        """Find the best move using the neural network evaluation.
        
        Args:
            board: A chess.Board object
            depth: Search depth
            
        Returns:
            chess.Move: The best move
        """
        best_score = float('-inf') if board.turn == chess.WHITE else float('inf')
        best_move = None
        
        for move in board.legal_moves:
            board.push(move)
            
            if depth > 1:
                # Recursive minimax search
                score = self._minimax(board, depth - 1, float('-inf'), float('inf'), board.turn != chess.WHITE)
            else:
                # Direct evaluation
                score = self.get_centipawn_score(board)
                if board.turn != chess.WHITE:
                    score = -score
            
            board.pop()
            
            # Update best move
            if board.turn == chess.WHITE:
                if score > best_score:
                    best_score = score
                    best_move = move
            else:
                if score < best_score:
                    best_score = score
                    best_move = move
        
        return best_move
    
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
    best_move = evaluator.get_best_move(board)
    print(f"Best move: {best_move.uci()} ({board.san(best_move)})")

if __name__ == "__main__":
    main()