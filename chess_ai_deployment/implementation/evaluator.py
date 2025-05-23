import torch
import chess
import numpy as np
import os
import logging
import time
import math
import threading
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from implementation.data_prep import board_to_features

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TranspositionEntry:
    """Entry in the transposition table with enhanced data."""
    score: float
    depth: int
    flag: str  # 'exact', 'lower_bound', 'upper_bound'
    best_move: Optional[chess.Move]
    age: int

class ChessEvaluator:
    """Enhanced Chess position evaluator using advanced AI techniques."""
    
    def __init__(self, model_path=None):
        """Initialize the evaluator with a trained model.
        
        Args:
            model_path: Path to the trained model file
        """
        if model_path is None:
            model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "traced_chess_model.pt")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        # Load the model
        self.model = torch.jit.load(model_path)
        self.model.eval()
        logger.info(f"Loaded model from {model_path}")
        
        # Enhanced transposition table with aging
        self.transposition_table: Dict[str, TranspositionEntry] = {}
        self.tt_age = 0
        self.nodes_evaluated = 0
        self.time_limit = None
        self.start_time = None
        
        # Killer moves heuristic (non-capture moves that cause beta cutoffs)
        self.killer_moves = [[None, None] for _ in range(64)]  # 2 killer moves per depth
        
        # History heuristic (move ordering)
        self.history_table = {}
        
        # Piece-Square Tables for positional evaluation
        self.piece_square_tables = self._initialize_piece_square_tables()
        
        # Advanced piece values with positional considerations
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
        
        # Endgame piece values (different from middlegame)
        self.endgame_piece_values = {
            chess.PAWN: 120,
            chess.KNIGHT: 290,
            chess.BISHOP: 320,
            chess.ROOK: 550,
            chess.QUEEN: 1000,
            chess.KING: 20000
        }
        
        # Contempt factor (to avoid draws)
        self.contempt = 15
        
        # Late move reduction parameters
        self.lmr_depth_threshold = 3
        self.lmr_move_threshold = 4
    
    def _initialize_piece_square_tables(self):
        """Initialize piece-square tables for positional evaluation."""
        # Pawn table (encourages center control and advancement)
        pawn_table = np.array([
            [ 0,  0,  0,  0,  0,  0,  0,  0],
            [50, 50, 50, 50, 50, 50, 50, 50],
            [10, 10, 20, 30, 30, 20, 10, 10],
            [ 5,  5, 10, 25, 25, 10,  5,  5],
            [ 0,  0,  0, 20, 20,  0,  0,  0],
            [ 5, -5,-10,  0,  0,-10, -5,  5],
            [ 5, 10, 10,-20,-20, 10, 10,  5],
            [ 0,  0,  0,  0,  0,  0,  0,  0]
        ])
        
        # Knight table (encourages central squares)
        knight_table = np.array([
            [-50,-40,-30,-30,-30,-30,-40,-50],
            [-40,-20,  0,  0,  0,  0,-20,-40],
            [-30,  0, 10, 15, 15, 10,  0,-30],
            [-30,  5, 15, 20, 20, 15,  5,-30],
            [-30,  0, 15, 20, 20, 15,  0,-30],
            [-30,  5, 10, 15, 15, 10,  5,-30],
            [-40,-20,  0,  5,  5,  0,-20,-40],
            [-50,-40,-30,-30,-30,-30,-40,-50]
        ])
        
        # Bishop table (encourages long diagonals)
        bishop_table = np.array([
            [-20,-10,-10,-10,-10,-10,-10,-20],
            [-10,  0,  0,  0,  0,  0,  0,-10],
            [-10,  0,  5, 10, 10,  5,  0,-10],
            [-10,  5,  5, 10, 10,  5,  5,-10],
            [-10,  0, 10, 10, 10, 10,  0,-10],
            [-10, 10, 10, 10, 10, 10, 10,-10],
            [-10,  5,  0,  0,  0,  0,  5,-10],
            [-20,-10,-10,-10,-10,-10,-10,-20]
        ])
        
        # Rook table (encourages 7th rank and open files)
        rook_table = np.array([
            [ 0,  0,  0,  0,  0,  0,  0,  0],
            [ 5, 10, 10, 10, 10, 10, 10,  5],
            [-5,  0,  0,  0,  0,  0,  0, -5],
            [-5,  0,  0,  0,  0,  0,  0, -5],
            [-5,  0,  0,  0,  0,  0,  0, -5],
            [-5,  0,  0,  0,  0,  0,  0, -5],
            [-5,  0,  0,  0,  0,  0,  0, -5],
            [ 0,  0,  0,  5,  5,  0,  0,  0]
        ])
        
        # Queen table (slightly encourages center)
        queen_table = np.array([
            [-20,-10,-10, -5, -5,-10,-10,-20],
            [-10,  0,  0,  0,  0,  0,  0,-10],
            [-10,  0,  5,  5,  5,  5,  0,-10],
            [ -5,  0,  5,  5,  5,  5,  0, -5],
            [  0,  0,  5,  5,  5,  5,  0, -5],
            [-10,  5,  5,  5,  5,  5,  0,-10],
            [-10,  0,  5,  0,  0,  0,  0,-10],
            [-20,-10,-10, -5, -5,-10,-10,-20]
        ])
        
        # King table for middlegame (encourages castling)
        king_middle_table = np.array([
            [-30,-40,-40,-50,-50,-40,-40,-30],
            [-30,-40,-40,-50,-50,-40,-40,-30],
            [-30,-40,-40,-50,-50,-40,-40,-30],
            [-30,-40,-40,-50,-50,-40,-40,-30],
            [-20,-30,-30,-40,-40,-30,-30,-20],
            [-10,-20,-20,-20,-20,-20,-20,-10],
            [ 20, 20,  0,  0,  0,  0, 20, 20],
            [ 20, 30, 10,  0,  0, 10, 30, 20]
        ])
        
        # King table for endgame (encourages centralization)
        king_end_table = np.array([
            [-50,-40,-30,-20,-20,-30,-40,-50],
            [-30,-20,-10,  0,  0,-10,-20,-30],
            [-30,-10, 20, 30, 30, 20,-10,-30],
            [-30,-10, 30, 40, 40, 30,-10,-30],
            [-30,-10, 30, 40, 40, 30,-10,-30],
            [-30,-10, 20, 30, 30, 20,-10,-30],
            [-30,-30,  0,  0,  0,  0,-30,-30],
            [-50,-30,-30,-30,-30,-30,-30,-50]
        ])
        
        return {
            chess.PAWN: pawn_table,
            chess.KNIGHT: knight_table,
            chess.BISHOP: bishop_table,
            chess.ROOK: rook_table,
            chess.QUEEN: queen_table,
            chess.KING: (king_middle_table, king_end_table)
        }
    
    def evaluate_position(self, board):
        """Enhanced position evaluation combining neural network with classical techniques."""
        # Get neural network evaluation
        features = board_to_features(board)
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            nn_eval = self.model(features_tensor).item()
        
        # Classical evaluation components
        classical_eval = self._classical_evaluation(board)
        
        # Combine evaluations (70% neural network, 30% classical)
        combined_eval = 0.7 * nn_eval + 0.3 * (classical_eval / 4000 + 0.5)
        
        # Ensure result is in [0, 1] range
        return max(0.0, min(1.0, combined_eval))
    
    def _classical_evaluation(self, board):
        """Classical chess evaluation with multiple factors."""
        score = 0
        
        # Material evaluation
        score += self._evaluate_material(board)
        
        # Positional evaluation
        score += self._evaluate_position_classical(board)
        
        # King safety
        score += self._evaluate_king_safety(board)
        
        # Pawn structure
        score += self._evaluate_pawn_structure(board)
        
        # Piece activity
        score += self._evaluate_piece_activity(board)
        
        return score
    
    def _evaluate_material(self, board):
        """Evaluate material balance with phase consideration."""
        score = 0
        total_material = 0
        
        # Count material for both sides
        for color in [chess.WHITE, chess.BLACK]:
            color_material = 0
            for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
                count = len(board.pieces(piece_type, color))
                piece_value = self.piece_values[piece_type]
                color_material += count * piece_value
                
                if color == chess.WHITE:
                    score += count * piece_value
                else:
                    score -= count * piece_value
            
            total_material += color_material
        
        # Determine game phase (opening/middlegame/endgame)
        self.game_phase = min(1.0, total_material / 7800)  # Normalize to [0, 1]
        
        return score
    
    def _evaluate_position_classical(self, board):
        """Evaluate positional factors using piece-square tables."""
        score = 0
        is_endgame = self.game_phase < 0.3
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                piece_type = piece.piece_type
                color = piece.color
                
                # Get piece-square table value
                row = 7 - (square // 8) if color == chess.WHITE else square // 8
                col = square % 8
                
                if piece_type == chess.KING:
                    # Use appropriate king table based on game phase
                    table = self.piece_square_tables[piece_type][1 if is_endgame else 0]
                else:
                    table = self.piece_square_tables[piece_type]
                
                table_value = table[row][col]
                
                if color == chess.WHITE:
                    score += table_value
                else:
                    score -= table_value
        
        return score
    
    def _evaluate_king_safety(self, board):
        """Evaluate king safety."""
        score = 0
        
        for color in [chess.WHITE, chess.BLACK]:
            king_square = board.king(color)
            if king_square is None:
                continue
            
            # Penalize exposed king in middlegame
            if self.game_phase > 0.5:
                # Count attackers near king
                king_zone = []
                king_file = chess.square_file(king_square)
                king_rank = chess.square_rank(king_square)
                
                for file_offset in [-1, 0, 1]:
                    for rank_offset in [-1, 0, 1]:
                        new_file = king_file + file_offset
                        new_rank = king_rank + rank_offset
                        
                        if 0 <= new_file <= 7 and 0 <= new_rank <= 7:
                            king_zone.append(chess.square(new_file, new_rank))
                
                attackers = 0
                for square in king_zone:
                    if board.is_attacked_by(not color, square):
                        attackers += 1
                
                king_safety_penalty = attackers * 20
                
                if color == chess.WHITE:
                    score -= king_safety_penalty
                else:
                    score += king_safety_penalty
        
        return score
    
    def _evaluate_pawn_structure(self, board):
        """Evaluate pawn structure (doubled, isolated, passed pawns)."""
        score = 0
        
        for color in [chess.WHITE, chess.BLACK]:
            pawns = board.pieces(chess.PAWN, color)
            
            # Count pawns per file
            files = [0] * 8
            for pawn_square in pawns:
                file = chess.square_file(pawn_square)
                files[file] += 1
            
            # Penalize doubled pawns
            for file_count in files:
                if file_count > 1:
                    doubled_penalty = (file_count - 1) * 15
                    if color == chess.WHITE:
                        score -= doubled_penalty
                    else:
                        score += doubled_penalty
            
            # Penalize isolated pawns
            for file, count in enumerate(files):
                if count > 0:
                    has_neighbor = False
                    if file > 0 and files[file - 1] > 0:
                        has_neighbor = True
                    if file < 7 and files[file + 1] > 0:
                        has_neighbor = True
                    
                    if not has_neighbor:
                        isolated_penalty = 20
                        if color == chess.WHITE:
                            score -= isolated_penalty
                        else:
                            score += isolated_penalty
            
            # Bonus for passed pawns
            for pawn_square in pawns:
                if self._is_passed_pawn(board, pawn_square, color):
                    passed_bonus = 30
                    if color == chess.WHITE:
                        score += passed_bonus
                    else:
                        score -= passed_bonus
        
        return score
    
    def _is_passed_pawn(self, board, pawn_square, color):
        """Check if a pawn is passed."""
        file = chess.square_file(pawn_square)
        rank = chess.square_rank(pawn_square)
        
        # Check files in front of the pawn
        direction = 1 if color == chess.WHITE else -1
        
        for check_file in [file - 1, file, file + 1]:
            if 0 <= check_file <= 7:
                for check_rank in range(rank + direction, 8 if direction > 0 else -1, direction):
                    if 0 <= check_rank <= 7:
                        check_square = chess.square(check_file, check_rank)
                        piece = board.piece_at(check_square)
                        if piece and piece.piece_type == chess.PAWN and piece.color != color:
                            return False
        
        return True
    
    def _evaluate_piece_activity(self, board):
        """Evaluate piece mobility and activity."""
        score = 0
        
        for color in [chess.WHITE, chess.BLACK]:
            mobility = 0
            
            # Count legal moves (rough mobility measure)
            temp_board = board.copy()
            temp_board.turn = color
            
            legal_moves = list(temp_board.legal_moves)
            mobility = len(legal_moves)
            
            # Bonus for higher mobility
            mobility_bonus = mobility * 2
            
            if color == chess.WHITE:
                score += mobility_bonus
            else:
                score -= mobility_bonus
        
        return score
    
    def get_centipawn_score(self, board):
        """Get evaluation in centipawns."""
        raw_score = self.evaluate_position(board)
        
        # Convert from [0,1] to centipawns with enhanced range
        centipawns = int((raw_score - 0.5) * 5000)  # Increased range for more dramatic evaluations
        
        # Add contempt factor to avoid draws
        if abs(centipawns) < 50:
            contempt_adjustment = self.contempt if board.turn == chess.WHITE else -self.contempt
            centipawns += contempt_adjustment
        
        return centipawns if board.turn == chess.WHITE else -centipawns
    
    def get_best_move(self, board, depth=2, use_advanced=False, time_limit=None):
        """Find the best move using advanced search techniques."""
        self.nodes_evaluated = 0
        self.start_time = time.time()
        self.time_limit = time_limit
        self.tt_age += 1
        
        # Clear old transposition table entries periodically
        if len(self.transposition_table) > 500000:
            self._clean_transposition_table()
        
        if use_advanced and depth >= 8:
            # Use iterative deepening with aspiration windows for expert mode
            return self._iterative_deepening_search(board, depth)
        else:
            # Standard search for lower difficulties
            return self._alpha_beta_search(board, depth, use_advanced)
    
    def _iterative_deepening_search(self, board, max_depth):
        """Iterative deepening with aspiration windows - used for expert mode."""
        best_move = None
        best_score = 0
        
        # Start with a reasonable aspiration window
        window_size = 50
        
        for current_depth in range(1, max_depth + 1):
            if self._is_time_up():
                break
            
            alpha = best_score - window_size
            beta = best_score + window_size
            
            # Search with aspiration window
            score, move = self._aspiration_search(board, current_depth, alpha, beta)
            
            if move:
                best_move = move
                best_score = score
                
                logger.info(f"Depth {current_depth}: Best move {move.uci()}, Score: {score}")
            
            # Adjust window size based on whether we had a fail-high or fail-low
            if score <= alpha or score >= beta:
                window_size = min(window_size * 2, 500)  # Widen window
            else:
                window_size = max(window_size // 2, 25)  # Narrow window
        
        elapsed = time.time() - self.start_time
        logger.info(f"Iterative deepening completed: depth={max_depth}, nodes={self.nodes_evaluated}, time={elapsed:.2f}s")
        
        return best_move
    
    def _aspiration_search(self, board, depth, alpha, beta):
        """Search with aspiration window."""
        best_move = None
        best_score = float('-inf') if board.turn == chess.WHITE else float('inf')
        
        moves = list(board.legal_moves)
        moves = self._order_moves_advanced(board, moves, depth)
        
        for i, move in enumerate(moves):
            if self._is_time_up():
                break
            
            board.push(move)
            
            # Use principal variation search for non-first moves
            if i == 0:
                score = -self._negamax_advanced(board, depth - 1, -beta, -alpha, -1 if board.turn else 1)
            else:
                # Try with null window first (PVS)
                score = -self._negamax_advanced(board, depth - 1, -alpha - 1, -alpha, -1 if board.turn else 1)
                
                # If it fails high, do a full search
                if alpha < score < beta:
                    score = -self._negamax_advanced(board, depth - 1, -beta, -score, -1 if board.turn else 1)
            
            board.pop()
            
            if board.turn == chess.WHITE:
                if score > best_score:
                    best_score = score
                    best_move = move
                    alpha = max(alpha, score)
                    
                    # Update transposition table with best move
                    self._store_tt_entry(board, depth, score, 'exact', move)
            else:
                if score < best_score:
                    best_score = score
                    best_move = move
                    beta = min(beta, score)
                    
                    self._store_tt_entry(board, depth, score, 'exact', move)
            
            if alpha >= beta:
                break
        
        return best_score, best_move
    
    def _negamax_advanced(self, board, depth, alpha, beta, color):
        """Advanced negamax with multiple enhancements."""
        if self._is_time_up():
            return color * self.get_centipawn_score(board)
        
        self.nodes_evaluated += 1
        original_alpha = alpha
        
        # Check transposition table
        tt_entry = self._probe_tt(board, depth, alpha, beta)
        if tt_entry is not None:
            return tt_entry
        
        # Check for terminal positions
        if board.is_game_over():
            if board.is_checkmate():
                return -10000 * color + depth  # Prefer faster mates
            return 0  # Draw
        
        # Quiescence search at leaf nodes
        if depth <= 0:
            return self._quiescence_search_advanced(board, alpha, beta, color)
        
        # Null move pruning with verification
        if (depth >= 3 and not board.is_check() and 
            self._has_major_pieces(board, board.turn) and
            not self._is_pv_node(alpha, beta)):
            
            R = 2 + depth // 4  # Adaptive reduction
            board.push(chess.Move.null())
            null_score = -self._negamax_advanced(board, depth - 1 - R, -beta, -beta + 1, -color)
            board.pop()
            
            if null_score >= beta:
                # Verify with reduced depth to avoid zugzwang
                verify_score = self._negamax_advanced(board, depth - R, alpha, beta, color)
                if verify_score >= beta:
                    return beta
        
        # Internal iterative deepening for move ordering
        if depth >= 4 and not self._has_tt_move(board):
            self._negamax_advanced(board, depth - 2, alpha, beta, color)
        
        # Move generation and ordering
        moves = list(board.legal_moves)
        moves = self._order_moves_advanced(board, moves, depth)
        
        best_score = float('-inf')
        best_move = None
        moves_searched = 0
        
        for move in moves:
            if self._is_time_up():
                break
            
            board.push(move)
            
            # Late move reductions
            reduction = 0
            if (moves_searched >= self.lmr_move_threshold and 
                depth >= self.lmr_depth_threshold and 
                not board.is_check() and 
                not board.is_capture(move) and 
                move not in self.killer_moves[min(depth, 63)]):
                
                reduction = 1 + moves_searched // 4
                reduction = min(reduction, depth - 1)
            
            # Principal variation search
            if moves_searched == 0:
                score = -self._negamax_advanced(board, depth - 1 - reduction, -beta, -alpha, -color)
            else:
                # Search with null window
                score = -self._negamax_advanced(board, depth - 1 - reduction, -alpha - 1, -alpha, -color)
                
                # If it fails high and we used reduction, search again without reduction
                if score > alpha and reduction > 0:
                    score = -self._negamax_advanced(board, depth - 1, -alpha - 1, -alpha, -color)
                
                # If it still fails high, do full window search
                if score > alpha and score < beta:
                    score = -self._negamax_advanced(board, depth - 1, -beta, -alpha, -color)
            
            board.pop()
            moves_searched += 1
            
            if score > best_score:
                best_score = score
                best_move = move
            
            alpha = max(alpha, score)
            
            if alpha >= beta:
                # Update killer moves and history
                if not board.is_capture(move):
                    self._update_killer_moves(move, depth)
                    self._update_history(move, depth)
                break
        
        # Store in transposition table
        flag = 'exact'
        if best_score <= original_alpha:
            flag = 'upper_bound'
        elif best_score >= beta:
            flag = 'lower_bound'
        
        self._store_tt_entry(board, depth, best_score, flag, best_move)
        
        return best_score
    
    def _quiescence_search_advanced(self, board, alpha, beta, color):
        """Enhanced quiescence search with more selective captures."""
        if self._is_time_up():
            return color * self.get_centipawn_score(board)
        
        self.nodes_evaluated += 1
        
        stand_pat = color * self.get_centipawn_score(board)
        
        if stand_pat >= beta:
            return beta
        
        # Delta pruning - don't consider captures that can't improve position significantly
        delta = 900  # Queen value
        if stand_pat < alpha - delta:
            return alpha
        
        alpha = max(alpha, stand_pat)
        
        # Generate and order only promising captures
        captures = []
        for move in board.legal_moves:
            if board.is_capture(move) or move.promotion:
                if self._is_good_capture_advanced(board, move):
                    captures.append(move)
        
        # Order captures by MVV-LVA
        captures.sort(key=lambda m: self._get_capture_score(board, m), reverse=True)
        
        for move in captures:
            if self._is_time_up():
                break
            
            board.push(move)
            score = -self._quiescence_search_advanced(board, -beta, -alpha, -color)
            board.pop()
            
            if score >= beta:
                return beta
            alpha = max(alpha, score)
        
        return alpha
    
    def _is_good_capture_advanced(self, board, move):
        """Enhanced capture evaluation using SEE (Static Exchange Evaluation)."""
        return self._see(board, move) >= 0
    
    def _see(self, board, move):
        """Static Exchange Evaluation."""
        to_square = move.to_square
        from_square = move.from_square
        
        # Get the piece being captured
        captured = board.piece_at(to_square)
        if captured is None and not board.is_en_passant(move):
            return 0
        
        # Initialize gain array
        gain = [0] * 32
        depth = 0
        
        # Value of captured piece
        if board.is_en_passant(move):
            gain[depth] = self.piece_values[chess.PAWN]
        elif captured:
            gain[depth] = self.piece_values[captured.piece_type]
        else:
            gain[depth] = 0
        
        # Add promotion value
        if move.promotion:
            gain[depth] += self.piece_values[move.promotion] - self.piece_values[chess.PAWN]
        
        # Simulate the exchange
        attacking_piece = board.piece_at(from_square)
        current_attacker_value = self.piece_values[attacking_piece.piece_type]
        
        # This is a simplified SEE - full implementation would be more complex
        # For now, just use basic material exchange
        if gain[depth] >= current_attacker_value:
            return gain[depth] - current_attacker_value
        else:
            return gain[depth]
    
    def _get_capture_score(self, board, move):
        """Get score for capture move ordering."""
        to_square = move.to_square
        from_square = move.from_square
        
        score = 0
        
        # Captured piece value
        captured = board.piece_at(to_square)
        if captured:
            score += self.piece_values[captured.piece_type] * 10
        elif board.is_en_passant(move):
            score += self.piece_values[chess.PAWN] * 10
        
        # Subtract attacker value (MVV-LVA)
        attacker = board.piece_at(from_square)
        if attacker:
            score -= self.piece_values[attacker.piece_type]
        
        # Promotion bonus
        if move.promotion:
            score += self.piece_values[move.promotion] * 8
        
        return score
    
    def _order_moves_advanced(self, board, moves, depth):
        """Advanced move ordering with multiple heuristics."""
        move_scores = []
        
        # Get TT move
        tt_move = self._get_tt_move(board)
        
        for move in moves:
            score = 0
            
            # TT move gets highest priority
            if tt_move and move == tt_move:
                score += 10000
            
            # Captures and promotions
            elif board.is_capture(move) or move.promotion:
                score += self._get_capture_score(board, move) + 5000
            
            # Killer moves
            elif move in self.killer_moves[min(depth, 63)]:
                if move == self.killer_moves[min(depth, 63)][0]:
                    score += 4000  # First killer move
                else:
                    score += 3900  # Second killer move
            
            # History heuristic
            else:
                move_key = (move.from_square, move.to_square)
                score += self.history_table.get(move_key, 0)
            
            # Bonus for checks
            board.push(move)
            if board.is_check():
                score += 100
            board.pop()
            
            # Penalty for moving pieces to attacked squares (simple)
            if board.is_attacked_by(not board.turn, move.to_square):
                attacking_piece = board.piece_at(move.from_square)
                if attacking_piece:
                    score -= self.piece_values[attacking_piece.piece_type] // 10
            
            move_scores.append((move, score))
        
        # Sort by score (descending)
        move_scores.sort(key=lambda x: x[1], reverse=True)
        return [move for move, _ in move_scores]
    
    def _update_killer_moves(self, move, depth):
        """Update killer moves table."""
        depth = min(depth, 63)
        if move != self.killer_moves[depth][0]:
            self.killer_moves[depth][1] = self.killer_moves[depth][0]
            self.killer_moves[depth][0] = move
    
    def _update_history(self, move, depth):
        """Update history heuristic table."""
        move_key = (move.from_square, move.to_square)
        self.history_table[move_key] = self.history_table.get(move_key, 0) + depth * depth
    
    def _probe_tt(self, board, depth, alpha, beta):
        """Probe transposition table."""
        board_hash = self._get_board_hash(board)
        
        if board_hash in self.transposition_table:
            entry = self.transposition_table[board_hash]
            
            if entry.depth >= depth:
                if entry.flag == 'exact':
                    return entry.score
                elif entry.flag == 'lower_bound' and entry.score >= beta:
                    return beta
                elif entry.flag == 'upper_bound' and entry.score <= alpha:
                    return alpha
        
        return None
    
    def _store_tt_entry(self, board, depth, score, flag, best_move):
        """Store entry in transposition table."""
        board_hash = self._get_board_hash(board)
        
        # Always replace or store if not present
        self.transposition_table[board_hash] = TranspositionEntry(
            score=score,
            depth=depth,
            flag=flag,
            best_move=best_move,
            age=self.tt_age
        )
    
    def _get_board_hash(self, board):
        """Get hash for board position."""
        return board.fen().split(' ')[0]  # Position without move counters
    
    def _get_tt_move(self, board):
        """Get best move from transposition table."""
        board_hash = self._get_board_hash(board)
        if board_hash in self.transposition_table:
            return self.transposition_table[board_hash].best_move
        return None
    
    def _has_tt_move(self, board):
        """Check if position has a move in transposition table."""
        return self._get_tt_move(board) is not None
    
    def _clean_transposition_table(self):
        """Clean old entries from transposition table."""
        # Remove entries that are too old
        cutoff_age = self.tt_age - 10
        keys_to_remove = []
        
        for key, entry in self.transposition_table.items():
            if entry.age < cutoff_age:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.transposition_table[key]
        
        logger.info(f"Cleaned {len(keys_to_remove)} old TT entries")
    
    def _is_time_up(self):
        """Check if time limit is exceeded."""
        if self.time_limit is None:
            return False
        return time.time() - self.start_time >= self.time_limit
    
    def _is_pv_node(self, alpha, beta):
        """Check if this is a PV (Principal Variation) node."""
        return beta - alpha > 1
    
    def _has_major_pieces(self, board, color):
        """Check if side has any major pieces (for null move pruning)."""
        for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            if len(board.pieces(piece_type, color)) > 0:
                return True
        return False
    
    def _alpha_beta_search(self, board, depth, use_advanced):
        """Standard alpha-beta search for non-expert modes."""
        best_score = float('-inf') if board.turn == chess.WHITE else float('inf')
        best_move = None
        alpha = float('-inf')
        beta = float('inf')
        
        moves = list(board.legal_moves)
        if use_advanced:
            moves = self._order_moves_advanced(board, moves, depth)
        else:
            moves = self._order_moves_simple(board, moves)
        
        for move in moves:
            board.push(move)
            
            if use_advanced:
                score = -self._negamax_advanced(board, depth-1, -beta, -alpha, -1 if board.turn else 1)
            else:
                score = self._minimax_simple(board, depth-1, alpha, beta, board.turn != chess.WHITE)
            
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
            
            if alpha >= beta:
                break
        
        elapsed = time.time() - self.start_time
        logger.info(f"Search completed: depth={depth}, nodes={self.nodes_evaluated}, time={elapsed:.2f}s")
        
        return best_move
    
    def _order_moves_simple(self, board, moves):
        """Simple move ordering for basic modes."""
        move_scores = []
        
        for move in moves:
            score = 0
            
            # Captures are good
            if board.is_capture(move):
                score += self._get_capture_score(board, move)
            
            # Promotions are good
            if move.promotion:
                score += self.piece_values.get(move.promotion, 0)
            
            # Checks are interesting
            board.push(move)
            if board.is_check():
                score += 50
            board.pop()
            
            move_scores.append((move, score))
        
        move_scores.sort(key=lambda x: x[1], reverse=True)
        return [move for move, _ in move_scores]
    
    def _minimax_simple(self, board, depth, alpha, beta, maximizing):
        """Simple minimax for basic difficulty levels."""
        self.nodes_evaluated += 1
        
        if depth == 0 or board.is_game_over():
            return self.get_centipawn_score(board)
        
        if maximizing:
            max_eval = float('-inf')
            for move in board.legal_moves:
                board.push(move)
                eval = self._minimax_simple(board, depth - 1, alpha, beta, False)
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
                eval = self._minimax_simple(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

def main():
    """Demo the enhanced chess evaluator."""
    board = chess.Board()
    
    # Make some moves to create an interesting position
    moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5"]
    for move_uci in moves:
        board.push_uci(move_uci)
    
    print("Current position:")
    print(board)
    
    # Create enhanced evaluator
    evaluator = ChessEvaluator()
    
    # Test different difficulty levels
    print("\n=== Testing Different Difficulty Levels ===")
    
    # Beginner level
    start_time = time.time()
    beginner_move = evaluator.get_best_move(board, depth=2, use_advanced=False)
    beginner_time = time.time() - start_time
    print(f"Beginner (depth 2): {beginner_move.uci()} ({board.san(beginner_move)}) - {beginner_time:.2f}s")
    
    # Intermediate level
    start_time = time.time()
    intermediate_move = evaluator.get_best_move(board, depth=4, use_advanced=True)
    intermediate_time = time.time() - start_time
    print(f"Intermediate (depth 4): {intermediate_move.uci()} ({board.san(intermediate_move)}) - {intermediate_time:.2f}s")
    
    # Expert level (the monster)
    start_time = time.time()
    expert_move = evaluator.get_best_move(board, depth=10, use_advanced=True, time_limit=30.0)
    expert_time = time.time() - start_time
    print(f"Expert (depth 10): {expert_move.uci()} ({board.san(expert_move)}) - {expert_time:.2f}s")
    
    # Show evaluation
    score = evaluator.get_centipawn_score(board)
    print(f"\nPosition evaluation: {score} centipawns")
    print(f"Game phase: {evaluator.game_phase:.2f} (0=endgame, 1=opening)")

if __name__ == "__main__":
    main()
