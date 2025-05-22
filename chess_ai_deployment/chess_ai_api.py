import chess
import logging
import random
import os
from flask import Flask, request, jsonify
from flask_socketio import SocketIO
from implementation.evaluator import ChessEvaluator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Create a global evaluator instance
evaluator = ChessEvaluator()

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    logger.info("Client connected")

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    logger.info("Client disconnected")

@socketio.on('request_ai_move')
def handle_ai_move_request(data):
    """Socket event handler for AI move requests.
    
    Expected data format:
    {
        'moves': ['e4', 'e5', ...],  # List of moves in algebraic notation
        'difficulty': 'beginner|intermediate|expert',
        'game_id': 'some-unique-id'
    }
    """
    logger.info(f"Received AI move request: {data}")
    
    try:
        # Extract data from the request
        moves = data.get('moves', [])
        difficulty = data.get('difficulty', 'intermediate')
        game_id = data.get('game_id', 'unknown')
        
        # Create a board and apply all moves
        board = chess.Board()
        for move_str in moves:
            try:
                # Try as UCI (e2e4)
                move = chess.Move.from_uci(move_str)
                if move in board.legal_moves:
                    board.push(move)
                    continue
            except ValueError:
                pass
            
            try:
                # Try as SAN (e4)
                move = board.parse_san(move_str)
                board.push(move)
            except ValueError:
                logger.error(f"Invalid move: {move_str}")
                socketio.emit('ai_move_response', {
                    'success': False,
                    'error': f"Invalid move: {move_str}",
                    'game_id': game_id
                })
                return
        
        # Get AI move based on difficulty
        move = get_move_by_difficulty(board, difficulty)
        
        if move:
            # Prepare response with the AI move
            response = {
                'success': True,
                'move': move.uci(),  # UCI format (e.g., "e2e4")
                'san': board.san(move),  # SAN format (e.g., "e4")
                'evaluation': evaluator.get_centipawn_score(board),
                'game_id': game_id
            }
            logger.info(f"AI suggests move: {move.uci()} ({board.san(move)})")
        else:
            response = {
                'success': False,
                'error': 'No legal moves available',
                'game_id': game_id
            }
            logger.warning("No legal moves available")
        
        # Emit the response back to the client
        socketio.emit('ai_move_response', response)
        logger.info(f"Sent AI move response for game {game_id}")
    
    except Exception as e:
        logger.error(f"Error handling socket AI move request: {e}")
        socketio.emit('ai_move_response', {
            'success': False,
            'error': str(e),
            'game_id': data.get('game_id', 'unknown')
        })

def get_move_by_difficulty(board, difficulty):
    """Get an AI move based on the difficulty level.
    
    Args:
        board: A chess.Board object
        difficulty: 'beginner', 'intermediate', or 'expert'
        
    Returns:
        chess.Move: The selected move
    """
    if difficulty == 'beginner':
        # Beginner: Shallow search with occasional random moves
        if random.random() < 0.3:  # 30% chance of random move
            legal_moves = list(board.legal_moves)
            if legal_moves:
                return random.choice(legal_moves)
        
        # Otherwise use a shallow search
        return evaluator.get_best_move(board, depth=1)
    
    elif difficulty == 'intermediate':
        # Intermediate: Medium depth search
        return evaluator.get_best_move(board, depth=2)
    
    elif difficulty == 'expert':
        # Expert: Deep search
        return evaluator.get_best_move(board, depth=3)
    
    else:
        # Default to intermediate
        logger.warning(f"Unknown difficulty '{difficulty}', defaulting to intermediate")
        return evaluator.get_best_move(board, depth=2)

if __name__ == '__main__':
    logger.info("Starting Chess AI API server")
    # Get port from environment variable (Railway sets this automatically)
    port = int(os.environ.get("PORT", 5001))
    socketio.run(app, host='0.0.0.0', port=port, debug=False)