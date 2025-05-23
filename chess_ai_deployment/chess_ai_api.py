import os
import sys
import collections

# Compatibility fix for MutableMapping for Python 3.10+
try:
    from collections.abc import MutableMapping
except ImportError:
    from collections import MutableMapping

if not hasattr(collections, 'MutableMapping'):
    collections.MutableMapping = MutableMapping

import chess
import logging
import random
import json
from flask import Flask, request, jsonify
from flask_socketio import SocketIO
from flask_cors import CORS
from implementation.evaluator import ChessEvaluator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Setup CORS for specific frontend origins
CORS(app, origins=[
    "http://localhost:5173",
    "https://chess-rating.onrender.com",
    "https://chess-rating.vercel.app",
    "https://chess-sh.netlify.app",
    "https://chess-analyzer-api-production.up.railway.app",
    "https://chess-ai-webs.onrender.com"
])

# Initialize SocketIO with matching CORS policy
socketio = SocketIO(app, cors_allowed_origins=[
    "http://localhost:5173",
    "https://chess-rating.onrender.com",
    "https://chess-rating.vercel.app",
    "https://chess-sh.netlify.app",
    "https://chess-analyzer-api-production.up.railway.app",
    "https://chess-ai-webs.onrender.com"
])

# Create a global evaluator instance
try:
    # Specify the path to your traced model
    model_path = os.path.join(os.path.dirname(__file__), "models", "traced_chess_model.pt")
    evaluator = ChessEvaluator(model_path)
    logger.info(f"Chess evaluator initialized successfully with model: {model_path}")
except Exception as e:
    logger.error(f"Error initializing chess evaluator: {e}")
    evaluator = None  # Set to None so we can check if it's available later
    # Let the app start, but AI moves will fail if evaluator is broken

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
    
    # Check if evaluator is available
    if evaluator is None:
        socketio.emit('ai_move_response', {
            'success': False,
            'error': "Chess evaluator not available. Model could not be loaded.",
            'game_id': data.get('game_id', 'unknown') if isinstance(data, dict) else 'unknown'
        })
        return
    
    try:
        # Parse stringified JSON data if needed
        if isinstance(data, str):
            try:
                data = json.loads(data)
                logger.info(f"Parsed string data into JSON: {data}")
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON data: {e}")
                socketio.emit('ai_move_response', {
                    'success': False,
                    'error': "Invalid data format. Expected JSON object.",
                    'game_id': 'unknown'
                })
                return
        
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
            'game_id': data.get('game_id', 'unknown') if isinstance(data, dict) else 'unknown'
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
        return evaluator.get_best_move(board, depth=1, use_advanced=False)
    
    elif difficulty == 'intermediate':
        # Intermediate: Medium depth search with some advanced techniques
        return evaluator.get_best_move(board, depth=3, use_advanced=True)
    
    elif difficulty == 'expert':
        # Expert: Deep search with all advanced techniques
        return evaluator.get_best_move(board, depth=5, use_advanced=True)
    
    else:
        # Default to intermediate
        logger.warning(f"Unknown difficulty '{difficulty}', defaulting to intermediate")
        return evaluator.get_best_move(board, depth=3, use_advanced=True)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring."""
    return jsonify({
        'status': 'ok',
        'model_loaded': evaluator is not None
    })

if __name__ == '__main__':
    logger.info("Starting Chess AI API server")
    # Get port from environment variable (Railway sets this automatically)
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"Starting server on port {port}")
    socketio.run(app, host='0.0.0.0', port=port, debug=False)
