import os
import sys
import collections
import time
import threading

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
from functools import wraps

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Setup CORS for specific frontend origins
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "https://chess-rating.onrender.com",
    "https://chess-rating.vercel.app",
    "https://chess-sh.netlify.app",
    "https://chess-analyzer-api-production.up.railway.app",
    "https://chess-ai-webs.onrender.com"
]

CORS(app, origins=ALLOWED_ORIGINS)

# Initialize SocketIO with matching CORS policy
socketio = SocketIO(app, cors_allowed_origins=ALLOWED_ORIGINS)

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

# Rate limiting implementation
class RateLimiter:
    def __init__(self, max_calls, time_frame):
        self.max_calls = max_calls
        self.time_frame = time_frame  # in seconds
        self.calls = {}
        self.lock = threading.Lock()
    
    def is_allowed(self, client_id):
        current_time = time.time()
        with self.lock:
            # Clean up old entries
            self.calls = {k: v for k, v in self.calls.items() 
                         if current_time - v[-1] < self.time_frame}
            
            # Check if client exists and has calls
            if client_id not in self.calls:
                self.calls[client_id] = [current_time]
                return True
            
            # Check if client has exceeded rate limit
            client_calls = self.calls[client_id]
            if len(client_calls) < self.max_calls:
                client_calls.append(current_time)
                return True
            
            # Check if oldest call is outside time frame
            if current_time - client_calls[0] > self.time_frame:
                client_calls.pop(0)
                client_calls.append(current_time)
                return True
            
            return False

# Create rate limiters
move_limiter = RateLimiter(max_calls=10, time_frame=60)  # 10 moves per minute
eval_limiter = RateLimiter(max_calls=20, time_frame=60)  # 20 evaluations per minute

# Difficulty settings with corresponding search parameters
DIFFICULTY_SETTINGS = {
    'beginner': {
        'depth': 2,
        'use_advanced': False,
        'time_limit': 1.0,
        'random_chance': 0.2
    },
    'intermediate': {
        'depth': 4,
        'use_advanced': True,
        'time_limit': 3.0,
        'random_chance': 0.0
    },
    'expert': {
        'depth': 10,
        'use_advanced': True,
        'time_limit': 30.0,
        'random_chance': 0.0
    }
}

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    logger.info(f"Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    logger.info(f"Client disconnected: {request.sid}")

def rate_limit_socket(limiter):
    """Decorator for rate limiting socket events."""
    def decorator(f):
        @wraps(f)
        def wrapped(data):
            client_id = request.sid
            if not limiter.is_allowed(client_id):
                socketio.emit('error', {
                    'message': 'Rate limit exceeded. Please wait before making more requests.',
                    'game_id': data.get('game_id', 'unknown') if isinstance(data, dict) else 'unknown'
                }, room=client_id)
                return
            return f(data)
        return wrapped
    return decorator

@socketio.on('request_ai_move')
@rate_limit_socket(move_limiter)
def handle_ai_move_request(data):
    """Socket event handler for AI move requests.
    
    Expected data format:
    {
        'moves': ['e4', 'e5', ...],  # List of moves in algebraic notation
        'fen': 'optional FEN string',  # Alternative to moves list
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
        fen = data.get('fen', None)
        difficulty = data.get('difficulty', 'intermediate')
        game_id = data.get('game_id', 'unknown')
        
        # Create a board from FEN or apply moves
        board = create_board_from_request(moves, fen)
        if board is None:
            socketio.emit('ai_move_response', {
                'success': False,
                'error': "Could not create valid board position from input",
                'game_id': game_id
            })
            return
        
        # Check if game is already over
        if board.is_game_over():
            result = get_game_result(board)
            socketio.emit('ai_move_response', {
                'success': False,
                'error': f"Game is already over: {result}",
                'game_id': game_id,
                'result': result
            })
            return
        
        # Get AI move based on difficulty
        move = get_move_by_difficulty(board, difficulty)
        
        if move:
            # Apply the move to get post-move evaluation
            board.push(move)
            post_move_eval = evaluator.get_centipawn_score(board)
            
            # Check if the move leads to game over
            game_over = board.is_game_over()
            result = get_game_result(board) if game_over else None
            
            # Prepare response with the AI move
            response = {
                'success': True,
                'move': move.uci(),  # UCI format (e.g., "e2e4")
                'san': board.san(move),  # SAN format (e.g., "e4")
                'evaluation': post_move_eval,
                'game_id': game_id,
                'game_over': game_over,
                'result': result,
                'fen': board.fen()
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

@socketio.on('request_position_evaluation')
@rate_limit_socket(eval_limiter)
def handle_evaluation_request(data):
    """Socket event handler for position evaluation requests.
    
    Expected data format:
    {
        'moves': ['e4', 'e5', ...],  # List of moves in algebraic notation
        'fen': 'optional FEN string',  # Alternative to moves list
        'depth': 3,  # Optional evaluation depth
        'game_id': 'some-unique-id'
    }
    """
    logger.info(f"Received evaluation request: {data}")
    
    # Check if evaluator is available
    if evaluator is None:
        socketio.emit('evaluation_response', {
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
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON data: {e}")
                socketio.emit('evaluation_response', {
                    'success': False,
                    'error': "Invalid data format. Expected JSON object.",
                    'game_id': 'unknown'
                })
                return
        
        # Extract data from the request
        moves = data.get('moves', [])
        fen = data.get('fen', None)
        depth = data.get('depth', 3)
        game_id = data.get('game_id', 'unknown')
        
        # Create a board from FEN or apply moves
        board = create_board_from_request(moves, fen)
        if board is None:
            socketio.emit('evaluation_response', {
                'success': False,
                'error': "Could not create valid board position from input",
                'game_id': game_id
            })
            return
        
        # Get evaluation
        evaluation = evaluator.get_centipawn_score(board)
        
        # Get best move at requested depth
        best_move = evaluator.get_best_move(board, depth=depth, use_advanced=True)
        
        # Check if game is over
        game_over = board.is_game_over()
        result = get_game_result(board) if game_over else None
        
        # Prepare response
        response = {
            'success': True,
            'evaluation': evaluation,
            'best_move': best_move.uci() if best_move else None,
            'best_move_san': board.san(best_move) if best_move else None,
            'game_id': game_id,
            'fen': board.fen(),
            'game_over': game_over,
            'result': result
        }
        
        # Emit the response back to the client
        socketio.emit('evaluation_response', response)
        logger.info(f"Sent evaluation response for game {game_id}")
    
    except Exception as e:
        logger.error(f"Error handling evaluation request: {e}")
        socketio.emit('evaluation_response', {
            'success': False,
            'error': str(e),
            'game_id': data.get('game_id', 'unknown') if isinstance(data, dict) else 'unknown'
        })

def create_board_from_request(moves=None, fen=None):
    """Create a chess board from moves list or FEN string.
    
    Args:
        moves: List of moves in algebraic notation
        fen: FEN string representing a position
        
    Returns:
        chess.Board or None if invalid input
    """
    try:
        if fen:
            # Create board from FEN
            return chess.Board(fen)
        elif moves:
            # Create board and apply moves
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
                    return None
            return board
        else:
            # No input provided, return starting position
            return chess.Board()
    except Exception as e:
        logger.error(f"Error creating board: {e}")
        return None

def get_game_result(board):
    """Get the result of a game that's over.
    
    Args:
        board: A chess.Board object that's in a game over state
        
    Returns:
        str: Description of the result
    """
    if board.is_checkmate():
        winner = "Black" if board.turn == chess.WHITE else "White"
        return f"Checkmate - {winner} wins"
    elif board.is_stalemate():
        return "Draw by stalemate"
    elif board.is_insufficient_material():
        return "Draw by insufficient material"
    elif board.is_fifty_moves():
        return "Draw by fifty-move rule"
    elif board.is_repetition():
        return "Draw by threefold repetition"
    else:
        return "Game over"

def get_move_by_difficulty(board, difficulty):
    """Get an AI move based on the difficulty level.
    
    Args:
        board: A chess.Board object
        difficulty: 'beginner', 'intermediate', or 'expert'
        
    Returns:
        chess.Move: The selected move
    """
    # Get difficulty settings or default to intermediate
    settings = DIFFICULTY_SETTINGS.get(difficulty, DIFFICULTY_SETTINGS['intermediate'])
    
    # Check for random move chance
    if random.random() < settings['random_chance']:
        legal_moves = list(board.legal_moves)
        if legal_moves:
            return random.choice(legal_moves)
    
    # Use evaluator with appropriate settings
    return evaluator.get_best_move(
        board, 
        depth=settings['depth'], 
        use_advanced=settings['use_advanced'],
        time_limit=settings['time_limit']
    )

@app.route('/api/move', methods=['POST'])
def api_move():
    """REST API endpoint for getting an AI move."""
    # Check if evaluator is available
    if evaluator is None:
        return jsonify({
            'success': False,
            'error': "Chess evaluator not available. Model could not be loaded."
        }), 503
    
    # Check rate limit
    client_id = request.remote_addr
    if not move_limiter.is_allowed(client_id):
        return jsonify({
            'success': False,
            'error': "Rate limit exceeded. Please wait before making more requests."
        }), 429
    
    try:
        # Get JSON data
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': "No data provided"
            }), 400
        
        # Extract data
        moves = data.get('moves', [])
        fen = data.get('fen', None)
        difficulty = data.get('difficulty', 'intermediate')
        
        # Create board
        board = create_board_from_request(moves, fen)
        if board is None:
            return jsonify({
                'success': False,
                'error': "Could not create valid board position from input"
            }), 400
        
        # Check if game is already over
        if board.is_game_over():
            result = get_game_result(board)
            return jsonify({
                'success': False,
                'error': f"Game is already over: {result}",
                'result': result
            }), 400
        
        # Get AI move
        move = get_move_by_difficulty(board, difficulty)
        
        if move:
            # Apply the move to get post-move evaluation
            board.push(move)
            post_move_eval = evaluator.get_centipawn_score(board)
            
            # Check if the move leads to game over
            game_over = board.is_game_over()
            result = get_game_result(board) if game_over else None
            
            return jsonify({
                'success': True,
                'move': move.uci(),
                'san': board.san(move),
                'evaluation': post_move_eval,
                'game_over': game_over,
                'result': result,
                'fen': board.fen()
            })
        else:
            return jsonify({
                'success': False,
                'error': 'No legal moves available'
            }), 400
    
    except Exception as e:
        logger.error(f"Error handling API move request: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/evaluate', methods=['POST'])
def api_evaluate():
    """REST API endpoint for evaluating a position."""
    # Check if evaluator is available
    if evaluator is None:
        return jsonify({
            'success': False,
            'error': "Chess evaluator not available. Model could not be loaded."
        }), 503
    
    # Check rate limit
    client_id = request.remote_addr
    if not eval_limiter.is_allowed(client_id):
        return jsonify({
            'success': False,
            'error': "Rate limit exceeded. Please wait before making more requests."
        }), 429
    
    try:
        # Get JSON data
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': "No data provided"
            }), 400
        
        # Extract data
        moves = data.get('moves', [])
        fen = data.get('fen', None)
        depth = data.get('depth', 3)
        
        # Create board
        board = create_board_from_request(moves, fen)
        if board is None:
            return jsonify({
                'success': False,
                'error': "Could not create valid board position from input"
            }), 400
        
        # Get evaluation
        evaluation = evaluator.get_centipawn_score(board)
        
        # Get best move at requested depth
        best_move = evaluator.get_best_move(board, depth=depth, use_advanced=True)
        
        # Check if game is over
        game_over = board.is_game_over()
        result = get_game_result(board) if game_over else None
        
        return jsonify({
            'success': True,
            'evaluation': evaluation,
            'best_move': best_move.uci() if best_move else None,
            'best_move_san': board.san(best_move) if best_move else None,
            'fen': board.fen(),
            'game_over': game_over,
            'result': result
        })
    
    except Exception as e:
        logger.error(f"Error handling API evaluation request: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring."""
    return jsonify({
        'status': 'ok',
        'model_loaded': evaluator is not None,
        'version': '1.0.0'
    })

if __name__ == '__main__':
    logger.info("Starting Chess AI API server")
    # Get port from environment variable (Railway sets this automatically)
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"Starting server on port {port}")
    socketio.run(app, host='0.0.0.0', port=port, debug=False)
