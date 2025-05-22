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
    evaluator = ChessEvaluator()
    logger.info("Chess evaluator initialized successfully")
except Exception as e:
    logger.error(f"Error initializing chess evaluator: {e}")
    # Let the app start, but AI moves will fail if evaluator is broken

@socketio.on('connect')
def handle_connect():
    logger.info("Client connected")

@socketio.on('disconnect')
def handle_disconnect():
    logger.info("Client disconnected")

@socketio.on('request_ai_move')
def handle_ai_move_request(data):
    logger.info(f"Received AI move request: {data}")
    
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
        
        moves = data.get('moves', [])
        difficulty = data.get('difficulty', 'intermediate')
        game_id = data.get('game_id', 'unknown')

        board = chess.Board()
        for move_str in moves:
            try:
                move = chess.Move.from_uci(move_str)
                if move in board.legal_moves:
                    board.push(move)
                    continue
            except ValueError:
                pass
            
            try:
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
        
        move = get_move_by_difficulty(board, difficulty)

        if move:
            response = {
                'success': True,
                'move': move.uci(),
                'san': board.san(move),
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
        
        socketio.emit('ai_move_response', response)
        logger.info(f"Sent AI move response for game {game_id}")
    
    except Exception as e:
        logger.error(f"Error handling AI move: {e}")
        socketio.emit('ai_move_response', {
            'success': False,
            'error': str(e),
            'game_id': data.get('game_id', 'unknown') if isinstance(data, dict) else 'unknown'
        })

def get_move_by_difficulty(board, difficulty):
    """Select a move based on difficulty."""
    if difficulty == 'beginner':
        if random.random() < 0.3:
            legal_moves = list(board.legal_moves)
            if legal_moves:
                return random.choice(legal_moves)
        return evaluator.get_best_move(board, depth=1)
    elif difficulty == 'intermediate':
        return evaluator.get_best_move(board, depth=2)
    elif difficulty == 'expert':
        return evaluator.get_best_move(board, depth=3)
    else:
        logger.warning(f"Unknown difficulty '{difficulty}', defaulting to intermediate")
        return evaluator.get_best_move(board, depth=2)

if __name__ == '__main__':
    logger.info("Starting Chess AI API server")
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"Starting server on port {port}")
    socketio.run(app, host='0.0.0.0', port=port, debug=False)
