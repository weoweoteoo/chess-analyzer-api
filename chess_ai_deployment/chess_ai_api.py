import os
import sys

# Compatibility fix for Python 3.10+ (collections.MutableMapping deprecated)
import collections
if not hasattr(collections, 'MutableMapping'):
    collections.MutableMapping = collections.abc.MutableMapping

import chess
import logging
import random
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
try:
    evaluator = ChessEvaluator()
    logger.info("Chess evaluator initialized successfully")
except Exception as e:
    logger.error(f"Error initializing chess evaluator: {e}")
    # Continue startup; AI will fail gracefully

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

    except Exception as e:
        logger.error(f"Error handling AI move: {e}")
        socketio.emit('ai_move_response', {
            'success': False,
            'error': str(e),
            'game_id': data.get('game_id', 'unknown')
        })

def get_move_by_difficulty(board, difficulty):
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

    logger.warning(f"Unknown difficulty '{difficulty}', defaulting to intermediate")
    return evaluator.get_best_move(board, depth=2)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Render sets this to 10000
    logger.info(f"Starting Chess AI API on port {port}")
    socketio.run(app, host='0.0.0.0', port=port, debug=False)
