import os
import sys
import collections

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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

CORS(app, origins=[
    "http://localhost:5173",
    "https://chess-rating.onrender.com",
    "https://chess-rating.vercel.app",
    "https://chess-sh.netlify.app",
    "https://chess-analyzer-api-production.up.railway.app",
    "https://chess-ai-webs.onrender.com"
])

socketio = SocketIO(app, cors_allowed_origins=[
    "http://localhost:5173",
    "https://chess-rating.onrender.com",
    "https://chess-rating.vercel.app",
    "https://chess-sh.netlify.app",
    "https://chess-analyzer-api-production.up.railway.app",
    "https://chess-ai-webs.onrender.com"
])

try:
    model_path = os.path.join(os.path.dirname(__file__), "models", "traced_chess_model.pt")
    evaluator = ChessEvaluator(model_path)
    logger.info(f"Chess evaluator initialized successfully with model: {model_path}")
except Exception as e:
    logger.error(f"Error initializing chess evaluator: {e}")
    evaluator = None

@socketio.on('connect')
def handle_connect():
    logger.info("Client connected")

@socketio.on('disconnect')
def handle_disconnect():
    logger.info("Client disconnected")

@socketio.on('request_ai_move')
def handle_ai_move_request(data):
    logger.info(f"Received AI move request: {data}")
    
    if evaluator is None:
        socketio.emit('ai_move_response', {
            'success': False,
            'error': "Chess evaluator not available. Model could not be loaded.",
            'game_id': data.get('game_id', 'unknown') if isinstance(data, dict) else 'unknown'
        })
        return
    
    try:
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
        
        ai_move = get_move_by_difficulty(board, difficulty)
        
        if ai_move:
            ai_move_san = board.san(ai_move)
            board.push(ai_move)
            human_suggestion = get_human_suggestion(board, difficulty)
            
            response = {
                'success': True,
                'ai_move': {
                    'uci': ai_move.uci(),
                    'san': ai_move_san,
                    'evaluation': evaluator.get_centipawn_score(board)
                },
                'human_suggestion': None,
                'game_id': game_id
            }
            
            if human_suggestion:
                board_copy = board.copy()
                board_copy.push(human_suggestion)
                suggestion_evaluation = evaluator.get_centipawn_score(board_copy)
                
                response['human_suggestion'] = {
                    'uci': human_suggestion.uci(),
                    'san': board.san(human_suggestion),
                    'evaluation': suggestion_evaluation
                }
                logger.info(f"Human move suggestion: {human_suggestion.uci()} ({board.san(human_suggestion)})")
            
            logger.info(f"AI suggests move: {ai_move.uci()} ({ai_move_san})")
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
        logger.error(f"Error handling socket AI move request: {e}")
        socketio.emit('ai_move_response', {
            'success': False,
            'error': str(e),
            'game_id': data.get('game_id', 'unknown') if isinstance(data, dict) else 'unknown'
        })

def get_move_by_difficulty(board, difficulty):
    if difficulty == 'beginner':
        if random.random() < 0.3:
            legal_moves = list(board.legal_moves)
            if legal_moves:
                return random.choice(legal_moves)
        return evaluator.get_best_move(board, depth=1, use_advanced=False)
    
    elif difficulty == 'intermediate':
        return evaluator.get_best_move(board, depth=3, use_advanced=True)
    
    elif difficulty == 'expert':
        return evaluator.get_best_move(board, depth=5, use_advanced=True)
    
    else:
        logger.warning(f"Unknown difficulty '{difficulty}', defaulting to intermediate")
        return evaluator.get_best_move(board, depth=3, use_advanced=True)

def get_human_suggestion(board, difficulty):
    try:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        
        if difficulty == 'beginner':
            suggestion_depth = 2
        elif difficulty == 'intermediate':
            suggestion_depth = 3
        elif difficulty == 'expert':
            suggestion_depth = 4
        else:
            suggestion_depth = 3
        
        best_move = evaluator.get_best_move(board, depth=suggestion_depth, use_advanced=True)
        
        if best_move and best_move in legal_moves:
            return best_move
        else:
            return random.choice(legal_moves) if legal_moves else None
            
    except Exception as e:
        logger.error(f"Error getting human suggestion: {e}")
        legal_moves = list(board.legal_moves)
        return random.choice(legal_moves) if legal_moves else None

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'model_loaded': evaluator is not None
    })

if __name__ == '__main__':
    logger.info("Starting Chess AI API server")
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"Starting server on port {port}")
    socketio.run(app, host='0.0.0.0', port=port, debug=False)
