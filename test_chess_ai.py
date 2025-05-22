import socketio
import time
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create a Socket.IO client
sio = socketio.Client()

@sio.event
def connect():
    logger.info("Connected to server")

@sio.event
def disconnect():
    logger.info("Disconnected from server")

@sio.event
def ai_move_response(data):
    logger.info(f"Received AI move response: {data}")
    if data['success']:
        logger.info(f"AI played: {data['move']} ({data['san']})")
        logger.info(f"Evaluation: {data['evaluation']} centipawns")
    else:
        logger.error(f"Error: {data['error']}")

def test_ai():
    try:
        # Connect to the server
        sio.connect('http://localhost:5001')
        
        # Test with a simple opening position
        logger.info("Testing with initial position + e4")
        sio.emit('request_ai_move', {
            'moves': ['e4'],
            'difficulty': 'intermediate',
            'game_id': 'test-game-1'
        })
        
        # Wait for response
        time.sleep(3)
        
        # Test with a more complex position
        logger.info("Testing with a more complex position")
        sio.emit('request_ai_move', {
            'moves': ['e4', 'e5', 'Nf3', 'Nc6', 'Bb5'],
            'difficulty': 'expert',
            'game_id': 'test-game-2'
        })
        
        # Wait for response
        time.sleep(3)
        
        # Test with beginner difficulty
        logger.info("Testing with beginner difficulty")
        sio.emit('request_ai_move', {
            'moves': ['d4', 'd5', 'c4'],
            'difficulty': 'beginner',
            'game_id': 'test-game-3'
        })
        
        # Wait for response
        time.sleep(3)
        
    except Exception as e:
        logger.error(f"Error during test: {e}")
    finally:
        # Disconnect from the server
        sio.disconnect()

if __name__ == "__main__":
    logger.info("Starting Chess AI API test")
    test_ai()