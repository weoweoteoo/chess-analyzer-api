from flask import Flask, request, jsonify
import logging
from chess_analysis import analyze_game

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

ENGINE_PATH = "/usr/games/stockfish"  # Change this if your Stockfish path differs


def remove_consecutive_duplicates(moves):
    """
    Remove consecutive duplicate moves.
    """
    deduped = []
    for move in moves:
        if not deduped or deduped[-1] != move:
            deduped.append(move)
    return deduped


@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    logging.info(f"Received data from frontend: {data}")

    try:
        raw_moves = data['moves']
        player_color = data['playerColor']
        winner = data['winner']

        # Clean up repeated moves
        cleaned_moves = remove_consecutive_duplicates(raw_moves)

        result = analyze_game(cleaned_moves, player_color, winner, ENGINE_PATH)
        return jsonify(result)

    except Exception as e:
        logging.exception("Error analyzing game:")
        return jsonify({"error": str(e)}), 500


@app.route('/api/result', methods=['GET'])
def result():
    return jsonify({"message": "Endpoint working"})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
