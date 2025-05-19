from flask import Flask, request, jsonify
from flask_cors import CORS
from utils.analyzer import analyze_game
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Enable CORS for frontend URLs
CORS(app, origins=[
    "http://localhost:5173",
    "https://chess-rating.onrender.com",
    "https://chess-rating.vercel.app",
    "https://chess-sh.netlify.app",
    "https://chess-analyzer-api-production.up.railway.app"
])

# Store analysis results using (playerId, matchId) as keys
analysis_cache = {}

def remove_consecutive_duplicates(moves):
    """
    Removes consecutive duplicate moves.
    """
    cleaned = []
    for move in moves:
        if not cleaned or cleaned[-1] != move:
            cleaned.append(move)
    return cleaned

@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    logging.info("Received data from frontend: %s", data)

    moves = data.get("moves")
    player_color = data.get("playerColor")
    winner = data.get("winner")
    player_id = data.get("playerId")
    match_id = data.get("matchId")
    engine_path = "engine/stockfish"

    # Validate required fields
    if not all([moves, player_color, winner, player_id, match_id]):
        return jsonify({"error": "Missing required fields: moves, playerColor, winner, playerId, or matchId"}), 400

    # Clean up moves
    cleaned_moves = remove_consecutive_duplicates(moves)
    logging.info("Cleaned moves for analysis: %s", cleaned_moves)

    try:
        result = analyze_game(cleaned_moves, player_color, winner, engine_path)

        # Cache result using a tuple key (playerId, matchId)
        analysis_cache[(player_id, match_id)] = result

        return jsonify(result)
    except Exception as e:
        logging.exception("Error during analysis")
        return jsonify({"error": str(e)}), 500

@app.route("/api/result", methods=["GET"])
def get_result():
    player_id = request.args.get("playerId")
    match_id = request.args.get("matchId")  # Optional

    if not player_id:
        return jsonify({"error": "Missing playerId"}), 400

    # Try to fetch exact match if matchId is given
    if match_id:
        result = analysis_cache.get((player_id, match_id))
        if not result:
            return jsonify({"message": "No analysis available for this playerId and matchId"}), 404
        return jsonify(result)

    # If matchId not provided, return the latest match for this player
    matches = [(pid, mid) for (pid, mid) in analysis_cache if pid == player_id]
    if not matches:
        return jsonify({"message": "No analysis available for this playerId"}), 404

    # Return the last match added (most recent by insertion order)
    latest_key = matches[-1]
    return jsonify(analysis_cache[latest_key])

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
