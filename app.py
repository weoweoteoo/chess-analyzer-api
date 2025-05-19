from flask import Flask, request, jsonify
from flask_cors import CORS
from utils.analyzer import analyze_game
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Enable CORS
CORS(app, origins=[
    "http://localhost:5173",
    "https://chess-rating.onrender.com",
    "https://chess-rating.vercel.app",
    "https://chess-sh.netlify.app",
    "https://chess-analyzer-api-production.up.railway.app"
])

# In-memory storage for all results: { playerId: { matchId: result } }
last_analysis_result = {}

def remove_consecutive_duplicates(moves):
    """
    Removes consecutive duplicate moves (often sent by frontend).
    """
    cleaned = []
    for move in moves:
        if not cleaned or cleaned[-1] != move:
            cleaned.append(move)
    return cleaned

@app.route("/api/analyze", methods=["POST"])
def analyze():
    global last_analysis_result
    data = request.get_json()

    logging.info("Received data from frontend: %s", data)

    moves = data.get("moves")
    player_color = data.get("playerColor")
    winner = data.get("winner")
    player_id = data.get("playerId")
    match_id = data.get("matchId")
    engine_path = "engine/stockfish"

    if not all([moves, player_color, winner, player_id, match_id]):
        return jsonify({"error": "Missing one or more required fields: moves, playerColor, winner, playerId, matchId"}), 400

    # Clean up moves
    cleaned_moves = remove_consecutive_duplicates(moves)
    logging.info("Cleaned moves for analysis: %s", cleaned_moves)

    try:
        result = analyze_game(cleaned_moves, player_color, winner, engine_path)

        # Store the result per player and match
        if player_id not in last_analysis_result:
            last_analysis_result[player_id] = {}
        last_analysis_result[player_id][match_id] = result

        return jsonify(result)
    except Exception as e:
        logging.exception("Error during analysis")
        return jsonify({"error": str(e)}), 500

@app.route("/api/result", methods=["GET"])
def get_result():
    player_id = request.args.get("playerId")
    match_id = request.args.get("matchId")

    if not player_id or not match_id:
        return jsonify({"error": "Missing playerId or matchId parameter"}), 400

    result = last_analysis_result.get(player_id, {}).get(match_id)

    if not result:
        return jsonify({"message": "No analysis available for this match"}), 404

    return jsonify(result)

if _name_ == "_main_":
    app.run(debug=True, host="0.0.0.0", port=5000)
