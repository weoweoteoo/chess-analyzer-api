from flask import Flask, request, jsonify
from flask_cors import CORS
from utils.analyzer import analyze_game
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Enable CORS for allowed origins
CORS(app, origins=[
    "http://localhost:5173",
    "https://chess-rating.onrender.com",
    "https://chess-rating.vercel.app",
    "https://chess-sh.netlify.app",
    "https://chess-analyzer-api-production.up.railway.app"
])

# Global variable to hold last result
last_analysis_result = {}

@app.route("/api/analyze", methods=["POST"])
def analyze():
    global last_analysis_result
    data = request.get_json()

    # Log the received JSON data
    logging.info("Received data from frontend: %s", data)

    moves = data.get("moves")
    player_color = data.get("playerColor")
    winner = data.get("winner")
    engine_path = "engine/stockfish"

    if not all([moves, player_color, winner]):
        return jsonify({"error": "Missing required fields: moves, playerColor, or winner"}), 400

    try:
        result = analyze_game(moves, player_color, winner, engine_path)
        last_analysis_result = result
        return jsonify(result)
    except Exception as e:
        logging.error("Error during analysis: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route("/api/result", methods=["GET"])
def get_result():
    if not last_analysis_result:
        return jsonify({"message": "No analysis available yet"}), 404
    return jsonify(last_analysis_result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
