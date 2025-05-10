from flask import Flask, request, jsonify
from flask_cors import CORS
from utils.analyzer import analyze_game

app = Flask(__name__)


CORS(app, origins=[
    "http://localhost:5173",
    "https://chess-rating.onrender.com",
    "https://chess-rating.vercel.app",
    "https://chess-sh.netlify.app",
    "https://chess-analyzer-api-production.up.railway.app"
])

@app.route("/api/analyze", methods=["GET"])
def analyze():
    pgn_path = "data/sample_game.pgn"
    engine_path = "engine/stockfish"

    try:
        result = analyze_game(pgn_path, engine_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
