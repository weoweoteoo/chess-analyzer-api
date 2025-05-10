from flask import Flask, request, jsonify
from utils.analyzer import analyze_game

app = Flask(__name__)

@app.route("/api/analyze", methods=["GET"])
def analyze():
    pgn_path = "data/sample_game.pgn"  # You can change this to dynamic via request.args
    engine_path = "engine/stockfish"

    try:
        result = analyze_game(pgn_path, engine_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)