from flask import Flask, request, jsonify
from flask_cors import CORS
from utils.analyzer import analyze_game_from_moves  # You need to implement this function

app = Flask(__name__)

CORS(app, origins=[
    "http://localhost:5173",
    "https://chess-rating.onrender.com",
    "https://chess-rating.vercel.app",
    "https://chess-sh.netlify.app",
    "https://chess-analyzer-api-production.up.railway.app"
])

@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()
        moves = data.get("moves")
        player_color = data.get("playerColor")
        winner = data.get("winner")

        if not moves or not player_color or not winner:
            return jsonify({"error": "Missing required fields: moves, playerColor, or winner."}), 400

        result = analyze_game_from_moves(moves, player_color, winner)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
