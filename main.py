import os
from utils.analyzer import analyze_game

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))

    pgn_path = os.path.join(base_dir, "data", "sample_game.pgn")
    engine_path = os.path.join(base_dir, "engine", "stockfish")

    analysis_results = analyze_game(pgn_path, engine_path)
    print(analysis_results)
