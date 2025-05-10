import os
from utils.analyzer import analyze_game

if __name__ == "__main__":
    # Set up paths
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Absolute path to PGN file and Stockfish engine
    pgn_path = os.path.join(base_dir, "data", "sample_game.pgn")
    engine_path = os.path.join(base_dir, "engine", "stockfish")

    # Analyze the game and print the results
    analysis_results = analyze_game(pgn_path, engine_path)
    print(analysis_results)