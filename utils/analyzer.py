import chess.pgn
import chess.engine
import os
import json
import matplotlib.pyplot as plt


def classify_move(cp_loss):
    """
    Classify the move based on the CP loss.
    """
    if cp_loss < 50:
        return "Best"
    elif cp_loss < 100:
        return "Inaccuracy"
    elif cp_loss < 200:
        return "Mistake"
    else:
        return "Blunder"


def suggest_from_counts(blunders, mistakes, inaccuracies):
    """
    Generate improvement suggestions based on the number of blunders, mistakes, and inaccuracies.
    """
    suggestions = []
    if blunders > 0:
        suggestions.append("Avoid blunders by thinking deeper before major captures or sacrifices.")
    if mistakes > 0:
        suggestions.append("Focus on tactical awareness to reduce mistakes.")
    if inaccuracies > 0:
        suggestions.append("Improve your positional understanding and mid-game planning.")
    if blunders + mistakes + inaccuracies == 0:
        suggestions.append("Great game! Keep practicing to maintain accuracy.")
    return suggestions


def analyze_game(pgn_path, engine_path):
    """
    Analyze a chess game from a PGN file using Stockfish engine, and return a dictionary with the analysis results.
    
    :param pgn_path: Path to the PGN file containing the game.
    :param engine_path: Path to the Stockfish engine executable.
    :return: A dictionary containing the game analysis.
    """
    # Open the PGN file
    with open(pgn_path) as pgn:
        game = chess.pgn.read_game(pgn)

    # Get player name (White or Black)
    player_name = game.headers.get("White", "Player")
    board = game.board()
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)

    move_data = []
    total_cp_loss = 0
    blunders = 0
    mistakes = 0
    inaccuracies = 0
    move_number = 1

    node = game
    while node.variations:
        next_node = node.variation(0)
        move = next_node.move

        # Get the move in SAN (Standard Algebraic Notation)
        san_move = board.san(move)

        # Evaluate before the move
        info_before = engine.analyse(board, chess.engine.Limit(depth=15))
        score_before = info_before["score"].relative.score(mate_score=10000)

        board.push(move)

        # Evaluate after the move
        info_after = engine.analyse(board, chess.engine.Limit(depth=15))
        score_after = info_after["score"].relative.score(mate_score=10000)

        # Calculate CP loss (change in evaluation)
        cp_loss = (score_before - score_after) if score_before is not None and score_after is not None else 0
        classification = classify_move(abs(cp_loss))

        if classification == "Blunder":
            blunders += 1
        elif classification == "Mistake":
            mistakes += 1
        elif classification == "Inaccuracy":
            inaccuracies += 1

        move_data.append({
            "Move Number": move_number,
            "Move": san_move,
            "CP Loss": cp_loss,
            "Classification": classification
        })

        total_cp_loss += abs(cp_loss)
        node = next_node
        move_number += 1

    engine.quit()

    total_moves = len(move_data)
    good_moves = total_moves - blunders - mistakes - inaccuracies
    accuracy = round((good_moves / total_moves) * 100, 2) if total_moves > 0 else 0

    # Create the game analysis summary
    summary = {
        "player": player_name,
        "total_moves": total_moves,
        "accuracy": accuracy,
        "blunders": blunders,
        "mistakes": mistakes,
        "inaccuracies": inaccuracies,
        "suggestions": suggest_from_counts(blunders, mistakes, inaccuracies),
        "worst_moves": sorted(move_data, key=lambda x: abs(x["CP Loss"]), reverse=True)[:5]  # Top 5 worst moves
    }

    # Return the analysis summary dictionary
    return summary
