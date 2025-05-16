import chess
import chess.engine

def classify_move(cp_loss):
    if cp_loss < 50:
        return "Best"
    elif cp_loss < 100:
        return "Inaccuracy"
    elif cp_loss < 200:
        return "Mistake"
    else:
        return "Blunder"

def suggest_from_counts(blunders, mistakes, inaccuracies):
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

def analyze_game(moves, player_color, winner, engine_path):
    board = chess.Board()
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)

    move_data = []
    blunders = 0
    mistakes = 0
    inaccuracies = 0
    total_cp_loss = 0
    move_number = 1

    for i, move in enumerate(moves):
        try:
            chess_move = board.parse_san(move)
        except ValueError:
            try:
                chess_move = board.parse_uci(move)
            except ValueError:
                print(f"⚠ Invalid move at index {i}: {move} — skipping rest of game.")
                break

        info_before = engine.analyse(board, chess.engine.Limit(depth=15))
        score_before = info_before["score"].relative.score(mate_score=10000)

        board.push(chess_move)

        info_after = engine.analyse(board, chess.engine.Limit(depth=15))
        score_after = info_after["score"].relative.score(mate_score=10000)

        cp_loss = (score_before - score_after) if (score_before is not None and score_after is not None) else 0
        classification = classify_move(abs(cp_loss))

        if classification == "Blunder":
            blunders += 1
        elif classification == "Mistake":
            mistakes += 1
        elif classification == "Inaccuracy":
            inaccuracies += 1

        move_data.append({
            "Move Number": move_number,
            "Move": move,
            "CP Loss": cp_loss,
            "Classification": classification
        })

        total_cp_loss += abs(cp_loss)
        move_number += 1

    engine.quit()

    total_moves = len(move_data)
    good_moves = total_moves - blunders - mistakes - inaccuracies
    accuracy = round((good_moves / total_moves) * 100, 2) if total_moves > 0 else 0

    return {
        "player_color": player_color,
        "winner": winner,
        "total_moves": total_moves,
        "accuracy": accuracy,
        "blunders": blunders,
        "mistakes": mistakes,
        "inaccuracies": inaccuracies,
        "suggestions": suggest_from_counts(blunders, mistakes, inaccuracies),
        "worst_moves": sorted(move_data, key=lambda x: abs(x["CP Loss"]), reverse=True)[:5],
        "full_moves": move_data
    }
