import chess
import chess.pgn
import chess.engine
import io

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

def analyze_game_from_moves(moves, player_color, winner):
    game = chess.pgn.Game()
    board = game.board()

    for move_str in moves:
        try:
            move = board.push_san(move_str)
        except ValueError:
            raise ValueError(f"Invalid move: {move_str}")

    engine_path = "engine/stockfish"
    analysis = []
    total_cp_loss = 0
    blunders = 0
    mistakes = 0
    inaccuracies = 0
    move_number = 1

    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        board = game.board()
        for move_str in moves:
            move = board.push_san(move_str)
            info_before = engine.analyse(board, chess.engine.Limit(depth=15))
            score_before = info_before["score"].white().score(mate_score=10000)

            info_after = engine.analyse(board, chess.engine.Limit(depth=15))
            score_after = info_after["score"].white().score(mate_score=10000)

            cp_loss = (score_before - score_after) if score_before is not None and score_after is not None else 0
            classification = classify_move(abs(cp_loss))

            if classification == "Blunder":
                blunders += 1
            elif classification == "Mistake":
                mistakes += 1
            elif classification == "Inaccuracy":
                inaccuracies += 1

            analysis.append({
                "Move Number": move_number,
                "Move": move_str,
                "CP Loss": cp_loss,
                "Classification": classification
            })

            total_cp_loss += abs(cp_loss)
            move_number += 1

        engine.quit()

    total_moves = len(analysis)
    good_moves = total_moves - blunders - mistakes - inaccuracies
    accuracy = round((good_moves / total_moves) * 100, 2) if total_moves > 0 else 0

    return {
        "playerColor": player_color,
        "winner": winner,
        "totalMoves": total_moves,
        "accuracy": accuracy,
        "blunders": blunders,
        "mistakes": mistakes,
        "inaccuracies": inaccuracies,
        "suggestions": suggest_from_counts(blunders, mistakes, inaccuracies),
        "worstMoves": sorted(analysis, key=lambda x: abs(x["CP Loss"]), reverse=True)[:5],
        "analysis": analysis
    }
