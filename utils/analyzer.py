import chess
import chess.engine
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    logging.info("Starting analysis with engine at %s", engine_path)

    board = chess.Board()
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)

    move_data = []
    total_cp_loss = 0
    blunders = 0
    mistakes = 0
    inaccuracies = 0
    move_number = 1

    for uci_move in moves:
        if board.is_game_over():
            break

        try:
            move = board.parse_uci(uci_move)
        except Exception as e:
            logging.error("Invalid move format: %s", uci_move)
            continue

        if not move in board.legal_moves:
            logging.warning("Illegal move skipped: %s", uci_move)
            continue

        # Evaluate before move
        info_before = engine.analyse(board, chess.engine.Limit(depth=15))
        score_before = info_before["score"].relative.score(mate_score=10000)

        board.push(move)

        # Evaluate after move
        info_after = engine.analyse(board, chess.engine.Limit(depth=15))
        score_after = info_after["score"].relative.score(mate_score=10000)

        cp_loss = 0
        if score_before is not None and score_after is not None:
            cp_loss = score_before - score_after

        classification = classify_move(abs(cp_loss))

        if classification == "Blunder":
            blunders += 1
        elif classification == "Mistake":
            mistakes += 1
        elif classification == "Inaccuracy":
            inaccuracies += 1

        san = board.san(board.peek())  # last move in SAN

        logging.info("Move %d: %s | CP Loss: %s | Class: %s", move_number, san, cp_loss, classification)

        move_data.append({
            "Move Number": move_number,
            "Move": san,
            "CP Loss": cp_loss,
            "Classification": classification
        })

        total_cp_loss += abs(cp_loss)
        move_number += 1

    engine.quit()

    total_moves = len(move_data)
    good_moves = total_moves - blunders - mistakes - inaccuracies
    accuracy = round((good_moves / total_moves) * 100, 2) if total_moves > 0 else 0

    summary = {
        "player": player_color,
        "total_moves": total_moves,
        "accuracy": accuracy,
        "blunders": blunders,
        "mistakes": mistakes,
        "inaccuracies": inaccuracies,
        "winner": winner,
        "suggestions": suggest_from_counts(blunders, mistakes, inaccuracies),
        "worst_moves": sorted(move_data, key=lambda x: abs(x["CP Loss"]), reverse=True)[:5]
    }

    logging.info("Analysis complete: %d moves, %d blunders, %d mistakes, %d inaccuracies", 
                 total_moves, blunders, mistakes, inaccuracies)

    return summary
