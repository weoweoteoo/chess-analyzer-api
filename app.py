import chess
import chess.engine
import logging

# Setup logging for analyzer (if not already)
logger = logging.getLogger(_name_)
logging.basicConfig(level=logging.INFO)

def classify_move(cp_loss):
    """
    Classify the move based on the CP (centipawn) loss.
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

def analyze_game(moves, player_color, winner, engine_path):
    board = chess.Board()
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)

    move_data = []
    blunders = 0
    mistakes = 0
    inaccuracies = 0
    total_cp_loss = 0

    player_color = player_color.lower()
    is_player_white = (player_color == "white")

    player_move_index = 0  # To count only player's moves

    for i, move in enumerate(moves):
        logger.info(f"Analyzing move {i+1}: {move}")

        try:
            chess_move = board.parse_san(move)
        except ValueError:
            try:
                chess_move = board.parse_uci(move)
            except ValueError:
                logger.warning(f"⚠ Invalid move at index {i}: {move} — skipping rest of game.")
                break

        try:
            info_before = engine.analyse(board, chess.engine.Limit(depth=15))
            score_before = info_before["score"].relative.score(mate_score=10000)
        except Exception as e:
            logger.warning(f"⚠ Could not evaluate position before move {move}: {e}")
            score_before = 0

        board.push(chess_move)

        try:
            info_after = engine.analyse(board, chess.engine.Limit(depth=15))
            score_after = info_after["score"].relative.score(mate_score=10000)
        except Exception as e:
            logger.warning(f"⚠ Could not evaluate position after move {move}: {e}")
            score_after = 0

        if score_before is None or score_after is None:
            cp_loss = 0
        else:
            cp_loss = score_before - score_after

        classification = classify_move(abs(cp_loss))

        move_entry = {
            "Move Number": (i // 2) + 1,
            "Move": move,
            "CP Loss": cp_loss,
            "Classification": classification
        }

        # Count mistakes only for the selected player's moves
        if (is_player_white and board.turn == chess.BLACK) or (not is_player_white and board.turn == chess.WHITE):
            if classification == "Blunder":
                blunders += 1
            elif classification == "Mistake":
                mistakes += 1
            elif classification == "Inaccuracy":
                inaccuracies += 1
            player_move_index += 1

        move_data.append(move_entry)
        total_cp_loss += abs(cp_loss)

    engine.quit()

    good_moves = player_move_index - blunders - mistakes - inaccuracies
    accuracy = round((good_moves / player_move_index) * 100, 2) if player_move_index > 0 else 0

    result = {
        "player_color": player_color,
        "winner": winner,
        "total_moves": len(move_data),
        "accuracy": accuracy,
        "blunders": blunders,
        "mistakes": mistakes,
        "inaccuracies": inaccuracies,
        "suggestions": suggest_from_counts(blunders, mistakes, inaccuracies),
        "worst_moves": sorted(move_data, key=lambda x: abs(x["CP Loss"]), reverse=True)[:5],
        "full_moves": move_data
    }

    logger.info("Final analysis result: %s", result)
    return result
