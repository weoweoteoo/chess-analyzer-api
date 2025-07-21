# ♟️ Chess Analyzer API

A Flask-based API that analyzes chess games using Stockfish and evaluates player performance with insights like accuracy, blunders, mistakes, and more.

## 🔧 Features

- 🧮 Calculates centipawn loss (CP loss) for each move
- 📊 Classifies moves as Best, Inaccuracy, Mistake, or Blunder
- 🎯 Computes overall accuracy and statistics
- 📝 Returns improvement suggestions based on performance
- 💾 Caches analysis results using player and match IDs
- 🌐 CORS-enabled for frontend integration

---

## 🚀 Live Demo

> Hosted at: [https://chess-analyzer-api-production.up.railway.app]([https://chess-analyzer-api-production.up.railway.app](https://chess-analyzer-api.onrender.com)])

---

## 🛠️ Tech Stack

- **Python** 3.10
- **Flask** (REST API)
- **python-chess**
- **Stockfish Engine**
- **Docker** (for deployment)
- **Render (for hosting)

---

## 📦 Installation

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/chess-analyzer-api.git
cd chess-analyzer-api


```
### 2. Install Dependencies
pip install -r requirements.txt

### Add Stockfish Engine
chmod +x engine/stockfish

### Usage
run python app.py




!!!! U CAN TEST IT USING POSTMAN...... with endpoint POST /api/analyze
{
  "moves": ["e4", "e5", "Nf3", "Nc6", ...],
  "playerColor": "white",
  "winner": "black",
  "playerId": "123",
  "matchId": "456"
}


