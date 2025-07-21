# ♟️ Chess AI API

An advanced Flask-based backend that uses a trained PyTorch model to play chess, suggest moves based on difficulty, and communicate with a frontend in real time via WebSockets.

---

## 🚀 Features

- 🤖 AI-generated chess moves using a custom-trained model
- 📶 Real-time move suggestions via Socket.IO
- 🔀 Difficulty modes: Beginner, Intermediate, Expert
- 💬 Human move suggestion for learning and comparison
- 🧠 Centipawn evaluation of each move
- 📦 Model loading with health-check endpoint
- 🔄 CORS and WebSocket enabled for frontend communication

---

## 🌐 Live Deployment 

> API is deployed at: `https://chess-ai-webs.onrender.com`

---

## 🛠️ Tech Stack

- Python 3.10
- Flask + Flask-SocketIO
- PyTorch (CPU version)
- python-chess (board logic)
- Trained `.pt` model (TorchScript)
- WebSockets for real-time game play

---

## 🧪 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/weoweoteoo/chess-ai-api.git
cd chess-ai-api
```
### 2. Install Dependencies
pip install -r requirements.txt

### 3. Add models
models/traced_chess_model.pt 
### 4. Run
gunicorn -k eventlet -w 1 app:app --bind 0.0.0.0:10000


