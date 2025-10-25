from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
from typing import List
import uvicorn
import json
from datetime import datetime

app = FastAPI()

# CORS - Allow React frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load AI models on startup
print("🤖 Loading AI models... (this takes ~30 seconds)")
toxicity_model = pipeline("text-classification", model="unitary/toxic-bert")
sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
print("✅ Models loaded successfully!")

# WebSocket connection manager for real-time features
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)

manager = ConnectionManager()

# Routes
@app.get("/")
def root():
    return {
        "message": "AI Threads Backend Running 🚀",
        "endpoints": {
            "moderate": "/api/moderate",
            "sentiment": "/api/sentiment",
            "summarize": "/api/summarize"
        }
    }

@app.post("/api/moderate")
async def moderate_content(data: dict):
    """Check if text is toxic/offensive"""
    text = data.get("text", "")
    
    if not text:
        return {"error": "No text provided"}
    
    # Limit to 512 tokens (BERT limit)
    result = toxicity_model(text[:512])[0]
    
    is_toxic = result['score'] > 0.7  # 70% threshold
    
    return {
        "is_toxic": is_toxic,
        "confidence": round(result['score'], 3),
        "label": result['label'],
        "text_length": len(text)
    }

@app.post("/api/sentiment")
async def analyze_sentiment(data: dict):
    """Analyze sentiment of text"""
    text = data.get("text", "")
    
    if not text:
        return {"error": "No text provided"}
    
    result = sentiment_model(text[:512])[0]
    
    # Map to emoji for UI
    sentiment_emoji = {
        "POSITIVE": "😊",
        "NEGATIVE": "😠",
        "NEUTRAL": "😐"
    }
    
    return {
        "sentiment": result['label'],
        "confidence": round(result['score'], 3),
        "emoji": sentiment_emoji.get(result['label'], "😐")
    }

@app.post("/api/summarize")
async def summarize_thread(data: dict):
    """Summarize a thread (placeholder - you can add BART model later)"""
    posts = data.get("posts", [])
    
    if not posts:
        return {"error": "No posts provided"}
    
    # Simple summarization for now (first/last posts)
    summary = f"Thread with {len(posts)} posts discussing various topics."
    
    return {
        "summary": summary,
        "post_count": len(posts)
    }

# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Broadcast to all connected clients
            await manager.broadcast({
                "type": message.get("type", "message"),
                "content": message.get("content"),
                "timestamp": datetime.now().isoformat(),
                "user": message.get("user", "Anonymous")
            })
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
