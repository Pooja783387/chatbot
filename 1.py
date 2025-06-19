from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import pandas as pd

app = FastAPI()

# CORS Settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount public/ folder for static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="public"), name="static")

@app.get("/")
def read_index():
    return FileResponse("public/index.html")

# Load model and CSV
model = SentenceTransformer("all-MiniLM-L6-v2")
df = pd.read_csv("backend/10.csv", header=None, names=["Questions", "Answers"])

questions = df["Questions"].astype(str).tolist()
answers = df["Answers"].astype(str).tolist()
combined_texts = [f"Q: {q} A: {a}" for q, a in zip(questions, answers)]
combined_embeddings = model.encode(combined_texts, convert_to_tensor=True)

greeting_responses = {
    "hi": "Hello! How can I help you today?",
    "hello": "Hi there! Ask me anything.",
    "thanks": "You're welcome!",
    "bye": "Goodbye!",
    "how are you": "I'm a bot, but doing great. You?"
}
greeting_phrases = list(greeting_responses.keys())
greeting_embeddings = model.encode(greeting_phrases, convert_to_tensor=True)

class Query(BaseModel):
    query: str

@app.post("/ask")
def ask_question(data: Query):
    user_input = data.query.strip().lower()
    user_embedding = model.encode(user_input, convert_to_tensor=True)

    greet_score = util.cos_sim(user_embedding, greeting_embeddings).max().item()
    if greet_score > 0.3:
        idx = util.cos_sim(user_embedding, greeting_embeddings).argmax().item()
        return {"answer": greeting_responses[greeting_phrases[idx]]}

    scores = util.cos_sim(user_embedding, combined_embeddings)
    max_score = scores.max().item()
    if max_score < 0.4:
        return {"answer": "Sorry, I couldn't find a relevant answer for that."}
    top_idx = scores.argmax().item()
    return {"answer": answers[top_idx]}
