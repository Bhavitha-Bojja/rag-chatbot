from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_logic import load_vector_store, answer_question

app = FastAPI()

vector_store = load_vector_store()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # we will tighten this later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(req: ChatRequest):
    answer = answer_question(req.message, vector_store)
    return {"answer": answer}