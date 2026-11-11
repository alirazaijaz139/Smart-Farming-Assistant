from pathlib import Path

import chromadb
from fastapi import APIRouter, Depends, HTTPException
from groq import Groq
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from .. import models
from ..config import GROQ_API_KEY, GROQ_MODEL
from ..security import get_current_user

router = APIRouter(prefix="/rag", tags=["Farming Q&A (RAG)"])

BASE_DIR = Path(__file__).resolve().parent.parent.parent
embedder = SentenceTransformer("all-MiniLM-L6-v2")
collection = chromadb.PersistentClient(
    path=str(BASE_DIR / "rag" / "chroma_db")
).get_or_create_collection("farming", metadata={"hnsw:space": "cosine"})
llm = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

SYSTEM_PROMPT = (
    "You are a farming assistant. Answer ONLY using the provided context. "
    "If the context does not contain the answer, say you don't know and suggest "
    "asking a local agriculture expert. Keep answers short and practical."
)


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=500)


@router.post("/ask")
def ask(body: Question, user: models.User = Depends(get_current_user)):
    if llm is None:
        raise HTTPException(500, "GROQ_API_KEY is not set")
    if collection.count() == 0:
        raise HTTPException(503, "Knowledge base is empty. Run: python rag/ingest.py")

    q_emb = embedder.encode([body.question], normalize_embeddings=True).tolist()
    res = collection.query(query_embeddings=q_emb, n_results=4)
    docs, metas = res["documents"][0], res["metadatas"][0]

    context = "\n\n".join(f"[{i + 1}] ({m['source']}) {d}" for i, (d, m) in enumerate(zip(docs, metas)))
    reply = llm.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {body.question}"},
        ],
    )
    return {
        "answer": reply.choices[0].message.content,
        "sources": [{"source": m["source"], "chunk": m["chunk"]} for m in metas],
    }