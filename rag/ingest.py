from pathlib import Path

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "data" / "farming_docs"
DB_DIR = BASE_DIR / "rag" / "chroma_db"

CHUNK_SIZE = 800
OVERLAP = 150
BATCH = 500


def read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str):
    text = " ".join(text.split())
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + CHUNK_SIZE])
        start += CHUNK_SIZE - OVERLAP
    return [c for c in chunks if len(c) > 100]


def main():
    files = [p for p in DOCS_DIR.glob("*") if p.suffix.lower() in (".pdf", ".txt", ".md")]
    if not files:
        raise SystemExit(f"No documents found in {DOCS_DIR}")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        client.delete_collection("farming")
    except Exception:
        pass
    col = client.create_collection("farming", metadata={"hnsw:space": "cosine"})

    for f in files:
        chunks = chunk_text(read_file(f))
        if not chunks:
            print(f"Skipped (no text found): {f.name}")
            continue
        for i in range(0, len(chunks), BATCH):
            part = chunks[i:i + BATCH]
            col.add(
                ids=[f"{f.stem}-{i + j}" for j in range(len(part))],
                documents=part,
                embeddings=embedder.encode(part, normalize_embeddings=True).tolist(),
                metadatas=[{"source": f.name, "chunk": i + j} for j in range(len(part))],
            )
        print(f"{f.name}: {len(chunks)} chunks")

    print("Total chunks stored:", col.count())


if __name__ == "__main__":
    main()