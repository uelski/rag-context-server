from fastapi import FastAPI
from pathlib import Path
import os
import time
import tiktoken
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import pymupdf
from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv()

import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "rag-demo")

app = FastAPI(title="RAG Server")

client = OpenAI(api_key=OPENAI_API_KEY)

pc = Pinecone(api_key=PINECONE_API_KEY)
# create index if missing (serverless, adjust region if needed)
existing = {i["name"] for i in pc.list_indexes()}
if PINECONE_INDEX not in existing:
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
index = pc.Index(PINECONE_INDEX)

def parse_pdf(path: Path) -> list[dict]:
    """Return [{'page': int, 'text': str}, ...]"""
    pages = []
    doc = pymupdf.open(path)
    for i, page in enumerate(doc):
        text = page.get_text("text")
        if text and text.strip():
            pages.append({"page": i + 1, "text": text.strip()})
    return pages

def chunk_pages(pages: list[dict], max_tokens: int = 500, overlap: int = 60) -> list[dict]:
    """
    Token-aware concatenation and sliding-window chunking.
    Keeps page ranges in metadata.
    """
    enc = tiktoken.get_encoding("cl100k_base")
    # merge all pages but remember boundaries
    joined = []
    for p in pages:
        joined.append((p["page"], enc.encode(p["text"])))
    # flatten tokens with page markers
    tokens = []
    page_marks = []
    for page_no, toks in joined:
        start = len(tokens)
        tokens.extend(toks)
        end = len(tokens)
        page_marks.append((page_no, start, end))

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        window = tokens[start:end]

        # infer page range for this window
        win_start, win_end = start, end
        in_pages = [pm[0] for pm in page_marks if not (pm[2] <= win_start or pm[1] >= win_end)]
        page_start = min(in_pages) if in_pages else None
        page_end = max(in_pages) if in_pages else None

        text = enc.decode(window)
        if text.strip():
            chunks.append({
                "id": f"loc-{start}-{end}",
                "text": text.strip(),
                "page_start": page_start,
                "page_end": page_end,
            })

        if end == len(tokens):
            break
        start = max(end - overlap, 0)

    return chunks

def embed_texts(texts: list[str], model: str = "text-embedding-3-small", batch_size: int = 100) -> list[list[float]]:
    vecs: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        vecs.extend([d.embedding for d in resp.data])
        # tiny backoff
        time.sleep(0.05)
    return vecs

def upsert_chunks(chunks: list[dict], vectors: list[list[float]], doc_id: str):
    # Pinecone expects a list of dicts with id, values, metadata
    items = []
    for ch, vec in zip(chunks, vectors):
        items.append({
            "id": f"{doc_id}:{ch['id']}",
            "values": vec,
            "metadata": {
                "doc_id": doc_id,
                "text": ch["text"][:4000],  # keep metadata small
                "page_start": ch.get("page_start"),
                "page_end": ch.get("page_end"),
            }
        })
    # Batch upsert (Pinecone SDK handles chunking internally too)
    index.upsert(items)

# --- reuse OpenAI client, index, etc. from above

def embed_query(query: str, model: str = "text-embedding-3-small") -> list[float]:
    resp = client.embeddings.create(model=model, input=[query])
    return resp.data[0].embedding

def search_chunks(q_vec: list[float], top_k: int = 5, doc_id: str | None = None):
    filt = {"doc_id": {"$eq": doc_id}} if doc_id else None
    res = index.query(
        vector=q_vec,
        top_k=top_k,
        include_metadata=True,
        include_values=False,
        filter=filt,
    )
    # Pinecone v5 returns res.matches (list)
    out = []
    for m in res.matches or []:
        md = m.get("metadata", {}) if isinstance(m, dict) else getattr(m, "metadata", {})  # be permissive
        out.append({
            "id": m["id"] if isinstance(m, dict) else m.id,
            "score": m["score"] if isinstance(m, dict) else m.score,
            "text": (md.get("text") or "")[:400],  # short preview
            "page_start": md.get("page_start"),
            "page_end": md.get("page_end"),
            "doc_id": md.get("doc_id"),
        })
    return out

# --- end of helpers

# ------------ routes

@app.get("/health")
def health():
    return {"status": "ok"}

# basic post endpoint to read pdf and chunk it
@app.post("/test_read_pdf")
def test_read_pdf():
    """
    Ingest a local test.pdf located next to this main.py file:
    parse -> chunk -> embed -> upsert
    """
    pdf_path = Path(__file__).parent / "kandinsky.pdf"
    if not pdf_path.exists():
        return {"error": f"test.pdf not found at {pdf_path}"}

    pages = parse_pdf(pdf_path)
    chunks = chunk_pages(pages, max_tokens=500, overlap=60)
    vecs = embed_texts([c["text"] for c in chunks])
    doc_id = f"doc_{pdf_path.stem}"
    upsert_chunks(chunks, vecs, doc_id=doc_id)

    return {
        "file": str(pdf_path.name),
        "pages_parsed": len(pages),
        "chunks_upserted": len(chunks),
        "index": PINECONE_INDEX,
        "doc_id": doc_id,
    }

class QueryRequest(BaseModel):
    q: str
    k: int | None = 5
    doc_id: str | None = None

@app.post("/search_documents")
def search_documents(req: QueryRequest):
    q_vec = embed_query(req.q)
    hits = search_chunks(q_vec, top_k=req.k or 5, doc_id=req.doc_id)
    return {"query": req.q, "top_k": req.k or 5, "results": hits}
