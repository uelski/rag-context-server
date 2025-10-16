from fastapi import FastAPI, UploadFile, File, Depends, Request, Body, HTTPException
from pathlib import Path
from tempfile import NamedTemporaryFile
import os
import time
import tiktoken
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import pymupdf
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
from uuid import uuid4
from gcs_manifest import add_upload, list_uploads
from .rag_llm import generate_answer_from_matches
import shutil

from dotenv import load_dotenv
load_dotenv()


APP_COOKIE = "sid"

def gen_sid() -> str:
    return uuid4().hex  # opaque, unguessable

class AnonSessionMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        sid = request.cookies.get(APP_COOKIE) or gen_sid()
        # expose to handlers
        request.state.session_id = sid
        # run downstream
        response = await call_next(request)
        # (re)issue cookie if missing
        if not request.cookies.get(APP_COOKIE):
            response.set_cookie(
                key=APP_COOKIE,
                value=sid,
                httponly=True,
                secure=True,                 # set False for local HTTP if needed
                samesite="lax",
                max_age=60 * 60 * 24 * 7,   # 7 days
                path="/",
            )
        return response

# ---- OpenAI, Pinecone, etc.

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "rag-demo")

app = FastAPI(title="RAG Server")

# CORS (important: allow credentials)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # your Vite dev
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(AnonSessionMiddleware)

def get_session_id(request: Request) -> str:
    # middleware put it here
    return request.state.session_id

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

def parse_pdf(file: UploadFile) -> list[dict]:
    """Return [{'page': int, 'text': str}, ...]"""
    with NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
        # copy in chunks from the underlying sync file object
        file.file.seek(0)
        shutil.copyfileobj(file.file, tmp)   # zero extra buffering in Python
        tmp.flush()

        # Open by path
        doc = pymupdf.open(tmp.name)
        try:
            pages = []
            for i, page in enumerate(doc):
                txt = (page.get_text("text") or "").strip()
                if txt:
                    pages.append({"page": i + 1, "text": txt})
        finally:
            doc.close()
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

def upsert_chunks(chunks: list[dict], vectors: list[list[float]], doc_id: str, namespace: str | None = None):
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
    index.upsert(vectors=items, namespace=namespace)

# --- reuse OpenAI client, index, etc. from above

def embed_query(query: str, model: str = "text-embedding-3-small") -> list[float]:
    resp = client.embeddings.create(model=model, input=[query])
    return resp.data[0].embedding

def search_chunks(q_vec: list[float], top_k: int = 5, doc_id: str | None = None, namespace: str | None = None):
    filt = {"doc_id": {"$eq": doc_id}} if doc_id else None
    res = index.query(
        vector=q_vec,
        top_k=top_k,
        include_metadata=True,
        include_values=False,
        filter=filt,
        namespace=namespace,
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
@app.post("/upload_document")
async def upload_document(sid: str = Depends(get_session_id), file: UploadFile = File(...)):
    """
    Ingest a local test.pdf located next to this main.py file:
    parse -> chunk -> embed -> upsert
    """
    # 1) read file content
    raw = await file.read()
    if not file.filename:
        raise HTTPException(400, "No file")

    # 2) parse -> chunk -> embed
    # Make sure parse_pdf accepts bytes (or adapt accordingly)
    pages = parse_pdf(file) 
        
    if not pages:
        raise HTTPException(status_code=400, detail="No text extracted from file.")
    
    chunks = chunk_pages(pages, max_tokens=500, overlap=60)
    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks produced from document.")

    vecs = embed_texts([c["text"] for c in chunks])   # must match your index dimension (e.g., 1536)

    # 3) build a doc_id and (optionally) enrich metadata
    doc_id = f"doc_{file.filename}"
    for i, c in enumerate(chunks):
        c.setdefault("metadata", {})
        c["metadata"].update({
            "user_id": sid,                # optional but useful
            "filename": file.filename or "",
            "doc_id": doc_id,
            "chunk_index": i,
        })

    # 4) upsert to Pinecone scoped to this user's namespace
    upsert_chunks(chunks, vecs, doc_id=doc_id, namespace=sid)
    add_upload(sid, doc_id, file.filename, len(raw), len(pages), len(chunks))
    return {
        "file": str(file.filename),
        "pages_parsed": len(pages),
        "chunks_upserted": len(chunks),
        "index": PINECONE_INDEX,
        "namespace": sid,
        "doc_id": doc_id,
    }

class QueryRequest(BaseModel):
    q: str
    k: int | None = 5
    doc_id: str | None = None

def has_vectors(sid: str = Depends(get_session_id)):
    stats = index.describe_index_stats()
    ns = stats.get("namespaces", {}).get(sid, {})
    count = int(ns.get("vectorCount", 0))
    return {"has_any": count > 0, "count": count}

def retrieve_matches(namespace: str, query: str, top_k: int, doc_id: str | None = None) -> list[dict[str, any]]:
    q_vec = embed_query(query)  # your existing function
    hits = search_chunks(q_vec, top_k=top_k, doc_id=doc_id, namespace=namespace)  # your existing function
    # Ensure each hit includes metadata.text/source/page for rag_llm
    return hits

@app.post("/search_documents")
def search_documents(req: QueryRequest, sid: str = Depends(get_session_id), vecstat: dict = Depends(has_vectors)):
    if not vecstat["has_any"]:
        return JSONResponse(
            {"results": [], "info": "No uploads found for this session.", "count": 0},
            status_code=200
        )
    hits = retrieve_matches(sid, req.q, req.k or 5, req.doc_id)
    return {"query": req.q, "top_k": req.k or 5, "results": hits}

class QueryDocumentsIn(BaseModel):
    query: str
    top_k: int = 6
    model: str | None = None
    doc_id: str | None = None

class QueryDocumentsOut(BaseModel):
    answer: str
    citations: list[dict]

@app.post("/query_documents", response_model=QueryDocumentsOut)
def query_documents(
    body: QueryDocumentsIn,
    sid: str = Depends(get_session_id),
    vecstat: dict = Depends(has_vectors)
):
    if not vecstat["has_any"]:
        return QueryDocumentsOut(answer="No uploads found for this session.", citations=[])
    matches = retrieve_matches(namespace=sid, query=body.query, top_k=body.top_k, doc_id=body.doc_id)
    result = generate_answer_from_matches(question=body.query, matches=matches, model=body.model)
    return QueryDocumentsOut(answer=result["answer"], citations=result["citations"])

@app.get("/uploads")
def get_uploads(sid: str = Depends(get_session_id)):
    return list_uploads(sid)
