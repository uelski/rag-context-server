from fastapi import FastAPI

app = FastAPI(title="RAG Server")

@app.get("/health")
def health():
    return {"status": "ok"}