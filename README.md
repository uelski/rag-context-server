# RAG Server (FastAPI + OpenAI Embeddings + Pinecone)

Minimal backend for Retrieval-Augmented Generation (RAG): parse PDFs with PyMuPDF, embed chunks with OpenAI, store/search in Pinecone, craft responses with ChatGPT, and serve results via FastAPI.

The backend tracks the namespace for each individual user with a http-only cookie and saves the metadata associated with each sid created using json files saved in GCP Cloud Storage.

## Tech Stack
- **FastAPI** + **Uvicorn**
- **PyMuPDF**
- **OpenAI Embeddings** (e.g., `text-embedding-3-small`)
- **Pinecone** (vector DB)
- **Pydantic**, **Python 3.11+**
- Optional: **Docker**, deployable to **Cloud Run**
- **GCP Cloud Storage**

## API Endpoints
- `GET /health` — service health check
- `POST /upload_document` — This endpoint takes a file in the body of the request, parses the text using PyMuPDF, tokenizes and chunks the text, then passes the vector embeddings to the Pinecone index.<br>
  **Response (example)**:
  ```json
  {
    "file": "filename",
    "pages_parsed": 4,
    "chunks_upserted": 100,
    "index": "rag-api",
    "namespace": 1234,
    "doc_id": "file.filename",
    }
- `POST /search_documents` - This endpoint takes a string query, searches the vector db based on semantic similarity and returns the most relevant chunks in the original text.
- `POST /query_documents` - This endpoint takes a string query, searches the vector db, provides the relevant text chunks as context to the LLM and returns the response crafted by the language model.

## Deployment
This project is deployed through **GCP Cloud Run**, with the corresponding [frontend repo](https://github.com/uelski/rag-context-client) and live [site.](https://rag.uelski.dev/)
