# manifest_gcs.py
import json, time, os
from typing import List, Dict, Any
from google.cloud import storage
from dotenv import load_dotenv
load_dotenv()

_BUCKET = os.getenv("GCS_BUCKET")
_client = storage.Client()
_bucket = _client.bucket(_BUCKET)

def _blob_for_sid(sid: str):
    # 1 object per user/session
    return _bucket.blob(f"uploads/{sid}.json")

def add_upload(sid: str, doc_id: str, filename: str, bytes_len: int, pages: int, chunks: int):
    blob = _blob_for_sid(sid)
    # read current list (if exists)
    try:
        data = json.loads(blob.download_as_text())
        files = data.get("files", [])
    except Exception:
        files = []
    files.insert(0, {
        "doc_id": doc_id,
        "filename": filename,
        "bytes": bytes_len,
        "pages": pages,
        "chunks": chunks,
        "created_at": time.time(),
    })
    blob.upload_from_string(json.dumps({"files": files}, ensure_ascii=False), content_type="application/json")

def list_uploads(sid: str) -> Dict[str, Any]:
    blob = _blob_for_sid(sid)
    if not blob.exists():
        return {"files": []}
    try:
        return json.loads(blob.download_as_text())
    except Exception:
        return {"files": []}
