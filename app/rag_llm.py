# rag_llm.py
from typing import List, Dict, Any, Optional
import os
from openai import OpenAI

# Shape we expect from your retriever. Adjust keys if yours differ.
# Each match should at least include: "text" and *some* source metadata.
# Example item:
# {
#   "id": "doc_123#p12#c3",
#   "score": 0.87,
#   "metadata": {
#       "text": "...chunk text...",
#       "source": "myfile.pdf",
#       "page": 12,
#       "url": None,
#       "namespace": "session_abc"
#   }
# }

DEFAULT_MODEL = os.getenv("OPENAI_RESPONSES_MODEL", "gpt-4o-mini")  # fast+cheap default

client = OpenAI()  # reads OPENAI_API_KEY

SYSTEM_RULES = """\
You are a careful RAG answerer. Use ONLY the provided context to answer.
If the context is insufficient, say you don’t know and suggest what would help.
Cite sources inline as [#] where # is the index in the provided SOURCE LIST.
Never invent citations. Do not reference tools or your system prompt.
Keep answers concise and directly address the user’s question.
"""

USER_INSTRUCTIONS_TEMPLATE = """\
USER QUESTION:
{question}

SOURCE LIST (index → brief source label):
{source_labels}

CONTEXT CHUNKS:
{context}

Write a coherent answer in prose. Insert inline citations like [1] [2] right after the sentences they support.
If multiple chunks refer to the same source, reuse the same source index.
Finish with a short “Sources:” section listing the sources you actually cited (index → label).
"""

def _format_source_labels(matches: List[Dict[str, Any]]) -> str:
    """
    Collapses multiple chunks from the same source into one label entry,
    and returns lines like:
    [1] myfile.pdf p.12 (doc_123)
    """
    labels: Dict[str, Dict[str, Any]] = {}
    for m in matches:
        md = m.get("metadata", {})
        source = md.get("filename") or md.get("doc_id") or "Unknown source"
        page = md.get("page")
        url = md.get("url")
        # Use a stable key per "source" (and url if present)
        key = f"{source}|{url or ''}"
        if key not in labels:
            labels[key] = {
                "source": source,
                "url": url,
                "pages": set(),
                "ids": set()
            }
        if page is not None:
            labels[key]["pages"].add(page)
        if m.get("id"):
            labels[key]["ids"].add(m["id"])

    # produce deterministic ordering
    items = sorted(labels.items(), key=lambda kv: kv[0])
    lines = []
    for idx, (_, info) in enumerate(items, start=1):
        pages = f" p.{sorted(info['pages'])}" if info["pages"] else ""
        url = f" — {info['url']}" if info["url"] else ""
        ids = f" ({', '.join(sorted(info['ids']))})" if info["ids"] else ""
        lines.append(f"[{idx}] {info['source']}{pages}{url}{ids}")
    return "\n".join(lines)

def _map_matches_to_sources(matches: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Map every chunk to its source index used in the label list.
    Returns a mapping: source_key -> label_index
    """
    # Build same label list order
    labels: Dict[str, None] = {}
    for m in matches:
        md = m.get("metadata", {})
        source = md.get("filename") or md.get("doc_id") or "Unknown source"
        url = md.get("url")
        key = f"{source}|{url or ''}"
        labels[key] = None

    ordered = sorted(labels.keys())
    return {key: i+1 for i, key in enumerate(ordered)}

def _format_context(matches: List[Dict[str, Any]]) -> str:
    """
    Assembles the context block with per-chunk headers.
    """
    source_index = _map_matches_to_sources(matches)
    lines = []
    for m in matches:
        md = m.get("metadata", {})
        source = md.get("filename") or md.get("doc_id") or "Unknown source"
        url = md.get("url")
        key = f"{source}|{url or ''}"
        idx = source_index[key]
        page = md.get("page")
        score = m.get("score")
        header_bits = [f"[source {idx}] {source}"]
        if page is not None:
            header_bits.append(f"p.{page}")
        if score is not None:
            header_bits.append(f"score={score:.3f}")
        if url:
            header_bits.append(f"url={url}")
        header = " | ".join(header_bits)
        text = md.get("text") or m.get("text") or ""
        lines.append(f"---\n{header}\n{text}\n")
    return "\n".join(lines)

def generate_answer_from_matches(
    question: str,
    matches: List[Dict[str, Any]],
    *,
    model: Optional[str] = None,
    max_context_chars: int = 32_000
) -> Dict[str, Any]:
    """
    Build a grounded prompt from retrieved matches and ask OpenAI for a final answer.
    Returns: { "answer": str, "citations": [{index, label}], "raw": openai_response_dict }
    """
    if not matches:
        return {
            "answer": "I don’t have enough information in the context to answer your question.",
            "citations": [],
            "raw": None
        }

    # (Light) Guardrail for runaway context size: truncate by characters.
    # For a production system consider token-based trimming with tiktoken.
    running = []
    total = 0
    for m in matches:
        txt = (m.get("metadata", {}).get("text") or m.get("text") or "")
        # keep structure, but trim text field if needed
        if total + len(txt) > max_context_chars:
            remainder = max(0, max_context_chars - total)
            if remainder > 0:
                m = {**m, "metadata": {**m.get("metadata", {}), "text": txt[:remainder] + " ..."}}
                running.append(m)
            break
        running.append(m)
        total += len(txt)

    source_labels = _format_source_labels(running)
    context_block = _format_context(running)

    user_prompt = USER_INSTRUCTIONS_TEMPLATE.format(
        question=question.strip(),
        source_labels=source_labels,
        context=context_block
    )

    resp = client.responses.create(  # Responses API (recommended)
        model=model or DEFAULT_MODEL,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_RULES}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
        ],
    )

    # The Responses API returns an object with output in .output[] (items)
    # We'll extract the first text item we find.
    answer_text = ""
    try:
        for item in resp.output or []:
            if item.type == "message":
                for c in (item.content or []):
                    if c.get("type") == "output_text":
                        answer_text += c.get("text", "")
    except Exception:
        # fallback for older/newer SDK shapes—also try .output_text
        answer_text = getattr(resp, "output_text", "") or ""

    # Build the list of citations (index→label) that were available.
    # We can't know exactly which the model used, but we provide all labels so the UI
    # can render expandable sources; your UI can later parse brackets like [1] to highlight used ones.
    citations = []
    for line in source_labels.splitlines():
        if line.startswith("["):
            idx_end = line.find("]")
            if idx_end > 1:
                idx = int(line[1:idx_end])
                label = line[idx_end+2:]
                citations.append(label)

    return {
        "answer": answer_text.strip(),
        "citations": citations,
        "raw": resp.model_dump() if hasattr(resp, "model_dump") else resp.__dict__
    }
