import os
import re
import time
import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-120b"
ABSTENTION_PHRASE = "Insufficient evidence in the retrieved financial documents to answer this question."

SYSTEM_PROMPT = f"""You are FinHybridRAG, a strict citation-grounded financial Q&A assistant.
Follow these rules without exception:
1. Answer ONLY using facts explicitly stated in the provided Context Documents.
2. Cite the exact source document ID in square brackets like [Doc 376148] immediately after every claim or recommendation.
3. If the provided Context Documents do not contain enough information to answer the question, respond ONLY with:
   "{ABSTENTION_PHRASE}"
4. Never invent numbers, tax rules, or financial advice not present in the context.
5. Keep the response concise, well-structured (short paragraphs or bullet points), and easy to audit."""


def build_context_block(retrieved_docs: list[dict], max_chars_per_doc: int = 1200) -> str:
    blocks = []
    for doc in retrieved_docs:
        doc_id = str(doc["doc_id"])
        text = str(doc.get("text", ""))[:max_chars_per_doc].strip()
        blocks.append(f"[Doc {doc_id}]\n{text}")
    return "\n\n---\n\n".join(blocks)


def verify_citations(answer: str, retrieved_doc_ids: list[str]) -> dict:
    """
    Audits inline [Doc ID] citations against the retrieved top-K set.
    Detects abstentions, verified grounding, and hallucinated doc IDs.
    """
    if ABSTENTION_PHRASE.lower() in answer.lower() or "insufficient evidence" in answer.lower():
        return {
            "status": "ABSTAINED",
            "label": "GUARDRAIL TRIGGERED • ABSTAINED",
            "cited_ids": [],
            "valid_ids": [],
            "hallucinated_ids": [],
            "grounding_rate": 1.0,
        }

    raw_matches = re.findall(r"\bDoc(?:ument)?\s*#?(\d+)\b", answer, flags=re.IGNORECASE)
    # Deduplicate while preserving order
    cited_ids = list(dict.fromkeys(raw_matches))
    retrieved_set = {str(did) for did in retrieved_doc_ids}

    valid_ids = [cid for cid in cited_ids if cid in retrieved_set]
    hallucinated_ids = [cid for cid in cited_ids if cid not in retrieved_set]

    if not cited_ids:
        return {
            "status": "UNVERIFIED",
            "label": "WARNING • NO INLINE CITATIONS FOUND",
            "cited_ids": [],
            "valid_ids": [],
            "hallucinated_ids": [],
            "grounding_rate": 0.0,
        }

    if hallucinated_ids:
        return {
            "status": "HALLUCINATION",
            "label": f"ALERT • {len(hallucinated_ids)} INVALID CITATION(S) DETECTED",
            "cited_ids": cited_ids,
            "valid_ids": valid_ids,
            "hallucinated_ids": hallucinated_ids,
            "grounding_rate": round(len(valid_ids) / len(cited_ids), 3),
        }

    return {
        "status": "VERIFIED",
        "label": f"VERIFIED • {len(valid_ids)}/{len(cited_ids)} CITATIONS GROUNDED",
        "cited_ids": cited_ids,
        "valid_ids": valid_ids,
        "hallucinated_ids": [],
        "grounding_rate": 1.0,
    }


def generate_grounded_answer(
    query: str,
    retrieved_docs: list[dict],
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
) -> dict:
    key = (api_key or os.getenv("GROQ_API_KEY", "")).strip()
    retrieved_ids = [str(d["doc_id"]) for d in retrieved_docs]
    context_str = build_context_block(retrieved_docs)

    if not key:
        return {
            "answer": (
                "No GROQ_API_KEY detected. Operating in retrieval-only mode — "
                "add a free Groq key in `.env` or the sidebar to enable grounded LLM synthesis."
            ),
            "model": "retrieval-only",
            "latency_ms": 0.0,
            "audit": {
                "status": "RETRIEVAL_ONLY",
                "label": "RETRIEVAL ONLY • NO API KEY",
                "cited_ids": [],
                "valid_ids": [],
                "hallucinated_ids": [],
                "grounding_rate": 0.0,
            },
            "error": None,
        }

    user_prompt = (
        f"Context Documents:\n{context_str}\n\n"
        f"User Question: {query}\n\n"
        f"Provide a grounded answer with inline [Doc ID] citations:"
    )

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "temperature": 0.0,
        "seed": 42,
        "max_completion_tokens": 1024,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }
    if model.startswith("openai/gpt-oss"):
        payload["reasoning_effort"] = "low"

    t0 = time.perf_counter()
    try:
        resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
        if not resp.ok:
            return {
                "answer": "",
                "model": model,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                "audit": None,
                "error": f"LLM API Error ({resp.status_code}): {resp.text}",
            }
        data = resp.json()
        answer = (data["choices"][0]["message"].get("content") or "").strip()
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        audit = verify_citations(answer, retrieved_ids)
        return {
            "answer": answer,
            "model": model,
            "latency_ms": latency_ms,
            "audit": audit,
            "error": None,
        }
    except requests.RequestException as exc:
        return {
            "answer": "",
            "model": model,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            "audit": None,
            "error": f"LLM API Error: {exc}",
        }


if __name__ == "__main__":
    import pandas as pd

    try:
        from src.bm25_retriever import BM25Retriever
        from src.dense_retriever import DenseRetriever
    except ImportError:
        from bm25_retriever import BM25Retriever
        from dense_retriever import DenseRetriever

    bm25 = BM25Retriever()
    bm25.load_index()
    dense = DenseRetriever(model_key="bge")
    dense.load_index()

    corpus_df = pd.read_parquet("data/corpus.parquet")
    id_col = "doc_id" if "doc_id" in corpus_df.columns else "_id"
    id_to_text = dict(zip(corpus_df[id_col].astype(str), corpus_df["text"].fillna("").astype(str)))

    test_q = "Where should I park my rainy day emergency funds?"
    b_hits = bm25.search(test_q, k=50)
    d_hits = dense.search(test_q, k=50)

    # Optimal Weighted RRF (alpha=0.95)
    scores = {}
    for h in b_hits:
        scores[h["doc_id"]] = scores.get(h["doc_id"], 0.0) + (0.05 / (60 + h["rank"]))
    for h in d_hits:
        scores[h["doc_id"]] = scores.get(h["doc_id"], 0.0) + (0.95 / (60 + h["rank"]))

    top5_ids = [did for did, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]]
    top5_docs = [{"doc_id": did, "text": id_to_text.get(did, "")} for did in top5_ids]

    res = generate_grounded_answer(test_q, top5_docs)
    print(f"\nQuery  : {test_q}")
    print(f"Model  : {res['model']} | LLM Latency: {res['latency_ms']} ms")
    if res["error"]:
        print("Error  :", res["error"])
    else:
        print(f"Audit  : {res['audit']['label']} | Cited: {res['audit']['cited_ids']}")
        print("\n--- Grounded Answer ---\n")
        print(res["answer"])