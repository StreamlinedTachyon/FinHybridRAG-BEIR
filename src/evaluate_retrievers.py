import json
import math
import time
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from src.bm25_retriever import BM25Retriever
    from src.dense_retriever import DenseRetriever
    from src.hybrid_reranker import reciprocal_rank_fusion
except ImportError:
    from bm25_retriever import BM25Retriever
    from dense_retriever import DenseRetriever
    from hybrid_reranker import reciprocal_rank_fusion

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"None of {candidates} found in columns: {list(df.columns)}")


def load_eval_data():
    corpus_df = pd.read_parquet(DATA_DIR / "corpus.parquet")
    c_id_col = pick_col(corpus_df, ["doc_id", "_id", "corpus-id"])
    id_to_text = dict(
        zip(corpus_df[c_id_col].astype(str), corpus_df["text"].fillna("").astype(str))
    )

    qrels_df = pd.read_parquet(DATA_DIR / "qrels.parquet")
    q_col = pick_col(qrels_df, ["query_id", "query-id", "_id"])
    d_col = pick_col(qrels_df, ["doc_id", "corpus-id"])
    s_col = pick_col(qrels_df, ["score", "relevance"])

    qrels: dict[str, dict[str, int]] = {}
    for qid, did, score in zip(
        qrels_df[q_col].astype(str),
        qrels_df[d_col].astype(str),
        qrels_df[s_col].astype(int),
    ):
        if score > 0:
            qrels.setdefault(qid, {})[did] = int(score)

    test_q_path = DATA_DIR / "test_queries.parquet"
    if test_q_path.exists():
        queries_df = pd.read_parquet(test_q_path)
    else:
        queries_df = pd.read_parquet(DATA_DIR / "queries.parquet")

    q_id_col = pick_col(queries_df, ["query_id", "_id", "query-id"])
    queries_df[q_id_col] = queries_df[q_id_col].astype(str)
    queries_df = queries_df[queries_df[q_id_col].isin(qrels)].reset_index(drop=True)

    queries = list(
        zip(queries_df[q_id_col].tolist(), queries_df["text"].fillna("").astype(str).tolist())
    )
    return id_to_text, qrels, queries


def compute_metrics_single(ranked_doc_ids: list[str], rel_map: dict[str, int], k: int = 10):
    top_k = ranked_doc_ids[:k]

    # DCG@k and IDCG@k
    dcg = 0.0
    for rank, did in enumerate(top_k, start=1):
        rel = rel_map.get(did, 0)
        if rel > 0:
            dcg += (2**rel - 1) / math.log2(rank + 1)

    ideal_rels = sorted(rel_map.values(), reverse=True)[:k]
    idcg = sum((2**rel - 1) / math.log2(rank + 1) for rank, rel in enumerate(ideal_rels, start=1))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    # MRR@k
    mrr = 0.0
    for rank, did in enumerate(top_k, start=1):
        if rel_map.get(did, 0) > 0:
            mrr = 1.0 / rank
            break

    # Recall@k
    hits = sum(1 for did in top_k if rel_map.get(did, 0) > 0)
    recall = hits / len(rel_map) if rel_map else 0.0

    return ndcg, mrr, recall


def evaluate_predictions(
    preds: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    latencies_ms: list[float],
    name: str,
    k: int = 10,
) -> dict:
    ndcgs, mrrs, recalls = [], [], []
    for qid, rel_map in qrels.items():
        if qid not in preds:
            continue
        ndcg, mrr, recall = compute_metrics_single(preds[qid], rel_map, k=k)
        ndcgs.append(ndcg)
        mrrs.append(mrr)
        recalls.append(recall)

    return {
        "System": name,
        "Queries": len(ndcgs),
        "NDCG@10": round(float(np.mean(ndcgs)), 4),
        "MRR@10": round(float(np.mean(mrrs)), 4),
        "Recall@10": round(float(np.mean(recalls)), 4),
        "Mean_Latency_ms": round(float(np.mean(latencies_ms)), 2),
        "P95_Latency_ms": round(float(np.percentile(latencies_ms, 95)), 2),
    }


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    id_to_text, qrels, queries = load_eval_data()
    print(f"Loaded {len(queries)} ground-truth test queries across {len(id_to_text):,} docs.")

    # Load all three base retrievers
    bm25 = BM25Retriever()
    bm25.load_index()

    dense_minilm = DenseRetriever(model_key="minilm")
    dense_minilm.load_index()

    dense_bge = DenseRetriever(model_key="bge")
    dense_bge.load_index()

    # Warm-up (avoids first-call CPU jitter in latency stats)
    for _, qtext in queries[:3]:
        bm25.search(qtext, k=10)
        dense_minilm.search(qtext, k=10)
        dense_bge.search(qtext, k=10)

    # 1. Run BM25 (top-100)
    bm25_preds, bm25_lat = {}, []
    for qid, qtext in tqdm(queries, desc="Evaluating BM25"):
        t0 = time.perf_counter()
        hits = bm25.search(qtext, k=100)
        bm25_lat.append((time.perf_counter() - t0) * 1000)
        bm25_preds[qid] = [h["doc_id"] for h in hits]

    # 2. Run Dense MiniLM (top-100)
    minilm_preds, minilm_lat = {}, []
    for qid, qtext in tqdm(queries, desc="Evaluating Dense MiniLM"):
        t0 = time.perf_counter()
        hits = dense_minilm.search(qtext, k=100)
        minilm_lat.append((time.perf_counter() - t0) * 1000)
        minilm_preds[qid] = [h["doc_id"] for h in hits]

    # 3. Run Dense BGE (top-100)
    bge_preds, bge_lat = {}, []
    for qid, qtext in tqdm(queries, desc="Evaluating Dense BGE"):
        t0 = time.perf_counter()
        hits = dense_bge.search(qtext, k=100)
        bge_lat.append((time.perf_counter() - t0) * 1000)
        bge_preds[qid] = [h["doc_id"] for h in hits]

    # 4. Hybrid RRF (BM25 + MiniLM, k=60, pool=50)
    hyb_minilm_preds, hyb_minilm_lat = {}, []
    for idx, (qid, _) in enumerate(queries):
        t0 = time.perf_counter()
        fused = reciprocal_rank_fusion([bm25_preds[qid][:50], minilm_preds[qid][:50]], rrf_k=60)
        rrf_ms = (time.perf_counter() - t0) * 1000
        hyb_minilm_preds[qid] = [doc_id for doc_id, _ in fused[:100]]
        hyb_minilm_lat.append(bm25_lat[idx] + minilm_lat[idx] + rrf_ms)

    # 5. Hybrid RRF (BM25 + BGE, k=60, pool=100 for rerank export & pool=50 for fair comparison)
    hyb_bge_preds, hyb_bge_lat = {}, []
    for idx, (qid, _) in enumerate(queries):
        t0 = time.perf_counter()
        fused = reciprocal_rank_fusion([bm25_preds[qid][:50], bge_preds[qid][:50]], rrf_k=60)
        rrf_ms = (time.perf_counter() - t0) * 1000
        hyb_bge_preds[qid] = [doc_id for doc_id, _ in fused[:100]]
        hyb_bge_lat.append(bm25_lat[idx] + bge_lat[idx] + rrf_ms)

    # Summary Table (Stages 1 to 3)
    summary_rows = [
        evaluate_predictions(bm25_preds, qrels, bm25_lat, "1. BM25 (Sparse)"),
        evaluate_predictions(minilm_preds, qrels, minilm_lat, "2a. Dense (all-MiniLM-L6-v2)"),
        evaluate_predictions(bge_preds, qrels, bge_lat, "2b. Dense (bge-small-en-v1.5)"),
        evaluate_predictions(hyb_minilm_preds, qrels, hyb_minilm_lat, "3a. Hybrid RRF (BM25 + MiniLM)"),
        evaluate_predictions(hyb_bge_preds, qrels, hyb_bge_lat, "3b. Hybrid RRF (BM25 + BGE)"),
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULTS_DIR / "stage1_to_3_metrics.csv", index=False)

    # 6. RRF Smoothing Constant k Sensitivity Study (BM25 + BGE)
    rrf_rows = []
    for rrf_k in [1, 10, 30, 60, 100, 150]:
        preds_k, lats_k = {}, []
        for idx, (qid, _) in enumerate(queries):
            t0 = time.perf_counter()
            fused = reciprocal_rank_fusion([bm25_preds[qid][:50], bge_preds[qid][:50]], rrf_k=rrf_k)
            rrf_ms = (time.perf_counter() - t0) * 1000
            preds_k[qid] = [doc_id for doc_id, _ in fused[:10]]
            lats_k.append(bm25_lat[idx] + bge_lat[idx] + rrf_ms)
        row = evaluate_predictions(preds_k, qrels, lats_k, f"RRF k={rrf_k}")
        row["rrf_k"] = rrf_k
        rrf_rows.append(row)

    rrf_df = pd.DataFrame(rrf_rows)
    rrf_df.to_csv(RESULTS_DIR / "rrf_k_sensitivity.csv", index=False)

    # 7. Export Top-100 Hybrid (BM25 + BGE) Candidates for Fast Colab GPU Reranking
    print("\nExporting top-100 candidates for fast Colab Cross-Encoder reranking...")
    export_queries = []
    unique_doc_ids = set()

    for idx, (qid, qtext) in enumerate(queries):
        fused_100 = reciprocal_rank_fusion([bm25_preds[qid][:100], bge_preds[qid][:100]], rrf_k=60)[:100]
        cand_ids = [doc_id for doc_id, _ in fused_100]
        unique_doc_ids.update(cand_ids)
        export_queries.append(
            {
                "query_id": qid,
                "query_text": qtext,
                "bm25_top10": bm25_preds[qid][:10],
                "bge_top10": bge_preds[qid][:10],
                "hybrid_top100": cand_ids,
                "hybrid_base_latency_ms": round(float(hyb_bge_lat[idx]), 2),
            }
        )

    # Cap doc text at 2000 chars (Cross-Encoder max_length=512 tokens) to keep file compact
    doc_texts = {did: id_to_text.get(did, "")[:2000] for did in unique_doc_ids}
    payload = {
        "queries": export_queries,
        "doc_texts": doc_texts,
        "qrels": qrels,
    }
    out_json = RESULTS_DIR / "rerank_candidates_top100.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    size_mb = out_json.stat().st_size / (1024 * 1024)
    print(f"Exported {out_json} ({size_mb:.1f} MB, {len(unique_doc_ids):,} unique docs).")

    print("\n=== BENCHMARK RESULTS (STAGES 1-3, FULL 648 QUERIES) ===")
    print(summary_df.to_string(index=False))

    print("\n=== RRF k SENSITIVITY (BM25 + BGE) ===")
    print(rrf_df[["rrf_k", "NDCG@10", "MRR@10", "Recall@10"]].to_string(index=False))


if __name__ == "__main__":
    main()