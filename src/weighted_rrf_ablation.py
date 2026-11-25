import json
import math
import time
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from src.bm25_retriever import BM25Retriever
    from src.dense_retriever import DenseRetriever
except ImportError:
    from bm25_retriever import BM25Retriever
    from dense_retriever import DenseRetriever

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"None of {candidates} found in {list(df.columns)}")


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
    queries_df = pd.read_parquet(test_q_path if test_q_path.exists() else DATA_DIR / "queries.parquet")
    q_id_col = pick_col(queries_df, ["query_id", "_id", "query-id"])
    queries_df[q_id_col] = queries_df[q_id_col].astype(str)
    queries_df = queries_df[queries_df[q_id_col].isin(qrels)].reset_index(drop=True)

    queries = list(
        zip(queries_df[q_id_col].tolist(), queries_df["text"].fillna("").astype(str).tolist())
    )
    return id_to_text, qrels, queries


def weighted_rrf(bm25_ids: list[str], bge_ids: list[str], alpha: float, rrf_k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    w_bm25 = 1.0 - alpha
    w_bge = alpha

    if w_bm25 > 0:
        for rank, did in enumerate(bm25_ids, start=1):
            scores[did] = scores.get(did, 0.0) + (w_bm25 / (rrf_k + rank))
    if w_bge > 0:
        for rank, did in enumerate(bge_ids, start=1):
            scores[did] = scores.get(did, 0.0) + (w_bge / (rrf_k + rank))

    return [did for did, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def compute_metrics(ranked_ids: list[str], rel_map: dict[str, int]):
    top10 = ranked_ids[:10]
    dcg = sum(
        (2 ** rel_map[did] - 1) / math.log2(rank + 1)
        for rank, did in enumerate(top10, start=1)
        if rel_map.get(did, 0) > 0
    )
    ideal_rels = sorted(rel_map.values(), reverse=True)[:10]
    idcg = sum((2**rel - 1) / math.log2(rank + 1) for rank, rel in enumerate(ideal_rels, start=1))
    ndcg10 = dcg / idcg if idcg > 0 else 0.0

    mrr10 = 0.0
    for rank, did in enumerate(top10, start=1):
        if rel_map.get(did, 0) > 0:
            mrr10 = 1.0 / rank
            break

    rec10 = sum(1 for did in top10 if rel_map.get(did, 0) > 0) / len(rel_map)
    rec50 = sum(1 for did in ranked_ids[:50] if rel_map.get(did, 0) > 0) / len(rel_map)
    return ndcg10, mrr10, rec10, rec50


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    id_to_text, qrels, queries = load_eval_data()
    bm25 = BM25Retriever()
    bm25.load_index()
    dense_bge = DenseRetriever(model_key="bge")
    dense_bge.load_index()

    bm25_preds, bge_preds = {}, {}
    base_lats = []

    for qid, qtext in tqdm(queries, desc="Retrieving BM25 + BGE (648 queries)"):
        t0 = time.perf_counter()
        b_hits = bm25.search(qtext, k=100)
        d_hits = dense_bge.search(qtext, k=100)
        base_lats.append((time.perf_counter() - t0) * 1000)
        bm25_preds[qid] = [h["doc_id"] for h in b_hits]
        bge_preds[qid] = [h["doc_id"] for h in d_hits]

    alphas = [0.0, 0.3, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    rows = []

    for alpha in alphas:
        ndcgs, mrrs, rec10s, rec50s = [], [], [], []
        for qid, _ in queries:
            fused = weighted_rrf(bm25_preds[qid][:50], bge_preds[qid][:50], alpha=alpha, rrf_k=60)
            n10, m10, r10, r50 = compute_metrics(fused, qrels[qid])
            ndcgs.append(n10)
            mrrs.append(m10)
            rec10s.append(r10)
            rec50s.append(r50)

        rows.append(
            {
                "alpha_dense": alpha,
                "NDCG@10": round(float(np.mean(ndcgs)), 4),
                "MRR@10": round(float(np.mean(mrrs)), 4),
                "Recall@10": round(float(np.mean(rec10s)), 4),
                "Recall@50": round(float(np.mean(rec50s)), 4),
            }
        )

    alpha_df = pd.DataFrame(rows)
    alpha_df.to_csv(RESULTS_DIR / "weighted_rrf_ablation.csv", index=False)

    print("\n=== WEIGHTED RRF ABLATION (alpha = Dense BGE Weight, k=60) ===")
    print(alpha_df.to_string(index=False))

    # Pick best genuine hybrid alpha (0 < alpha < 1)
    hybrid_only = alpha_df[(alpha_df["alpha_dense"] > 0.0) & (alpha_df["alpha_dense"] < 1.0)]
    best_row = hybrid_only.loc[hybrid_only["NDCG@10"].idxmax()]
    best_alpha = float(best_row["alpha_dense"])
    print(f"\nBest Hybrid Alpha: {best_alpha} -> NDCG@10={best_row['NDCG@10']}, Recall@50={best_row['Recall@50']}")

    # Plot Figure 4: Weighted RRF Alpha Curve
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(alpha_df["alpha_dense"], alpha_df["NDCG@10"], marker="o", linewidth=2.2, label="NDCG@10")
    ax.plot(alpha_df["alpha_dense"], alpha_df["MRR@10"], marker="s", linewidth=2.0, label="MRR@10")
    ax.plot(alpha_df["alpha_dense"], alpha_df["Recall@50"], marker="^", linewidth=2.0, linestyle="--", label="Recall@50 (Reranker Pool)")

    ax.axvline(best_alpha, color="red", linestyle=":", alpha=0.8, label=f"Best Hybrid α = {best_alpha}")
    ax.set_xlabel("Dense Retriever Weight (α)  [0.0 = Pure BM25, 0.5 = Equal RRF, 1.0 = Pure BGE]")
    ax.set_ylabel("Score")
    ax.set_title("Fixing Asymmetric Fusion: Weighted RRF Sensitivity (BEIR FiQA, 648 Queries)", fontsize=11.5, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig4_weighted_rrf_alpha.png", dpi=300)
    plt.close()

    # Export Top-50 Weighted Hybrid Candidates for fast Colab bge-reranker-base scoring
    export_queries = []
    unique_doc_ids = set()

    for idx, (qid, qtext) in enumerate(queries):
        top50 = weighted_rrf(bm25_preds[qid][:100], bge_preds[qid][:100], alpha=best_alpha, rrf_k=60)[:50]
        unique_doc_ids.update(top50)
        export_queries.append(
            {
                "query_id": qid,
                "query_text": qtext,
                "bm25_top10": bm25_preds[qid][:10],
                "bge_top10": bge_preds[qid][:10],
                "weighted_top50": top50,
                "hybrid_base_latency_ms": round(float(base_lats[idx]), 2),
            }
        )

    doc_texts = {did: id_to_text.get(did, "")[:2000] for did in unique_doc_ids}
    payload = {
        "best_alpha": best_alpha,
        "queries": export_queries,
        "doc_texts": doc_texts,
        "qrels": qrels,
    }
    out_json = RESULTS_DIR / "weighted_candidates_top50.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    size_mb = out_json.stat().st_size / (1024 * 1024)
    print(f"Exported {out_json} ({size_mb:.1f} MB) and saved 'results/plots/fig4_weighted_rrf_alpha.png'!")


if __name__ == "__main__":
    main()