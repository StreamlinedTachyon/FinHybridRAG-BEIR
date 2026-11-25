import json
import math
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"


def compute_metrics_single(ranked_doc_ids: list[str], rel_map: dict[str, int], k: int = 10):
    top_k = ranked_doc_ids[:k]

    dcg = 0.0
    for rank, did in enumerate(top_k, start=1):
        rel = rel_map.get(did, 0)
        if rel > 0:
            dcg += (2**rel - 1) / math.log2(rank + 1)

    ideal_rels = sorted(rel_map.values(), reverse=True)[:k]
    idcg = sum((2**rel - 1) / math.log2(rank + 1) for rank, rel in enumerate(ideal_rels, start=1))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    mrr = 0.0
    for rank, did in enumerate(top_k, start=1):
        if rel_map.get(did, 0) > 0:
            mrr = 1.0 / rank
            break

    hits = sum(1 for did in top_k if rel_map.get(did, 0) > 0)
    recall = hits / len(rel_map) if rel_map else 0.0

    return ndcg, mrr, recall


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Stage 1-3 CSVs and Candidate/Score JSONs
    stage13_df = pd.read_csv(RESULTS_DIR / "stage1_to_3_metrics.csv")
    rrf_df = pd.read_csv(RESULTS_DIR / "rrf_k_sensitivity.csv")

    with open(RESULTS_DIR / "rerank_candidates_top100.json", "r", encoding="utf-8") as f:
        cand_payload = json.load(f)

    with open(RESULTS_DIR / "rerank_scores_top100.json", "r", encoding="utf-8") as f:
        score_payload = json.load(f)

    queries = cand_payload["queries"]
    qrels = cand_payload["qrels"]
    doc_texts = cand_payload["doc_texts"]
    ce_scores = score_payload["scores"]

    base_lats = [q["hybrid_base_latency_ms"] for q in queries]
    base_mean_lat = float(np.mean(base_lats))
    base_p95_lat = float(np.percentile(base_lats, 95))

    # 2. Evaluate Stage 4 (Hybrid + Cross-Encoder Rerank) for K = 20, 50, 100
    stage4_rows = []
    rerank_preds_by_k = {}

    for pool_k in [20, 50, 100]:
        ndcgs, mrrs, recalls = [], [], []
        preds_map = {}

        for item in queries:
            qid = item["query_id"]
            rel_map = qrels[qid]
            pool_ids = item["hybrid_top100"][:pool_k]
            q_scores = ce_scores[qid]

            reranked = sorted(pool_ids, key=lambda did: q_scores.get(did, -1e9), reverse=True)
            preds_map[qid] = reranked[:10]

            ndcg, mrr, recall = compute_metrics_single(reranked, rel_map, k=10)
            ndcgs.append(ndcg)
            mrrs.append(mrr)
            recalls.append(recall)

        rerank_preds_by_k[pool_k] = preds_map
        gpu_mean = float(score_payload["mean_gpu_latency_ms"][str(pool_k)])
        gpu_p95 = float(score_payload["p95_gpu_latency_ms"][str(pool_k)])

        stage4_rows.append(
            {
                "System": f"4. Hybrid + Rerank (K={pool_k})",
                "Queries": len(ndcgs),
                "NDCG@10": round(float(np.mean(ndcgs)), 4),
                "MRR@10": round(float(np.mean(mrrs)), 4),
                "Recall@10": round(float(np.mean(recalls)), 4),
                "Mean_Latency_ms": round(base_mean_lat + gpu_mean, 2),
                "P95_Latency_ms": round(base_p95_lat + gpu_p95, 2),
            }
        )

    final_df = pd.concat([stage13_df, pd.DataFrame(stage4_rows)], ignore_index=True)
    final_df.to_csv(RESULTS_DIR / "final_benchmark.csv", index=False)

    print("\n=== FINAL MASTER BENCHMARK (FULL 648 FIQA QUERIES) ===")
    print(final_df.to_string(index=False))

    # 3. Extract Qualitative Case Studies (Using K=50 Reranker)
    per_query_rows = []
    for item in queries:
        qid = item["query_id"]
        rel_map = qrels[qid]

        bm25_ndcg, _, _ = compute_metrics_single(item["bm25_top10"], rel_map, k=10)
        bge_ndcg, _, _ = compute_metrics_single(item["bge_top10"], rel_map, k=10)
        hyb_ndcg, _, _ = compute_metrics_single(item["hybrid_top100"][:10], rel_map, k=10)
        rr_ndcg, _, _ = compute_metrics_single(rerank_preds_by_k[50][qid], rel_map, k=10)

        per_query_rows.append(
            {
                "query_id": qid,
                "query_text": item["query_text"],
                "num_relevant_docs": len(rel_map),
                "bm25_ndcg10": round(bm25_ndcg, 4),
                "bge_ndcg10": round(bge_ndcg, 4),
                "hybrid_rrf_ndcg10": round(hyb_ndcg, 4),
                "rerank_k50_ndcg10": round(rr_ndcg, 4),
                "bm25_minus_bge": round(bm25_ndcg - bge_ndcg, 4),
                "rerank_gain_over_rrf": round(rr_ndcg - hyb_ndcg, 4),
                "top1_rerank_doc": rerank_preds_by_k[50][qid][0],
                "top1_rerank_snippet": doc_texts.get(rerank_preds_by_k[50][qid][0], "")[:160].replace("\n", " "),
            }
        )

    pq_df = pd.DataFrame(per_query_rows)
    case_frames = []

    bm25_wins = pq_df.sort_values("bm25_minus_bge", ascending=False).head(2).copy()
    bm25_wins["Category"] = "BM25 Beats Dense (Exact Lexical Match)"
    case_frames.append(bm25_wins)

    bge_wins = pq_df.sort_values("bm25_minus_bge", ascending=True).head(2).copy()
    bge_wins["Category"] = "Dense BGE Beats BM25 (Semantic Paraphrase)"
    case_frames.append(bge_wins)

    rerank_rescues = pq_df.sort_values("rerank_gain_over_rrf", ascending=False).head(2).copy()
    rerank_rescues["Category"] = "Cross-Encoder Rescue (Rerank Beats RRF)"
    case_frames.append(rerank_rescues)

    failures = pq_df[
        (pq_df["rerank_k50_ndcg10"] == 0.0) & (pq_df["bm25_ndcg10"] == 0.0) & (pq_df["bge_ndcg10"] == 0.0)
    ].head(2).copy()
    failures["Category"] = "Hard Failure Case (All Retrievers Miss)"
    case_frames.append(failures)

    cases_df = pd.concat(case_frames, ignore_index=True)
    cols_order = [
        "Category",
        "query_id",
        "query_text",
        "bm25_ndcg10",
        "bge_ndcg10",
        "hybrid_rrf_ndcg10",
        "rerank_k50_ndcg10",
        "top1_rerank_doc",
        "top1_rerank_snippet",
    ]
    cases_df = cases_df[cols_order]
    cases_df.to_csv(RESULTS_DIR / "qualitative_case_studies.csv", index=False)

    print("\n=== QUALITATIVE CASE STUDIES (8 REAL EXAMPLES FOR REPORT) ===")
    for _, r in cases_df.iterrows():
        print(f"\n[{r['Category']}] (QID: {r['query_id']})")
        print(f"  Query : {r['query_text']}")
        print(
            f"  NDCG@10 -> BM25: {r['bm25_ndcg10']} | BGE: {r['bge_ndcg10']} | "
            f"RRF: {r['hybrid_rrf_ndcg10']} | Rerank(K=50): {r['rerank_k50_ndcg10']}"
        )

    # ---------------- PLOTS (300 DPI for Report) ----------------
    plt.style.use("seaborn-v0_8-whitegrid")

    # Figure 1: Pipeline Comparison (NDCG@10, MRR@10, Recall@10)
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(final_df))
    w = 0.26

    b1 = ax.bar(x - w, final_df["NDCG@10"], width=w, label="NDCG@10", color="#1f77b4")
    b2 = ax.bar(x, final_df["MRR@10"], width=w, label="MRR@10", color="#ff7f0e")
    b3 = ax.bar(x + w, final_df["Recall@10"], width=w, label="Recall@10", color="#2ca02c")

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
                rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(final_df["System"], rotation=22, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim(0, max(final_df["MRR@10"].max(), final_df["Recall@10"].max()) + 0.12)
    ax.set_title("BEIR FiQA (648 Queries): Retrieval Pipeline Benchmark", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig1_pipeline_comparison.png", dpi=300)
    plt.close()

    # Figure 2: RRF Smoothing Constant k Sensitivity
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(rrf_df["rrf_k"], rrf_df["NDCG@10"], marker="o", linewidth=2, label="NDCG@10")
    ax.plot(rrf_df["rrf_k"], rrf_df["MRR@10"], marker="s", linewidth=2, label="MRR@10")
    ax.plot(rrf_df["rrf_k"], rrf_df["Recall@10"], marker="^", linewidth=2, label="Recall@10")

    for _, row in rrf_df.iterrows():
        ax.annotate(
            f"{row['NDCG@10']:.4f}",
            (row["rrf_k"], row["NDCG@10"]),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=8,
        )

    ax.set_xlabel("RRF Smoothing Constant (k)")
    ax.set_ylabel("Score")
    ax.set_title("Effect of RRF Smoothing Parameter (k) on Hybrid Retrieval (BM25 + BGE)", fontsize=12, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig2_rrf_k_sensitivity.png", dpi=300)
    plt.close()

    # Figure 3: Accuracy (NDCG@10) vs Mean Latency Trade-Off
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(
        final_df["Mean_Latency_ms"],
        final_df["NDCG@10"],
        s=110,
        c=range(len(final_df)),
        cmap="viridis",
        edgecolors="black",
        zorder=3,
    )

    for _, row in final_df.iterrows():
        ax.annotate(
            row["System"],
            (row["Mean_Latency_ms"], row["NDCG@10"]),
            textcoords="offset points",
            xytext=(7, 5),
            fontsize=8.5,
        )

    ax.set_xlabel("Mean Query Latency (ms) [CPU Retrieval + T4 GPU Rerank]")
    ax.set_ylabel("NDCG@10")
    ax.set_title("Retrieval Accuracy vs. Query Latency Trade-off (BEIR FiQA)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig3_accuracy_vs_latency.png", dpi=300)
    plt.close()

    print(f"\nSaved 3 high-resolution plots to '{PLOTS_DIR}' and CSVs to '{RESULTS_DIR}'!")


if __name__ == "__main__":
    main()