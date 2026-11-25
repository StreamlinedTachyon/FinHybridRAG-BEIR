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

    dcg = sum(
        (2 ** rel_map[did] - 1) / math.log2(rank + 1)
        for rank, did in enumerate(top_k, start=1)
        if rel_map.get(did, 0) > 0
    )
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

    prev_df = pd.read_csv(RESULTS_DIR / "final_benchmark.csv")

    with open(RESULTS_DIR / "weighted_candidates_top50.json", "r", encoding="utf-8") as f:
        cand_payload = json.load(f)

    with open(RESULTS_DIR / "bge_rerank_scores_top50.json", "r", encoding="utf-8") as f:
        bge_rr_payload = json.load(f)

    queries = cand_payload["queries"]
    qrels = cand_payload["qrels"]
    doc_texts = cand_payload["doc_texts"]
    best_alpha = cand_payload["best_alpha"]
    ce_scores = bge_rr_payload["scores"]

    base_lats = [q["hybrid_base_latency_ms"] for q in queries]
    base_mean_lat = float(np.mean(base_lats))
    base_p95_lat = float(np.percentile(base_lats, 95))

    # 1. Weighted RRF (no rerank) metrics
    wrrf_ndcgs, wrrf_mrrs, wrrf_recs = [], [], []
    for item in queries:
        qid = item["query_id"]
        n, m, r = compute_metrics_single(item["weighted_top50"], qrels[qid], k=10)
        wrrf_ndcgs.append(n)
        wrrf_mrrs.append(m)
        wrrf_recs.append(r)

    new_rows = [
        {
            "System": f"3c. Weighted RRF (BM25+BGE, α={best_alpha})",
            "Queries": len(wrrf_ndcgs),
            "NDCG@10": round(float(np.mean(wrrf_ndcgs)), 4),
            "MRR@10": round(float(np.mean(wrrf_mrrs)), 4),
            "Recall@10": round(float(np.mean(wrrf_recs)), 4),
            "Mean_Latency_ms": round(base_mean_lat, 2),
            "P95_Latency_ms": round(base_p95_lat, 2),
        }
    ]

    # 2. Weighted RRF + BGE-Reranker-Base (K=20 and K=50)
    bge_rr_preds = {}
    for pool_k in [20, 50]:
        ndcgs, mrrs, recs = [], [], []
        preds_map = {}

        for item in queries:
            qid = item["query_id"]
            rel_map = qrels[qid]
            pool_ids = item["weighted_top50"][:pool_k]
            q_scores = ce_scores[qid]

            reranked = sorted(pool_ids, key=lambda did: q_scores.get(did, -1e9), reverse=True)
            preds_map[qid] = reranked[:10]

            n, m, r = compute_metrics_single(reranked, rel_map, k=10)
            ndcgs.append(n)
            mrrs.append(m)
            recs.append(r)

        bge_rr_preds[pool_k] = preds_map
        gpu_mean = float(bge_rr_payload["mean_gpu_latency_ms"][str(pool_k)])
        gpu_p95 = float(bge_rr_payload["p95_gpu_latency_ms"][str(pool_k)])

        new_rows.append(
            {
                "System": f"5. Weighted + BGE-Rerank (K={pool_k})",
                "Queries": len(ndcgs),
                "NDCG@10": round(float(np.mean(ndcgs)), 4),
                "MRR@10": round(float(np.mean(mrrs)), 4),
                "Recall@10": round(float(np.mean(recs)), 4),
                "Mean_Latency_ms": round(base_mean_lat + gpu_mean, 2),
                "P95_Latency_ms": round(base_p95_lat + gpu_p95, 2),
            }
        )

    master_df = pd.concat([prev_df, pd.DataFrame(new_rows)], ignore_index=True)
    master_df.to_csv(RESULTS_DIR / "master_benchmark_v2.csv", index=False)

    print("\n=== MASTER BENCHMARK V2 (ALL ARCHITECTURES, FULL 648 FIQA QUERIES) ===")
    print(master_df.to_string(index=False))

    # Relative gain summary for report abstract/conclusion
    bm25_ndcg = float(master_df.loc[master_df["System"].str.startswith("1. BM25"), "NDCG@10"].iloc[0])
    best_idx = master_df["NDCG@10"].idxmax()
    best_sys = master_df.loc[best_idx, "System"]
    best_ndcg = float(master_df.loc[best_idx, "NDCG@10"])
    pct_gain = ((best_ndcg - bm25_ndcg) / bm25_ndcg) * 100
    print(f"\nWINNER: {best_sys} | NDCG@10 = {best_ndcg:.4f} (+{pct_gain:.1f}% relative over BM25)")

    # 3. Updated Master Bar Chart (Figure 1 v2)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(14, 6.5))
    x = np.arange(len(master_df))
    w = 0.26

    b1 = ax.bar(x - w, master_df["NDCG@10"], width=w, label="NDCG@10", color="#1f77b4")
    b2 = ax.bar(x, master_df["MRR@10"], width=w, label="MRR@10", color="#ff7f0e")
    b3 = ax.bar(x + w, master_df["Recall@10"], width=w, label="Recall@10", color="#2ca02c")

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
                fontsize=7,
                rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(master_df["System"], rotation=26, ha="right", fontsize=8.5)
    ax.set_ylabel("Score")
    ax.set_ylim(0, max(master_df["MRR@10"].max(), master_df["Recall@10"].max()) + 0.12)
    ax.set_title("FinHybridRAG-BEIR (648 Queries): Full Ablation & Reranker Benchmark", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig1_master_comparison_v2.png", dpi=300)
    plt.close()

    # 4. Updated Accuracy vs Latency Scatter (Figure 3 v2)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#7f7f7f" if "4." in s else ("#d62728" if "5." in s else "#1f77b4") for s in master_df["System"]]
    ax.scatter(
        master_df["Mean_Latency_ms"],
        master_df["NDCG@10"],
        s=120,
        c=colors,
        edgecolors="black",
        zorder=3,
    )

    for _, row in master_df.iterrows():
        ax.annotate(
            row["System"],
            (row["Mean_Latency_ms"], row["NDCG@10"]),
            textcoords="offset points",
            xytext=(7, 4),
            fontsize=8,
        )

    ax.set_xlabel("Mean Query Latency (ms) [CPU Retrieval + T4 GPU Rerank]")
    ax.set_ylabel("NDCG@10")
    ax.set_title("Accuracy vs. Latency Trade-off Across Retrieval & Reranking Stages", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig3_accuracy_vs_latency_v2.png", dpi=300)
    plt.close()

    # 5. Updated Qualitative Cases showing BGE-Reranker-Base impact
    pq_rows = []
    for item in queries:
        qid = item["query_id"]
        rel_map = qrels[qid]
        bm25_n, _, _ = compute_metrics_single(item["bm25_top10"], rel_map, k=10)
        bge_n, _, _ = compute_metrics_single(item["bge_top10"], rel_map, k=10)
        wrrf_n, _, _ = compute_metrics_single(item["weighted_top50"][:10], rel_map, k=10)
        rr50_n, _, _ = compute_metrics_single(bge_rr_preds[50][qid], rel_map, k=10)
        pq_rows.append(
            {
                "query_id": qid,
                "query_text": item["query_text"],
                "bm25_ndcg10": round(bm25_n, 4),
                "bge_ndcg10": round(bge_n, 4),
                "weighted_rrf_ndcg10": round(wrrf_n, 4),
                "bge_rerank_k50_ndcg10": round(rr50_n, 4),
                "rerank_gain_over_bge": round(rr50_n - bge_n, 4),
                "top1_doc": bge_rr_preds[50][qid][0],
                "top1_snippet": doc_texts.get(bge_rr_preds[50][qid][0], "")[:160].replace("\n", " "),
            }
        )

    pq_df = pd.DataFrame(pq_rows)
    top_rescues = pq_df.sort_values("rerank_gain_over_bge", ascending=False).head(3).copy()
    top_rescues["Category"] = "BGE-Reranker Beats Pure BGE Dense"
    top_rescues.to_csv(RESULTS_DIR / "bge_reranker_wins.csv", index=False)

    print("\n=== TOP 3 QUERIES WHERE BGE-RERANKER-BASE BEAT PURE BGE DENSE ===")
    for _, r in top_rescues.iterrows():
        print(f"\n(QID: {r['query_id']}) {r['query_text']}")
        print(
            f"  NDCG@10 -> BM25: {r['bm25_ndcg10']} | BGE: {r['bge_ndcg10']} | "
            f"W-RRF: {r['weighted_rrf_ndcg10']} | BGE-Rerank(K=50): {r['bge_rerank_k50_ndcg10']}"
        )

    print(f"\nSaved 'results/master_benchmark_v2.csv' and updated plots in '{PLOTS_DIR}'!")


if __name__ == "__main__":
    main()