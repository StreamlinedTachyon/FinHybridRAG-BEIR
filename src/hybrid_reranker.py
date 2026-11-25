import time
from pathlib import Path
import pandas as pd
from sentence_transformers import CrossEncoder

# Works whether run directly (`python src/hybrid_reranker.py`) or imported from root (`app.py`)
try:
    from src.bm25_retriever import BM25Retriever
    from src.dense_retriever import DenseRetriever
except ImportError:
    from bm25_retriever import BM25Retriever
    from dense_retriever import DenseRetriever


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    rrf_k: int = 60,
) -> list[tuple[str, float]]:
    """
    Pure-Python Reciprocal Rank Fusion (RRF).
    Formula: RRF(d) = sum(1 / (rrf_k + rank_m(d))) across all retrievers m.
    """
    fused_scores: dict[str, float] = {}
    for doc_ids in ranked_lists:
        for rank, doc_id in enumerate(doc_ids, start=1):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank))

    return sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)


class HybridReranker:
    def __init__(
        self,
        dense_model_key: str = "bge",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        corpus_path: str = "data/corpus.parquet",
        device: str = "cpu",
    ):
        self.dense_model_key = dense_model_key
        self.reranker_model_name = reranker_model
        self.corpus_path = Path(corpus_path)
        self.device = device

        self.bm25 = BM25Retriever(corpus_path=str(self.corpus_path))
        self.dense = DenseRetriever(model_key=dense_model_key)
        self.reranker = None
        self.id_to_text: dict[str, str] = {}

    def load_all(self, load_reranker: bool = True):
        print("Loading BM25 index...")
        self.bm25.load_index()

        print(f"Loading Dense index ({self.dense_model_key})...")
        self.dense.load_index()

        print("Loading corpus text lookup...")
        df = pd.read_parquet(self.corpus_path)
        id_col = "doc_id" if "doc_id" in df.columns else "_id"
        self.id_to_text = dict(zip(df[id_col].astype(str), df["text"].fillna("").astype(str)))

        if load_reranker:
            print(f"Loading Cross-Encoder ({self.reranker_model_name}) on {self.device}...")
            self.reranker = CrossEncoder(self.reranker_model_name, device=self.device)

    def search_bm25(self, query: str, k: int = 10) -> list[dict]:
        return self.bm25.search(query, k=k)

    def search_dense(self, query: str, k: int = 10) -> list[dict]:
        return self.dense.search(query, k=k)

    def search_hybrid(
        self,
        query: str,
        k: int = 10,
        candidates_k: int = 50,
        rrf_k: int = 60,
    ) -> list[dict]:
        bm25_hits = self.bm25.search(query, k=candidates_k)
        dense_hits = self.dense.search(query, k=candidates_k)

        bm25_ranks = {h["doc_id"]: h["rank"] for h in bm25_hits}
        dense_ranks = {h["doc_id"]: h["rank"] for h in dense_hits}

        fused = reciprocal_rank_fusion(
            [[h["doc_id"] for h in bm25_hits], [h["doc_id"] for h in dense_hits]],
            rrf_k=rrf_k,
        )[:k]

        results = []
        for rank, (doc_id, rrf_score) in enumerate(fused, start=1):
            results.append(
                {
                    "rank": rank,
                    "doc_id": doc_id,
                    "score": float(rrf_score),
                    "bm25_rank": bm25_ranks.get(doc_id),
                    "dense_rank": dense_ranks.get(doc_id),
                }
            )
        return results

    def search_hybrid_rerank(
        self,
        query: str,
        k: int = 10,
        candidates_k: int = 50,
        rrf_k: int = 60,
    ) -> list[dict]:
        if self.reranker is None:
            self.reranker = CrossEncoder(self.reranker_model_name, device=self.device)

        fused_candidates = self.search_hybrid(
            query=query,
            k=candidates_k,
            candidates_k=candidates_k,
            rrf_k=rrf_k,
        )

        pairs = [(query, self.id_to_text.get(hit["doc_id"], "")) for hit in fused_candidates]
        ce_scores = self.reranker.predict(pairs, batch_size=32)

        scored = []
        for hit, ce_score in zip(fused_candidates, ce_scores):
            scored.append(
                {
                    "doc_id": hit["doc_id"],
                    "score": float(ce_score),
                    "rrf_score": hit["score"],
                    "bm25_rank": hit["bm25_rank"],
                    "dense_rank": hit["dense_rank"],
                }
            )

        scored.sort(key=lambda x: x["score"], reverse=True)
        top_k = scored[:k]
        for rank, item in enumerate(top_k, start=1):
            item["rank"] = rank
        return top_k


if __name__ == "__main__":
    pipeline = HybridReranker(dense_model_key="bge", device="cpu")
    pipeline.load_all(load_reranker=True)

    sample_query = "Where should I park my rainy day emergency funds?"
    print(f"\nQuery: {sample_query}\n" + "=" * 75)

    # Stage 3: Hybrid RRF
    t0 = time.perf_counter()
    hybrid_hits = pipeline.search_hybrid(sample_query, k=5, candidates_k=50, rrf_k=60)
    t_hybrid = (time.perf_counter() - t0) * 1000

    print(f"\n--- Stage 3: Hybrid (BM25 + BGE via RRF) | Latency: {t_hybrid:.2f} ms ---")
    for h in hybrid_hits:
        snippet = pipeline.id_to_text.get(h["doc_id"], "")[:150].replace("\n", " ")
        print(
            f"[{h['rank']}] Doc {h['doc_id']} | RRF: {h['score']:.5f} "
            f"(BM25: #{h['bm25_rank']}, Dense: #{h['dense_rank']})"
        )
        print(f"    {snippet}...")

    # Stage 4: Hybrid + Cross-Encoder Reranker
    t0 = time.perf_counter()
    rerank_hits = pipeline.search_hybrid_rerank(sample_query, k=5, candidates_k=50, rrf_k=60)
    t_rerank = (time.perf_counter() - t0) * 1000

    print(f"\n--- Stage 4: Hybrid + Cross-Encoder Rerank | Latency: {t_rerank:.2f} ms ---")
    for h in rerank_hits:
        snippet = pipeline.id_to_text.get(h["doc_id"], "")[:150].replace("\n", " ")
        print(
            f"[{h['rank']}] Doc {h['doc_id']} | CE Score: {h['score']:.4f} "
            f"(BM25: #{h['bm25_rank']}, Dense: #{h['dense_rank']})"
        )
        print(f"    {snippet}...")