import json
import time
from pathlib import Path
import bm25s
import pandas as pd


class BM25Retriever:
    def __init__(
        self,
        corpus_path: str = "data/corpus.parquet",
        index_dir: str = "indexes/bm25_index",
    ):
        self.corpus_path = Path(corpus_path)
        self.index_dir = Path(index_dir)
        self.doc_ids_path = self.index_dir / "doc_ids.json"
        self.retriever = None
        self.doc_ids = []

    def build_index(self):
        print(f"Loading corpus from {self.corpus_path}...")
        df = pd.read_parquet(self.corpus_path)

        # Auto-detect ID column name (handles both 'doc_id' and '_id')
        id_col = "doc_id" if "doc_id" in df.columns else "_id"
        self.doc_ids = df[id_col].astype(str).tolist()
        texts = df["text"].fillna("").astype(str).tolist()

        print(f"Tokenizing {len(texts):,} documents...")
        t0 = time.perf_counter()
        corpus_tokens = bm25s.tokenize(texts, stopwords="en")

        print("Building Lucene BM25 index...")
        self.retriever = bm25s.BM25(method="lucene")
        self.retriever.index(corpus_tokens)

        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.retriever.save(str(self.index_dir))
        with open(self.doc_ids_path, "w", encoding="utf-8") as f:
            json.dump(self.doc_ids, f)

        elapsed = time.perf_counter() - t0
        print(f"BM25 index saved to '{self.index_dir}' in {elapsed:.2f}s!")

    def load_index(self):
        if not self.index_dir.exists() or not self.doc_ids_path.exists():
            raise FileNotFoundError("BM25 index not found. Run build_index() first.")

        self.retriever = bm25s.BM25.load(str(self.index_dir), load_corpus=False)
        with open(self.doc_ids_path, "r", encoding="utf-8") as f:
            self.doc_ids = json.load(f)

    def search(self, query: str, k: int = 10):
        if self.retriever is None:
            self.load_index()

        query_tokens = bm25s.tokenize([query], stopwords="en")
        results, scores = self.retriever.retrieve(query_tokens, k=k)

        hits = []
        for rank, (idx, score) in enumerate(zip(results[0], scores[0]), start=1):
            hits.append(
                {
                    "rank": rank,
                    "doc_id": self.doc_ids[int(idx)],
                    "score": float(score),
                }
            )
        return hits


if __name__ == "__main__":
    bm25 = BM25Retriever()
    bm25.build_index()

    # Quick sanity check on a real financial query
    corpus_df = pd.read_parquet("data/corpus.parquet")
    id_col = "doc_id" if "doc_id" in corpus_df.columns else "_id"
    id_to_text = dict(zip(corpus_df[id_col].astype(str), corpus_df["text"].fillna("")))

    sample_query = "Where should I park my rainy day emergency funds?"
    t0 = time.perf_counter()
    top_hits = bm25.search(sample_query, k=5)
    latency_ms = (time.perf_counter() - t0) * 1000

    print(f"\nQuery: {sample_query}")
    print(f"BM25 Search Latency: {latency_ms:.2f} ms\n")
    for hit in top_hits:
        snippet = id_to_text.get(hit["doc_id"], "")[:180].replace("\n", " ")
        print(f"[{hit['rank']}] Doc {hit['doc_id']} | Score: {hit['score']:.3f}")
        print(f"    {snippet}...\n")