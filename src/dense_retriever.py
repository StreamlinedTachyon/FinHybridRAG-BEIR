import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

MODEL_CONFIG = {
    "minilm": {
        "hf_name": "sentence-transformers/all-MiniLM-L6-v2",
        "emb_file": "embeddings_minilm.npy",
        "query_prefix": "",
    },
    "bge": {
        "hf_name": "BAAI/bge-small-en-v1.5",
        "emb_file": "embeddings_bge.npy",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
    },
}


class DenseRetriever:
    def __init__(self, model_key: str = "bge", index_dir: str = "indexes/dense"):
        if model_key not in MODEL_CONFIG:
            raise ValueError(f"model_key must be one of {list(MODEL_CONFIG.keys())}")
        self.model_key = model_key
        self.cfg = MODEL_CONFIG[model_key]
        self.index_dir = Path(index_dir)
        self.embeddings = None
        self.doc_ids = []
        self.model = None
        self.backend = None

    def load_index(self):
        emb_path = self.index_dir / self.cfg["emb_file"]
        ids_path = self.index_dir / "dense_doc_ids.json"
        if not emb_path.exists() or not ids_path.exists():
            raise FileNotFoundError(f"Missing files in {self.index_dir}.")

        # Keep matrix in float16 (~44 MB) instead of float32 (~88 MB) for low-RAM cloud deployment
        self.embeddings = np.load(emb_path)
        with open(ids_path, "r", encoding="utf-8") as f:
            self.doc_ids = json.load(f)

        # Prefer lightweight ONNX (fastembed, ~180 MB RAM); fallback to SentenceTransformer if absent
        try:
            from fastembed import TextEmbedding

            self.model = TextEmbedding(model_name=self.cfg["hf_name"])
            self.backend = "fastembed"
        except ImportError:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.cfg["hf_name"], device="cpu")
            self.backend = "sentence_transformers"

    def _encode_query(self, text: str) -> np.ndarray:
        if self.backend == "fastembed":
            vec = next(self.model.embed([text])).astype(np.float32)
            norm = float(np.linalg.norm(vec))
            return vec / norm if norm > 0 else vec

        return self.model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

    def search(self, query: str, k: int = 10):
        if self.embeddings is None or self.model is None:
            self.load_index()

        prefixed_query = self.cfg["query_prefix"] + query
        q_vec = self._encode_query(prefixed_query)

        # Memory-efficient chunked float16 -> float32 exact dot product
        n_docs = self.embeddings.shape[0]
        scores = np.empty(n_docs, dtype=np.float32)
        chunk_size = 16384
        for start in range(0, n_docs, chunk_size):
            end = min(start + chunk_size, n_docs)
            scores[start:end] = self.embeddings[start:end].astype(np.float32) @ q_vec

        # O(N) top-k selection using argpartition, then sort only top-k
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        hits = []
        for rank, idx in enumerate(top_idx, start=1):
            hits.append(
                {
                    "rank": rank,
                    "doc_id": self.doc_ids[int(idx)],
                    "score": float(scores[idx]),
                }
            )
        return hits


if __name__ == "__main__":
    corpus_df = pd.read_parquet("data/corpus.parquet")
    id_col = "doc_id" if "doc_id" in corpus_df.columns else "_id"
    id_to_text = dict(zip(corpus_df[id_col].astype(str), corpus_df["text"].fillna("")))

    sample_query = "Where should I park my rainy day emergency funds?"
    retriever = DenseRetriever(model_key="bge")
    retriever.load_index()

    t0 = time.perf_counter()
    top_hits = retriever.search(sample_query, k=5)
    latency_ms = (time.perf_counter() - t0) * 1000

    print(f"\n=== Dense Model: BGE ({retriever.backend}) | Latency: {latency_ms:.2f} ms ===")
    for hit in top_hits:
        snippet = id_to_text.get(hit["doc_id"], "")[:150].replace("\n", " ")
        print(f"[{hit['rank']}] Doc {hit['doc_id']} | Cosine: {hit['score']:.4f}")
        print(f"    {snippet}...")