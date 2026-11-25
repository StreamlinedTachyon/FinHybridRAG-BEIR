import json
import zipfile
from pathlib import Path
import pandas as pd
from datasets import load_dataset

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

def download_and_convert():
    print("Downloading BEIR FiQA from Hugging Face...")

    # 1. Load Corpus (57,638 financial documents)
    corpus_ds = load_dataset("BeIR/fiqa", "corpus", split="corpus")
    corpus_df = corpus_ds.to_pandas().rename(columns={"_id": "doc_id"})
    corpus_df["doc_id"] = corpus_df["doc_id"].astype(str)
    corpus_df = corpus_df[["doc_id", "title", "text"]]
    corpus_df.to_parquet(DATA_DIR / "corpus.parquet", index=False)

    # 2. Load Queries
    queries_ds = load_dataset("BeIR/fiqa", "queries", split="queries")
    queries_df = queries_ds.to_pandas().rename(columns={"_id": "query_id"})
    queries_df["query_id"] = queries_df["query_id"].astype(str)
    queries_df = queries_df[["query_id", "text"]]
    queries_df.to_parquet(DATA_DIR / "queries.parquet", index=False)

    # 3. Load Ground-Truth Test Qrels
    qrels_ds = load_dataset("BeIR/fiqa-qrels", split="test")
    qrels_df = qrels_ds.to_pandas().rename(
        columns={"query-id": "query_id", "corpus-id": "doc_id"}
    )
    qrels_df["query_id"] = qrels_df["query_id"].astype(str)
    qrels_df["doc_id"] = qrels_df["doc_id"].astype(str)
    qrels_df = qrels_df[qrels_df["score"] > 0].reset_index(drop=True)
    qrels_df.to_parquet(DATA_DIR / "qrels.parquet", index=False)

    # 4. Filter queries that have ground-truth test labels (648 queries)
    valid_qids = set(qrels_df["query_id"])
    test_queries_df = queries_df[queries_df["query_id"].isin(valid_qids)].reset_index(drop=True)
    test_queries_df.to_parquet(DATA_DIR / "test_queries.parquet", index=False)

    # Print quick EDA summary (save these numbers for your report!)
    avg_words = corpus_df["text"].str.split().str.len().mean()
    print("\n--- BEIR FiQA Dataset Ready ---")
    print(f"Total Corpus Documents : {len(corpus_df):,}")
    print(f"Average Doc Length     : {avg_words:.1f} words")
    print(f"Total Test Queries     : {len(test_queries_df):,}")
    print(f"Total Qrel Pairs       : {len(qrels_df):,}")
    print(f"\nSample Query: {test_queries_df.iloc[0]['text']}")

if __name__ == "__main__":
    download_and_convert()