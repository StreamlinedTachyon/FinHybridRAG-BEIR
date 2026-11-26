import html
import os
import re
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from src.bm25_retriever import BM25Retriever
from src.dense_retriever import DenseRetriever
from src.rag_generator import generate_grounded_answer

load_dotenv()

st.set_page_config(
    page_title="FinHybridRAG • BEIR FiQA Engine",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

RESULTS_DIR = Path("results")

SAMPLE_QUERIES = [
    "Where should I park my rainy day emergency funds?",
    "Regarding Australian CBS takeover of TEN",
    "US citizen transferring money to Indian fiance to buy property",
    "For insurance, why should you refuse $4,000/year for only 10 years and prefer $500/year indefinitely?",
    "Who won the 2026 FIFA World Cup?",
]

# ---------------- CUSTOM CSS (SHARP FINTECH / DOCS THEME) ----------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

code, pre, .mono {
    font-family: 'JetBrains Mono', monospace !important;
}

#MainMenu, footer, .stDeployButton {
    visibility: hidden;
}

.block-container {
    padding-top: 1.8rem;
    padding-bottom: 3rem;
    max-width: 1320px;
}

/* Header Banner */
.hero-wrap {
    border-bottom: 1px solid #1E293B;
    padding-bottom: 1.1rem;
    margin-bottom: 1.5rem;
}
.hero-kicker {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #10B981;
    margin-bottom: 0.35rem;
}
.hero-title {
    font-size: 1.85rem;
    font-weight: 700;
    color: #F8FAFC;
    letter-spacing: -0.025em;
    margin: 0;
}
.hero-sub {
    color: #94A3B8;
    font-size: 0.92rem;
    margin-top: 0.35rem;
}

/* Metric Cards */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.75rem;
    margin: 1rem 0 1.25rem 0;
}
.kpi-card {
    background: #111827;
    border: 1px solid #1F2937;
    border-radius: 8px;
    padding: 0.85rem 1rem;
    min-width: 0;
}
.kpi-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94A3B8;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.kpi-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.08rem;
    font-weight: 600;
    color: #F8FAFC;
    margin-top: 0.3rem;
    white-space: nowrap;
}
.kpi-sub {
    font-size: 0.73rem;
    color: #64748B;
    margin-top: 0.2rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* Answer Surface (Perplexity / Docs Card) */
.answer-card {
    background: linear-gradient(180deg, rgba(17, 24, 39, 0.92) 0%, rgba(15, 23, 42, 0.78) 100%);
    border: 1px solid #1E293B;
    border-radius: 10px;
    padding: 1.35rem 1.5rem;
    margin-bottom: 1.4rem;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.28);
}
.answer-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.5rem;
    padding-bottom: 0.85rem;
    margin-bottom: 0.95rem;
    border-bottom: 1px solid #1E293B;
}
.answer-title {
    font-size: 0.82rem;
    font-family: 'JetBrains Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #CBD5E1;
    font-weight: 600;
}
.answer-body {
    color: #E2E8F0;
    font-size: 0.97rem;
    line-height: 1.7;
}
.answer-body p {
    margin: 0 0 0.75rem 0;
}
.answer-body p:last-child {
    margin-bottom: 0;
}
.answer-body ul {
    margin: 0.4rem 0 0.75rem 1.25rem;
    padding: 0;
}
.answer-body li {
    margin-bottom: 0.4rem;
}

/* Audit Status Badges */
.badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 0.24rem 0.62rem;
    border-radius: 999px;
    letter-spacing: 0.04em;
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
}
.badge-verified {
    background: rgba(16, 185, 129, 0.14);
    color: #34D399;
    border: 1px solid rgba(16, 185, 129, 0.38);
}
.badge-abstained {
    background: rgba(245, 158, 11, 0.14);
    color: #FBBF24;
    border: 1px solid rgba(245, 158, 11, 0.38);
}
.badge-danger {
    background: rgba(239, 68, 68, 0.14);
    color: #F87171;
    border: 1px solid rgba(239, 68, 68, 0.38);
}
.badge-neutral {
    background: rgba(148, 163, 184, 0.12);
    color: #CBD5E1;
    border: 1px solid rgba(148, 163, 184, 0.3);
}

/* Inline Citation Pills */
.cite-pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.74rem;
    font-weight: 500;
    padding: 0.1rem 0.45rem;
    border-radius: 5px;
    margin: 0 0.12rem;
    background: rgba(56, 189, 248, 0.14);
    color: #7DD3FC;
    border: 1px solid rgba(56, 189, 248, 0.34);
    white-space: nowrap;
}
.cite-pill-invalid {
    background: rgba(239, 68, 68, 0.18);
    color: #FCA5A5;
    border: 1px solid rgba(239, 68, 68, 0.45);
}

/* Evidence & Workbench Cards */
.doc-card {
    background: #0F172A;
    border: 1px solid #1E293B;
    border-radius: 8px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.65rem;
}
.doc-card-cited {
    border-left: 3px solid #10B981;
}
.doc-meta {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-bottom: 0.45rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.74rem;
    color: #94A3B8;
}
.doc-id-tag {
    color: #F8FAFC;
    font-weight: 600;
}
.rank-chip {
    background: #1E293B;
    color: #CBD5E1;
    padding: 0.12rem 0.42rem;
    border-radius: 4px;
    font-size: 0.7rem;
}
.rank-chip-hit {
    background: rgba(16, 185, 129, 0.15);
    color: #6EE7B7;
    border: 1px solid rgba(16, 185, 129, 0.3);
}
.doc-snippet {
    color: #CBD5E1;
    font-size: 0.85rem;
    line-height: 1.55;
}

/* Sidebar Spec Box */
.spec-box {
    background: #0B0F19;
    border: 1px solid #1E293B;
    border-radius: 8px;
    padding: 0.85rem;
    font-size: 0.8rem;
    line-height: 1.65;
    color: #94A3B8;
}
.spec-box b {
    color: #E2E8F0;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.4rem;
    border-bottom: 1px solid #1E293B;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif;
    font-size: 0.88rem;
    font-weight: 500;
    padding: 0.55rem 1rem;
    color: #94A3B8;
}
.stTabs [aria-selected="true"] {
    color: #F8FAFC !important;
    border-bottom-color: #10B981 !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------- HELPERS ----------------
def clean_fiqa_text(raw: str, max_chars: int | None = None, single_line: bool = False) -> str:
    """Cleans HTML entities and doubled CSV quotes in FiQA corpus text safely."""
    txt = html.unescape(str(raw)).replace('""', '"').strip()
    if single_line:
        txt = re.sub(r"\s+", " ", txt)
    if max_chars is not None and len(txt) > max_chars:
        txt = txt[:max_chars].rstrip() + "..."
    return html.escape(txt)


def stage_sort_key(name: str) -> tuple:
    """Ensures logical stage order: 1 -> 2a/2b -> 3a/3b/3c -> 4 (K=20,50,100) -> 5 (K=20,50)."""
    prefix = name.split(".")[0].strip()
    order_map = {"1": 10, "2a": 20, "2b": 21, "3a": 30, "3b": 31, "3c": 32, "4": 40, "5": 50}
    k_match = re.search(r"K=(\d+)", name)
    k_val = int(k_match.group(1)) if k_match else 0
    return (order_map.get(prefix, 99), k_val, name)


def short_system_label(name: str) -> str:
    if name.startswith("1."):
        return "BM25"
    if name.startswith("2a."):
        return "MiniLM"
    if name.startswith("2b."):
        return "BGE"
    if name.startswith("3a."):
        return "RRF-MiniLM"
    if name.startswith("3b."):
        return "RRF-BGE"
    if name.startswith("3c."):
        return "W-RRF ★"
    if name.startswith("4."):
        m = re.search(r"K=(\d+)", name)
        return f"CE-K{m.group(1)}" if m else "CE"
    if name.startswith("5."):
        m = re.search(r"K=(\d+)", name)
        return f"BGE-RR{m.group(1)}" if m else "BGE-RR"
    return name[:10]


# ---------------- DATA & ENGINE LOADING ----------------
def ensure_artifacts():
    """Auto-downloads BEIR FiQA runtime artifacts from HF Hub when running in the cloud."""
    required = [
        Path("data/corpus.parquet"),
        Path("indexes/bm25_index/doc_ids.json"),
        Path("indexes/dense/embeddings_bge.npy"),
        Path("indexes/dense/dense_doc_ids.json"),
    ]
    if all(p.exists() for p in required):
        return

    dataset_repo = os.getenv("HF_DATASET_REPO", "theonlyeagle/finhybridrag-beir-artifacts").strip()
    if not dataset_repo:
        raise FileNotFoundError(
            "Local index files not found and HF_DATASET_REPO environment variable is not set."
        )

    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=dataset_repo,
        repo_type="dataset",
        local_dir=".",
    )


@st.cache_resource(show_spinner="Initializing BM25 + BGE-small indexes (57,638 documents)...")
def load_retrieval_engine():
    ensure_artifacts()
    bm25 = BM25Retriever()
    bm25.load_index()

    dense_bge = DenseRetriever(model_key="bge")
    dense_bge.load_index()

    corpus_df = pd.read_parquet("data/corpus.parquet")
    id_col = "doc_id" if "doc_id" in corpus_df.columns else "_id"
    id_to_text = dict(zip(corpus_df[id_col].astype(str), corpus_df["text"].fillna("").astype(str)))
    return bm25, dense_bge, id_to_text



@st.cache_resource(show_spinner="Loading optional Cross-Encoder (ms-marco-MiniLM-L-6-v2)...")
def load_cross_encoder():
    from sentence_transformers import CrossEncoder
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")


def run_weighted_rrf(bm25_hits, dense_hits, alpha: float = 0.95, rrf_k: int = 60, top_k: int = 5):
    b_ranks = {h["doc_id"]: h["rank"] for h in bm25_hits}
    d_ranks = {h["doc_id"]: h["rank"] for h in dense_hits}

    scores = {}
    w_bm25 = 1.0 - alpha
    w_dense = alpha

    if w_bm25 > 0:
        for did, r in b_ranks.items():
            scores[did] = scores.get(did, 0.0) + (w_bm25 / (rrf_k + r))
    if w_dense > 0:
        for did, r in d_ranks.items():
            scores[did] = scores.get(did, 0.0) + (w_dense / (rrf_k + r))

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        {
            "rank": i,
            "doc_id": did,
            "score": float(sc),
            "bm25_rank": b_ranks.get(did),
            "dense_rank": d_ranks.get(did),
        }
        for i, (did, sc) in enumerate(ranked, start=1)
    ]


def format_answer_html(raw_text: str, valid_doc_ids: set[str]) -> str:
    safe = html.escape(raw_text)
    safe = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe)

    def _pill(match):
        did = match.group(1)
        cls = "cite-pill" if did in valid_doc_ids else "cite-pill cite-pill-invalid"
        return f'<span class="{cls}">Doc {did}</span>'

    safe = re.sub(r"[\[\u3010\uff3b]\s*Doc\s*#?(\d+)\s*[\]\u3011\uff3d]", _pill, safe, flags=re.IGNORECASE)
    lines = safe.splitlines()
    out_parts = []
    in_list = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if in_list:
                out_parts.append("</ul>")
                in_list = False
            continue
        if stripped.startswith(("- ", "* ", "• ")):
            if not in_list:
                out_parts.append("<ul>")
                in_list = True
            out_parts.append(f"<li>{stripped[2:].strip()}</li>")
        else:
            if in_list:
                out_parts.append("</ul>")
                in_list = False
            out_parts.append(f"<p>{stripped}</p>")
    if in_list:
        out_parts.append("</ul>")

    return "".join(out_parts)


def audit_badge_html(audit: dict | None) -> str:
    if not audit:
        return '<span class="badge badge-neutral">NO AUDIT</span>'
    status = audit.get("status", "")
    label = html.escape(audit.get("label", ""))
    cls_map = {
        "VERIFIED": "badge-verified",
        "ABSTAINED": "badge-abstained",
        "HALLUCINATION": "badge-danger",
        "UNVERIFIED": "badge-danger",
        "RETRIEVAL_ONLY": "badge-neutral",
    }
    cls = cls_map.get(status, "badge-neutral")
    return f'<span class="badge {cls}">{label}</span>'


def plotly_layout_defaults(title: str, height: int = 410, top_margin: int = 72) -> dict:
    return dict(
        title=dict(text=title, x=0.01, y=0.96, font=dict(size=14, color="#F8FAFC", family="Inter")),
        template="plotly_dark",
        paper_bgcolor="rgba(17, 24, 39, 0.72)",
        plot_bgcolor="rgba(11, 15, 25, 0.65)",
        font=dict(family="Inter", color="#CBD5E1", size=12),
        height=height,
        margin=dict(l=52, r=24, t=top_margin, b=54),
        hoverlabel=dict(bgcolor="#0F172A", bordercolor="#334155", font_color="#F8FAFC"),
    )


# ---------------- SIDEBAR (NO EXPOSED API KEY) ----------------
st.sidebar.markdown("### FinHybridRAG-BEIR")
st.sidebar.caption("First-Principles Hybrid Retrieval & Grounded Financial QA")

api_key_input = ""
try:
    api_key_input = st.secrets.get("GROQ_API_KEY", "")
except Exception:
    api_key_input = ""
if not api_key_input:
    api_key_input = os.getenv("GROQ_API_KEY", "")

llm_status_html = (
    '<span class="badge badge-verified">● LLM ENGINE CONNECTED</span>'
    if api_key_input
    else '<span class="badge badge-neutral">● RETRIEVAL-ONLY MODE</span>'
)
st.sidebar.markdown(llm_status_html, unsafe_allow_html=True)

st.sidebar.markdown("#### Retrieval Parameters")
alpha_val = st.sidebar.slider(
    "Weighted RRF α (Dense Weight)",
    min_value=0.0,
    max_value=1.0,
    value=0.95,
    step=0.05,
    help="0.00 = Pure BM25 | 0.50 = Equal RRF | 0.95 = FiQA Optimal | 1.00 = Pure BGE Dense",
)
top_k_docs = st.sidebar.slider("Context Window (Top-K Docs)", min_value=3, max_value=10, value=5)
llm_model = st.sidebar.selectbox(
    "Synthesis Model (Groq)",
    ["openai/gpt-oss-120b", "openai/gpt-oss-20b"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div class="spec-box">
        <b>Corpus:</b> BEIR FiQA (57,638 docs)<br>
        <b>Eval Set:</b> 648 Queries • 1,706 Qrels<br>
        <b>Sparse:</b> Lucene BM25 (<code>bm25s</code>)<br>
        <b>Dense:</b> <code>bge-small-en-v1.5</code> (float16)<br>
        <b>Best NDCG@10:</b> <code>0.4070</code> (+75.0% vs BM25)
    </div>
    """,
    unsafe_allow_html=True,
)

bm25, dense_bge, id_to_text = load_retrieval_engine()

# ---------------- HEADER ----------------
st.markdown(
    """
    <div class="hero-wrap">
        <div class="hero-kicker">BEIR FiQA • First-Principles Information Retrieval & Grounded RAG</div>
        <h1 class="hero-title">FinHybridRAG</h1>
        <div class="hero-sub">
            Zero-black-box hybrid search combining Lucene BM25, L2-normalized NumPy dense dot-product retrieval,
            Weighted Reciprocal Rank Fusion (α = 0.95), and automated inline citation verification.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(
    [
        "Financial Q&A",
        "Retriever Workbench",
        "BEIR FiQA Benchmarks (648 Queries)",
    ]
)

# ================= TAB 1: GROUNDED QA =================
with tab1:
    q_col1, q_col2 = st.columns([1, 2], vertical_alignment="bottom")
    with q_col1:
        preset = st.selectbox(
            "Preset Financial Queries",
            ["Custom query..."] + SAMPLE_QUERIES,
            label_visibility="visible",
        )
    default_q = "" if preset == "Custom query..." else preset
    with q_col2:
        user_query = st.text_input(
            "Ask a financial question",
            value=default_q,
            placeholder="e.g., How does a Roth IRA differ from a traditional IRA?",
        )

    run_btn = st.button("Run Hybrid Retrieval & Synthesize", type="primary")

    if run_btn and user_query.strip():
        t0 = time.perf_counter()
        b_hits = bm25.search(user_query, k=50)
        t_bm25 = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        d_hits = dense_bge.search(user_query, k=50)
        t_dense = (time.perf_counter() - t1) * 1000

        fused = run_weighted_rrf(b_hits, d_hits, alpha=alpha_val, rrf_k=60, top_k=top_k_docs)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        retrieved_docs = [{**h, "text": id_to_text.get(h["doc_id"], "")} for h in fused]
        retrieved_id_set = {str(d["doc_id"]) for d in retrieved_docs}

        with st.spinner("Synthesizing citation-grounded answer..."):
            gen = generate_grounded_answer(
                query=user_query,
                retrieved_docs=retrieved_docs,
                api_key=api_key_input,
                model=llm_model,
            )

        audit = gen.get("audit") or {}
        cited_count = len(audit.get("valid_ids", []))
        total_cited = len(audit.get("cited_ids", []))
        cite_metric = (
            "Abstained"
            if audit.get("status") == "ABSTAINED"
            else f"{cited_count}/{total_cited}" if total_cited else "0"
        )

        st.markdown(
            f"""
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-label">Retrieval Latency</div>
                    <div class="kpi-value">{retrieval_ms:.1f} ms</div>
                    <div class="kpi-sub">BM25 {t_bm25:.1f}ms • BGE {t_dense:.1f}ms</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">LLM Synthesis</div>
                    <div class="kpi-value">{gen['latency_ms']:.0f} ms</div>
                    <div class="kpi-sub">{html.escape(gen['model'])}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Fusion Weight (<span style="text-transform:none;">α</span>)</div>
                    <div class="kpi-value">{alpha_val:.2f}</div>
                    <div class="kpi-sub">BM25 {1 - alpha_val:.2f} / BGE {alpha_val:.2f}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Citations Grounded</div>
                    <div class="kpi-value">{cite_metric}</div>
                    <div class="kpi-sub">Automated Regex Audit</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if gen["error"]:
            st.error(gen["error"])
        else:
            formatted_body = format_answer_html(gen["answer"], retrieved_id_set)
            badge = audit_badge_html(audit)
            st.markdown(
                f"""
                <div class="answer-card">
                    <div class="answer-header">
                        <span class="answer-title">Grounded Synthesis</span>
                        {badge}
                    </div>
                    <div class="answer-body">{formatted_body}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        cited_set = set(audit.get("valid_ids", []))
        st.markdown(f"#### Retrieved Evidence Pool (Top-{top_k_docs})")
        for doc in retrieved_docs:
            did = str(doc["doc_id"])
            is_cited = did in cited_set
            cited_tag = " • CITED IN ANSWER" if is_cited else ""
            b_rank = f"#{doc['bm25_rank']}" if doc["bm25_rank"] else "—"
            d_rank = f"#{doc['dense_rank']}" if doc["dense_rank"] else "—"
            exp_label = (
                f"Rank #{doc['rank']}  |  Doc {did}  |  W-RRF {doc['score']:.5f}  "
                f"(BM25: {b_rank}, BGE: {d_rank}){cited_tag}"
            )
            with st.expander(exp_label, expanded=(is_cited or doc["rank"] <= 2)):
                st.markdown(
                    f"""
                    <div class="doc-card {'doc-card-cited' if is_cited else ''}">
                        <div class="doc-meta">
                            <span class="doc-id-tag">Doc {did}</span>
                            <span>
                                <span class="rank-chip">BM25: {b_rank}</span>
                                <span class="rank-chip">BGE: {d_rank}</span>
                                <span class="rank-chip {'rank-chip-hit' if is_cited else ''}">
                                    {'Cited' if is_cited else 'Context'}
                                </span>
                            </span>
                        </div>
                        <div class="doc-snippet">{clean_fiqa_text(doc['text'])}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ================= TAB 2: LIVE RETRIEVER WORKBENCH =================
with tab2:
    st.caption(
        "Inspect lexical vs. semantic retrieval shifts across BM25, Dense BGE (Pure NumPy), "
        "Equal RRF (α=0.50), and Weighted RRF (α=0.95) on any query."
    )

    wb_col1, wb_col2 = st.columns([3.4, 1.1], vertical_alignment="bottom")
    with wb_col1:
        wb_query = st.text_input(
            "Diagnostic Query",
            value="Regarding Australian CBS takeover of TEN",
            key="wb_query_input",
        )
    with wb_col2:
        wb_btn = st.button("Compare", type="primary", width="stretch")

    run_ce = st.checkbox(
        "Enable optional Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2, K=20 on CPU — adds ~1.5s)",
        value=False,
    )

    if wb_btn and wb_query.strip():
        t0 = time.perf_counter()
        b_hits = bm25.search(wb_query, k=50)
        t_bm25 = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        d_hits = dense_bge.search(wb_query, k=50)
        t_dense = (time.perf_counter() - t1) * 1000

        eq_rrf = run_weighted_rrf(b_hits, d_hits, alpha=0.50, rrf_k=60, top_k=5)
        w_rrf = run_weighted_rrf(b_hits, d_hits, alpha=0.95, rrf_k=60, top_k=5)

        b_top5 = {h["doc_id"] for h in b_hits[:5]}
        d_top5 = {h["doc_id"] for h in d_hits[:5]}
        overlap = b_top5.intersection(d_top5)

        st.markdown(
            f"""
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-label">BM25 Latency</div>
                    <div class="kpi-value">{t_bm25:.1f} ms</div>
                    <div class="kpi-sub">Sparse Inverted Index</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Dense BGE Latency</div>
                    <div class="kpi-value">{t_dense:.1f} ms</div>
                    <div class="kpi-sub">57.6k × 384 Dot Product</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Top-5 Overlap</div>
                    <div class="kpi-value">{len(overlap)} / 5</div>
                    <div class="kpi-sub">Lexical vs. Dense Recall</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Fusion Pool</div>
                    <div class="kpi-value">K = 50</div>
                    <div class="kpi-sub">k = 60 Smoothing</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        def render_col_cards(hits_list, score_fmt, show_ranks=False):
            for h in hits_list[:5]:
                did = str(h["doc_id"])
                snippet = clean_fiqa_text(id_to_text.get(did, ""), max_chars=210, single_line=True)
                in_both = did in overlap
                if show_ranks:
                    br = f"#{h['bm25_rank']}" if h.get("bm25_rank") else "—"
                    dr = f"#{h['dense_rank']}" if h.get("dense_rank") else "—"
                    right_meta = f'<span class="rank-chip">B:{br} • D:{dr}</span>'
                else:
                    right_meta = f'<span class="rank-chip">{score_fmt(h["score"])}</span>'
                st.markdown(
                    f"""
                    <div class="doc-card {'doc-card-cited' if in_both else ''}">
                        <div class="doc-meta">
                            <span class="doc-id-tag">#{h['rank']} • Doc {did}</span>
                            {right_meta}
                        </div>
                        <div class="doc-snippet">{snippet}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown("##### 1. BM25 Sparse")
            render_col_cards(b_hits, lambda s: f"BM25 {s:.2f}")
        with c2:
            st.markdown("##### 2. Dense BGE")
            render_col_cards(d_hits, lambda s: f"Cos {s:.3f}")
        with c3:
            st.markdown("##### 3. Equal RRF (α=0.50)")
            render_col_cards(eq_rrf, lambda s: f"{s:.4f}", show_ranks=True)
        with c4:
            st.markdown("##### 4. Weighted RRF (α=0.95)")
            render_col_cards(w_rrf, lambda s: f"{s:.4f}", show_ranks=True)

        if run_ce:
            try:
                ce = load_cross_encoder()
            except ImportError:
                st.info(
                    "Live Cross-Encoder reranking requires PyTorch and is enabled in local mode; "
                    "full 648-query Cross-Encoder benchmarks are available in Tab 3."
                )
                ce = None

            if ce is not None:
                pool20 = run_weighted_rrf(b_hits, d_hits, alpha=0.95, rrf_k=60, top_k=20)
                t_ce0 = time.perf_counter()
                pairs = [(wb_query, id_to_text.get(h["doc_id"], "")[:1500]) for h in pool20]
                ce_scores = ce.predict(pairs)
                t_ce = (time.perf_counter() - t_ce0) * 1000

                reranked = sorted(zip(pool20, ce_scores), key=lambda x: x[1], reverse=True)[:5]
                st.markdown("---")
                st.markdown(f"##### 5. Cross-Encoder Reranker (K=20 Pool • CPU Latency: `{t_ce:.1f} ms`)")
                for rank, (h, sc) in enumerate(reranked, start=1):
                    did = str(h["doc_id"])
                    br = f"#{h['bm25_rank']}" if h.get("bm25_rank") else "—"
                    dr = f"#{h['dense_rank']}" if h.get("dense_rank") else "—"
                    snippet = clean_fiqa_text(id_to_text.get(did, ""), max_chars=280, single_line=True)
                    st.markdown(
                        f"""
                        <div class="doc-card">
                            <div class="doc-meta">
                                <span class="doc-id-tag">#{rank} • Doc {did}</span>
                                <span>
                                    <span class="rank-chip">BM25: {br}</span>
                                    <span class="rank-chip">BGE: {dr}</span>
                                    <span class="rank-chip rank-chip-hit">CE: {float(sc):.3f}</span>
                                </span>
                            </div>
                            <div class="doc-snippet">{snippet}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


# ================= TAB 3: INTERACTIVE PLOTLY BENCHMARKS =================
with tab3:
    master_csv = RESULTS_DIR / "master_benchmark_v2.csv"
    alpha_csv = RESULTS_DIR / "weighted_rrf_ablation.csv"
    rrf_k_csv = RESULTS_DIR / "rrf_k_sensitivity.csv"
    cases_csv = RESULTS_DIR / "qualitative_case_studies.csv"
    wins_csv = RESULTS_DIR / "bge_reranker_wins.csv"

    if master_csv.exists():
        df_master = pd.read_csv(master_csv)
        df_master = (
            df_master.assign(_sort=df_master["System"].map(stage_sort_key))
            .sort_values("_sort")
            .drop(columns=["_sort"])
            .reset_index(drop=True)
        )

        best_row = df_master.loc[df_master["NDCG@10"].idxmax()]
        bm25_ndcg = float(df_master.loc[df_master["System"].str.startswith("1. BM25"), "NDCG@10"].iloc[0])
        rel_gain = ((float(best_row["NDCG@10"]) - bm25_ndcg) / bm25_ndcg) * 100

        st.markdown(
            f"""
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-label">Winning Architecture</div>
                    <div class="kpi-value">W-RRF (α=0.95)</div>
                    <div class="kpi-sub">BM25 + bge-small-en-v1.5</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Best NDCG@10</div>
                    <div class="kpi-value">{best_row['NDCG@10']:.4f}</div>
                    <div class="kpi-sub">+{rel_gain:.1f}% relative over BM25</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Best MRR@10</div>
                    <div class="kpi-value">{best_row['MRR@10']:.4f}</div>
                    <div class="kpi-sub">Recall@10: {best_row['Recall@10']:.4f}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Evaluation Scope</div>
                    <div class="kpi-value">648 Queries</div>
                    <div class="kpi-sub">57,638 Financial Docs</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Chart 1: Master Pipeline Comparison (Clean Hover, Zero Label Overlap)
        fig1 = go.Figure()
        fig1.add_trace(
            go.Bar(
                name="NDCG@10",
                x=df_master["System"],
                y=df_master["NDCG@10"],
                marker_color="#38BDF8",
                hovertemplate="NDCG@10: <b>%{y:.4f}</b><extra></extra>",
            )
        )
        fig1.add_trace(
            go.Bar(
                name="MRR@10",
                x=df_master["System"],
                y=df_master["MRR@10"],
                marker_color="#10B981",
                hovertemplate="MRR@10: <b>%{y:.4f}</b><extra></extra>",
            )
        )
        fig1.add_trace(
            go.Bar(
                name="Recall@10",
                x=df_master["System"],
                y=df_master["Recall@10"],
                marker_color="#818CF8",
                hovertemplate="Recall@10: <b>%{y:.4f}</b><extra></extra>",
            )
        )
        fig1.update_layout(
            **plotly_layout_defaults("Full Architecture Ablation — BEIR FiQA (648 Queries)", height=460, top_margin=82),
            barmode="group",
            bargap=0.22,
            bargroupgap=0.06,
            hovermode="x unified",
            xaxis=dict(tickangle=-22, gridcolor="#1E293B", tickfont=dict(size=10.5)),
            yaxis=dict(range=[0, 0.56], gridcolor="#1E293B", title="Score"),
            legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1),
        )
        st.plotly_chart(fig1, width="stretch")

        # Row 2: Accuracy vs Latency Pareto + Weighted RRF Alpha Sweep
        ch1, ch2 = st.columns(2)

        with ch1:
            colors = [
                "#10B981" if "3c." in s else ("#F59E0B" if ("4." in s or "5." in s) else "#38BDF8")
                for s in df_master["System"]
            ]
            short_tags = [short_system_label(s) for s in df_master["System"]]
            fig_pareto = go.Figure()
            fig_pareto.add_trace(
                go.Scatter(
                    x=df_master["Mean_Latency_ms"],
                    y=df_master["NDCG@10"],
                    mode="markers+text",
                    text=short_tags,
                    textposition="top center",
                    textfont=dict(size=10, color="#CBD5E1"),
                    marker=dict(size=12, color=colors, line=dict(width=1.5, color="#0B0F19")),
                    customdata=df_master[["System", "MRR@10", "Recall@10", "P95_Latency_ms"]].values,
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "NDCG@10: %{y:.4f}<br>"
                        "MRR@10: %{customdata[1]:.4f}<br>"
                        "Recall@10: %{customdata[2]:.4f}<br>"
                        "Mean Latency: %{x:.1f} ms<br>"
                        "P95 Latency: %{customdata[3]:.1f} ms<extra></extra>"
                    ),
                )
            )
            fig_pareto.update_layout(
                **plotly_layout_defaults("Accuracy vs. Latency Pareto Trade-off", height=410, top_margin=64),
                xaxis=dict(title="Mean Query Latency (ms)", gridcolor="#1E293B", range=[-25, 650]),
                yaxis=dict(title="NDCG@10", gridcolor="#1E293B", range=[0.21, 0.44]),
            )
            st.plotly_chart(fig_pareto, width="stretch")

        with ch2:
            if alpha_csv.exists():
                df_alpha = pd.read_csv(alpha_csv)
                fig_alpha = go.Figure()
                fig_alpha.add_trace(
                    go.Scatter(
                        x=df_alpha["alpha_dense"],
                        y=df_alpha["NDCG@10"],
                        mode="lines+markers",
                        name="NDCG@10",
                        line=dict(color="#38BDF8", width=2.5),
                    )
                )
                fig_alpha.add_trace(
                    go.Scatter(
                        x=df_alpha["alpha_dense"],
                        y=df_alpha["MRR@10"],
                        mode="lines+markers",
                        name="MRR@10",
                        line=dict(color="#10B981", width=2.2),
                    )
                )
                fig_alpha.add_trace(
                    go.Scatter(
                        x=df_alpha["alpha_dense"],
                        y=df_alpha["Recall@50"],
                        mode="lines+markers",
                        name="Recall@50 (Pool)",
                        line=dict(color="#F59E0B", width=2, dash="dot"),
                    )
                )
                fig_alpha.add_vline(
                    x=0.95,
                    line_dash="dash",
                    line_color="#10B981",
                    annotation_text="Optimal α = 0.95",
                    annotation_position="top left",
                )
                fig_alpha.update_layout(
                    **plotly_layout_defaults("Weighted RRF α Sweep (Asymmetric Fusion)", height=410, top_margin=78),
                    hovermode="x unified",
                    xaxis=dict(title="Dense BGE Weight (α)", gridcolor="#1E293B"),
                    yaxis=dict(title="Score", gridcolor="#1E293B"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig_alpha, width="stretch")

        # Row 3: RRF k Sensitivity + Master Table
        ch3, ch4 = st.columns([1, 1.35])
        with ch3:
            if rrf_k_csv.exists():
                df_k = pd.read_csv(rrf_k_csv)
                fig_k = go.Figure()
                fig_k.add_trace(
                    go.Scatter(
                        x=df_k["rrf_k"],
                        y=df_k["NDCG@10"],
                        mode="lines+markers",
                        name="NDCG@10",
                        line=dict(color="#38BDF8", width=2.5),
                    )
                )
                fig_k.add_trace(
                    go.Scatter(
                        x=df_k["rrf_k"],
                        y=df_k["MRR@10"],
                        mode="lines+markers",
                        name="MRR@10",
                        line=dict(color="#10B981", width=2),
                    )
                )
                fig_k.add_trace(
                    go.Scatter(
                        x=df_k["rrf_k"],
                        y=df_k["Recall@10"],
                        mode="lines+markers",
                        name="Recall@10",
                        line=dict(color="#818CF8", width=2),
                    )
                )
                fig_k.update_layout(
                    **plotly_layout_defaults("RRF Smoothing Constant (k) Sensitivity", height=380, top_margin=78),
                    hovermode="x unified",
                    xaxis=dict(
                        title="RRF Smoothing Constant k",
                        gridcolor="#1E293B",
                        tickmode="array",
                        tickvals=[1, 10, 30, 60, 100, 150],
                    ),
                    yaxis=dict(title="Score", gridcolor="#1E293B"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig_k, width="stretch")

        with ch4:
            st.markdown("##### Master Benchmark Table (648 Test Queries)")
            st.dataframe(df_master, width="stretch", hide_index=True, height=340)

        # Qualitative Error Taxonomy & Reranker Wins
        st.markdown("---")
        st.markdown("##### Qualitative Error Taxonomy & Query-Level Case Studies")
        if cases_csv.exists():
            df_cases = pd.read_csv(cases_csv)
            st.dataframe(
                df_cases[
                    [
                        "Category",
                        "query_id",
                        "query_text",
                        "bm25_ndcg10",
                        "bge_ndcg10",
                        "hybrid_rrf_ndcg10",
                        "rerank_k50_ndcg10",
                    ]
                ],
                width="stretch",
                hide_index=True,
            )
        if wins_csv.exists():
            with st.expander("View Top Queries Where BGE-Reranker-Base Rescued Pure Dense BGE"):
                st.dataframe(pd.read_csv(wins_csv), width="stretch", hide_index=True)
    else:
        st.warning("Benchmark CSVs not found in `results/`. Run `src/finalize_benchmark.py` first.")