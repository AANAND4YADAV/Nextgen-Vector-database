import streamlit as st
import time
import numpy as np
import uuid
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from vector_db import VectorDB

st.set_page_config(page_title="NextGen Vector DB", layout="wide", page_icon="🗄️")

st.markdown("""
<style>
.stApp { background-color: #0f111a; color: #e2e8f0; }
.main-title {
    background: -webkit-linear-gradient(45deg, #FF6B6B, #4ECDC4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3.5rem; font-weight: 800; margin-bottom: 0.5rem;
}
.custom-card {
    background-color: #1e2130; border-radius: 12px; padding: 20px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.3); margin-bottom: 20px;
    border: 1px solid #2d3345;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.custom-card:hover { transform: translateY(-5px); box-shadow: 0 8px 15px rgba(0,0,0,0.4); }
div.stButton > button {
    background: linear-gradient(90deg, #4ECDC4 0%, #2f8e87 100%);
    color: white; border: none; border-radius: 25px;
    padding: 10px 24px; font-weight: bold; transition: all 0.3s ease;
}
div.stButton > button:hover { transform: scale(1.05); box-shadow: 0 5px 15px rgba(78,205,196,0.4); }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">🗄️ NextGen Vector Database</h1>', unsafe_allow_html=True)
st.markdown("### Compare Brute Force, KD-Tree, and HNSW simultaneously!")

if "dbs_inited" not in st.session_state:
    st.session_state.dbs_inited = False
if "docs" not in st.session_state:
    st.session_state.docs = []
if "dbs" not in st.session_state:
    st.session_state.dbs = {}

with st.sidebar:
    st.header("⚙️ Configuration")
    embed_backend = st.selectbox(
        "Embedding Backend",
        ["sentence-transformers", "ollama"],
        help="Streamlit Cloud pe sirf sentence-transformers kaam karega."
    )
    if embed_backend == "ollama":
        embed_model = st.text_input("Ollama Model", value="nomic-embed-text")
        st.warning("⚠️ Ollama sirf local machine pe kaam karta hai.")
    else:
        embed_model = st.selectbox(
            "Model",
            ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "paraphrase-MiniLM-L3-v2"],
            help="all-MiniLM-L6-v2 fastest & lightest hai."
        )

    st.markdown("---")
    st.markdown("### Search Settings")
    top_k = st.slider("Top K Results", min_value=1, max_value=20, value=5)

    if st.button("🔄 Reset Index"):
        st.session_state.dbs_inited = False
        st.session_state.dbs = {}
        st.session_state.docs = []
        st.cache_resource.clear()
        st.success("Index reset!")


@st.cache_resource
def make_db(index_type: str, backend: str, model_name: str) -> VectorDB:
    return VectorDB(index_type=index_type, embed_model=model_name, embed_backend=backend)


SAMPLE_TEXT = """Machine learning (ML) is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can effectively generalize and thus perform tasks without explicit instructions.

Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised.

Artificial intelligence (AI) is intelligence—perceiving, synthesizing, and inferring information—demonstrated by machines, as opposed to intelligence displayed by animals and humans.

A vector database stores vectors (fixed-length lists of numbers) along with other data items and typically implements one or more Approximate Nearest Neighbor (ANN) algorithms.

A k-d tree (short for k-dimensional tree) is a space-partitioning data structure for organizing points in a k-dimensional space, useful for range searches and nearest neighbor searches.

Hierarchical Navigable Small World (HNSW) is a state-of-the-art algorithm for approximate nearest neighbor search. It builds a multi-layer graph where the bottom layer captures fine-grained structure and higher layers capture macro structure.

Transformers are a type of deep learning model architecture introduced in 2017. They use self-attention mechanisms to process sequential data like natural language text efficiently.

An embedding is a representation of a discrete variable in a continuous vector space. In NLP, word embeddings capture semantic relationships between words based on their context in a large corpus.

Cosine similarity measures the cosine of the angle between two non-zero vectors and is widely used in high-dimensional similarity search and retrieval algorithms.

Euclidean distance is the straight-line distance between two points in space, a fundamental metric in distance-based clustering and vector indexing engines.
"""

tab1, tab2 = st.tabs(["📝 Insert Document", "🔍 Query & Compare"])

with tab1:
    st.header("Insert Your Corpus")
    st.info("Input a large document below. It will be split by paragraphs and indexed via 3 different algorithms.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📄 Load Sample Basics Data (10 paragraphs)"):
            st.session_state.doc_input = SAMPLE_TEXT
            st.rerun()
    with col2:
        if st.button("📰 Load AG News Dataset (1000 paragraphs)"):
            try:
                with st.spinner("Downloading AG News from HuggingFace..."):
                    from datasets import load_dataset
                    dataset = load_dataset("ag_news", split="test")
                    paragraphs = [dataset[i]['text'] for i in range(1000)]
                    st.session_state.doc_input = "\n\n".join(paragraphs)
                    st.rerun()
            except ImportError:
                st.error("Run: `pip install datasets`")

    doc_input = st.text_area(
        "Document Text", height=300,
        value=st.session_state.get('doc_input', ''),
        key='text_area_doc'
    )

    if st.button("⚡ Process & Index Document", key="process_btn"):
        if not doc_input.strip():
            st.error("Please insert some text to index.")
        else:
            paragraphs = [p.strip() for p in doc_input.split("\n\n") if p.strip()]
            if not paragraphs:
                st.error("No valid paragraphs found.")
            else:
                my_bar = st.progress(0, text="Initializing...")
                try:
                    from brute_force import BruteForceIndex
                    from kdtree import KDTreeIndex
                    from hnsw import HNSWIndex

                    dbs = {
                        "Brute Force": make_db("brute_force", embed_backend, embed_model),
                        "KD-Tree":     make_db("kdtree",      embed_backend, embed_model),
                        "HNSW":        make_db("hnsw",        embed_backend, embed_model),
                    }
                    dbs["Brute Force"].index = BruteForceIndex(metric='cosine')
                    dbs["KD-Tree"].index     = KDTreeIndex()
                    dbs["HNSW"].index        = HNSWIndex(m=16, ef_construction=100, m0=32)
                    for db in dbs.values():
                        db.documents = {}

                    st.session_state.dbs = dbs
                    st.session_state.dbs_inited = True
                    st.session_state.docs = []

                    vectors = []
                    ref_db = dbs["Brute Force"]
                    for i, para in enumerate(paragraphs):
                        vec = ref_db.get_embedding(para)
                        vectors.append((para, vec))
                        my_bar.progress(
                            int((i + 1) / len(paragraphs) * 50),
                            text=f"🔢 Embedding {i+1}/{len(paragraphs)}..."
                        )

                    for i, (para, vec) in enumerate(vectors):
                        doc_id = str(uuid.uuid4())
                        for db in dbs.values():
                            db.documents[doc_id] = {"text": para, "metadata": {}}
                            db.index.add(vec, doc_id)
                        st.session_state.docs.append(para)
                        my_bar.progress(
                            50 + int((i + 1) / len(vectors) * 50),
                            text=f"📥 Indexing {i+1}/{len(vectors)}..."
                        )

                    my_bar.empty()
                    st.success(
                        f"✅ {len(paragraphs)} paragraphs indexed using "
                        f"**{embed_backend}** / `{embed_model}`!"
                    )

                except Exception as e:
                    my_bar.empty()
                    st.error(f"❌ Error: {e}")
                    st.exception(e)

with tab2:
    st.header("Query the Indexes")
    query = st.text_input("Enter your search query:", placeholder="e.g. What is a neural network?")

    if st.button("🔍 Search", key="search_btn"):
        if not query.strip():
            st.warning("Please enter a query.")
        elif not st.session_state.dbs_inited:
            st.warning("⚠️ Pehle 'Insert Document' tab mein document index karo!")
        else:
            st.markdown("### 📊 Search Results Comparison")
            cols = st.columns(3)
            try:
                for idx, (db_name, db) in enumerate(st.session_state.dbs.items()):
                    with cols[idx]:
                        st.markdown(f"#### {db_name}")
                        start = time.perf_counter()
                        results = db.search(query, top_k=top_k)
                        elapsed_ms = (time.perf_counter() - start) * 1000
                        st.info(f"⏱️ **{elapsed_ms:.4f} ms**")
                        if not results:
                            st.warning("No results found.")
                        for r_idx, res in enumerate(results, 1):
                            st.markdown(
                                f'<div class="custom-card">'
                                f'<div style="font-size:0.8rem;color:#4ECDC4;margin-bottom:5px;">'
                                f'#{r_idx} | Score: {res["distance"]:.4f}</div>'
                                f'{res["text"]}'
                                f'</div>',
                                unsafe_allow_html=True
                            )
            except Exception as e:
                st.error(f"❌ Search error: {e}")
                st.exception(e)
