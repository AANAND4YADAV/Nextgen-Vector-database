import streamlit as st
import time
import numpy as np
import uuid
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vector_db import VectorDB

# Page config
st.set_page_config(page_title="Vector DB Visualizer", layout="wide", page_icon="🔍")

# Custom CSS for aesthetics
st.markdown("""
<style>
    .stApp {
        background-color: #0f111a;
        color: #e2e8f0;
    }
    .main-title {
        background: -webkit-linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .custom-card {
        background-color: #1e2130;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 20px;
        border: 1px solid #2d3345;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .custom-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.4);
    }
    div.stButton > button {
        background: linear-gradient(90deg, #4ECDC4 0%, #2f8e87 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 24px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(78, 205, 196, 0.4);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">🗄️ NextGen Vector Database</h1>', unsafe_allow_html=True)
st.markdown("### Compare Brute Force, KD-Tree, and HNSW simultaneously!")

@st.cache_resource
def load_db(index_type, backend="sentence-transformers", model_name="all-MiniLM-L6-v2"):
    return VectorDB(index_type=index_type, embed_model=model_name, embed_backend=backend)

# Init DB dictionaries
if "dbs_inited" not in st.session_state:
    st.session_state.dbs_inited = False
if "docs" not in st.session_state:
    st.session_state.docs = []

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Configuration")
    embed_backend = st.selectbox("Embedding Backend", ["sentence-transformers", "ollama"])
    if embed_backend == "ollama":
        embed_model = st.text_input("Ollama Model", value="nomic-embed-text")
    else:
        embed_model = st.text_input("HF Model Name", value="all-MiniLM-L6-v2")
    
    st.markdown("---")
    st.markdown("### Search Settings")
    top_k = st.slider("Top K Results", min_value=1, max_value=20, value=5)

# Main UI
tab1, tab2 = st.tabs(["📝 Insert Document", "🔍 Query & Compare"])

with tab1:
    st.header("Insert Your Corpus")
    st.info("Input a large document below. It will be split by paragraphs and index via 3 different algorithms.")
    
    sample_text = """Machine learning (ML) is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can effectively generalize and thus perform tasks without explicit instructions. Recently, artificial neural networks have been able to surpass many previous approaches in performance.

Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised.

Artificial intelligence (AI) is intelligence—perceiving, synthesizing, and inferring information—demonstrated by machines, as opposed to intelligence displayed by animals and humans. Example tasks in which this is done include speech recognition, computer vision, translation between (natural) languages, as well as other mappings of inputs.

A vector database is a database that can store vectors (fixed-length lists of numbers) along with other data items. Vector databases typically implement one or more Approximate Nearest Neighbor (ANN) algorithms, so that one can search the database with a query vector to retrieve the closest matching database records.

A k-d tree (short for k-dimensional tree) is a space-partitioning data structure for organizing points in a k-dimensional space. k-d trees are a useful data structure for several applications, such as searches involving a multidimensional search key (e.g. range searches and nearest neighbor searches).

Hierarchical Navigable Small World (HNSW) is a state-of-the-art algorithm for approximate nearest neighbor search. It builds a multi-layer graph where the bottom layer captures the fine-grained structure of the dataset and higher layers capture macro structure.

Transformers are a type of deep learning model architecture introduced in 2017. They are designed to process sequential data, such as natural language text, using self-attention mechanisms. This allows them to handle long-range dependencies more efficiently than older recurrent neural networks.

An embedding is a representation of a discrete variable in a continuous vector space. In natural language processing, word embeddings are widely used to capture semantic relationships between words based on their context in a large corpus.

Cosine similarity is a measure of similarity between two non-zero vectors of an inner product space. It measures the cosine of the angle between them and is highly used in high-dimensional search and retrieval algorithms.

Euclidean distance is the "ordinary" straight-line distance between two points in Euclidean space. It is a fundamental metric in distance-based clustering and vector indexing engines.
    """
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Load Sample Basics Data (10 paragraphs)"):
            st.session_state.doc_input = sample_text
            st.rerun()
    with col2:
        if st.button("Load Large AG News Dataset (1000 paragraphs)"):
            with st.spinner("Downloading AG News Dataset from HuggingFace..."):
                from datasets import load_dataset
                dataset = load_dataset("ag_news", split="test")
                paragraphs = [dataset[i]['text'] for i in range(1000)]
                st.session_state.doc_input = "\n\n".join(paragraphs)
            st.rerun()
            
    doc_input = st.text_area("Document Text", height=300, value=st.session_state.get('doc_input', ''), key='text_area_doc')
    
    if st.button("Process & Index Document", key="process_btn"):
        if not doc_input.strip():
            st.error("Please insert some text to index.")
        else:
            paragraphs = [p.strip() for p in doc_input.split("\n\n") if p.strip()]
            
            if not paragraphs:
                st.error("No valid paragraphs found.")
            else:
                progress_text = "Extracting embeddings and building index..."
                my_bar = st.progress(0, text=progress_text)
                
                try:
                    # Initialize databases
                    dbs = {
                        "Brute Force": load_db("brute_force", embed_backend, embed_model),
                        "KD-Tree": load_db("kdtree", embed_backend, embed_model),
                        "HNSW": load_db("hnsw", embed_backend, embed_model)
                    }
                    
                    st.session_state.dbs = dbs
                    st.session_state.dbs_inited = True
                    
                    # Compute all vectors first to avoid redundant inference 
                    vectors = []
                    for para in paragraphs:
                        vec = dbs["Brute Force"].get_embedding(para)
                        vectors.append((para, vec))
                    
                    for i, (para, vec) in enumerate(vectors):
                        doc_id = str(uuid.uuid4())
                        
                        # Add to all DBs explicitly avoiding duplicate get_embedding calls for speed
                        for db_name, db in st.session_state.dbs.items():
                            db.documents[doc_id] = {"text": para, "metadata": {}}
                            db.index.add(vec, doc_id)
                        
                        my_bar.progress((i + 1) / len(paragraphs), text=f"Indexed {i+1}/{len(paragraphs)} passages")
                        
                    st.success(f"Successfully constructed indices! {len(paragraphs)} vectors added to Brute Force, KD-Tree, and HNSW.")
                    st.session_state.docs.extend(paragraphs)
                    time.sleep(1)
                    my_bar.empty()
                    
                except Exception as e:
                    st.error(f"Error during indexing: {e}")

with tab2:
    st.header("Query the Indexes")
    query = st.text_input("Enter your question or search query:", placeholder="e.g. What is a neural network?")
    
    if st.button("Search Indexes", key="search_btn"):
        if not query:
            st.warning("Please enter a query.")
        elif not st.session_state.dbs_inited:
            st.warning("Please index a document first in the Add Documents tab!")
        else:
            st.markdown("### Search Results Comparison")
            cols = st.columns(3)
            
            try:
                for idx, (db_name, db) in enumerate(st.session_state.dbs.items()):
                    with cols[idx]:
                        st.markdown(f"**{db_name}**")
                        
                        # Time the search process
                        start_time = time.perf_counter()
                        results = db.search(query, top_k=top_k)
                        end_time = time.perf_counter()
                        elapsed_ms = (end_time - start_time) * 1000
                        
                        st.info(f"⏱️ **{elapsed_ms:.4f} ms**")
                        
                        for r_idx, res in enumerate(results, 1):
                            dist = res['distance']
                            text = res['text']
                            st.markdown(f'<div class="custom-card">'
                                        f'<div style="font-size:0.8rem; color:#4ECDC4; margin-bottom:5px;">#{r_idx} | Distance: {dist:.4f}</div>'
                                        f'{text}'
                                        f'</div>', unsafe_allow_html=True)
                                
            except Exception as e:
                st.error(f"Error during search: {e}")
