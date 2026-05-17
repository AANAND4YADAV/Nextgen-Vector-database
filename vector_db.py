import urllib.request
import json
import numpy as np

from brute_force import BruteForceIndex
from kdtree import KDTreeIndex
from hnsw import HNSWIndex


class VectorDB:
    def __init__(self, index_type='brute_force', embed_model='all-MiniLM-L6-v2', embed_backend='ollama'):
        """
        Initialize the Vector Database.
        index_type: 'brute_force', 'kdtree', or 'hnsw'
        embed_backend: 'ollama' or 'sentence-transformers'
        embed_model: embedding model name.
        """
        self.index_type = index_type
        self.embed_model = embed_model
        self.embed_backend = embed_backend
        self.documents = {}  # doc_id -> {"text": ..., "metadata": ...}
        self.model = None    # lazy-loaded only for sentence-transformers

        # ✅ FIX: Lazy import — sentence_transformers sirf tab load ho
        # jab user ne explicitly wo backend select kiya ho
        if embed_backend == 'sentence-transformers':
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(embed_model)
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed!\n"
                    "Run: pip install sentence-transformers\n"
                    "Ya Ollama backend use karo sidebar mein."
                )

        if index_type == 'brute_force':
            self.index = BruteForceIndex(metric='cosine')
        elif index_type == 'kdtree':
            self.index = KDTreeIndex()
        elif index_type == 'hnsw':
            self.index = HNSWIndex(m=16, ef_construction=100, m0=32)
        else:
            raise ValueError(f"Unknown index type: {index_type}")

    def get_embedding(self, text: str) -> list:
        """Get normalized embeddings for the given text."""
        if self.embed_backend == 'sentence-transformers':
            vector = self.model.encode(text, normalize_embeddings=True)
            return vector.tolist()

        # Ollama API
        url = "http://localhost:11434/api/embed"
        data = {"model": self.embed_model, "input": text}
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode())
                vec = np.array(result['embeddings'][0], dtype=np.float32)
                # Normalize for cosine similarity
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                return vec.tolist()
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Ollama server unreachable! Kya Ollama chal raha hai?\n"
                f"Run: ollama serve\n"
                f"Error: {e}"
            )
        except KeyError:
            raise RuntimeError(
                f"Ollama response mein 'embeddings' key nahi mili. "
                f"Model '{self.embed_model}' pull kiya hai?\n"
                f"Run: ollama pull {self.embed_model}"
            )

    def add(self, doc_id: str, text: str, metadata: dict = None):
        """Embed and add a document to the index."""
        vector = self.get_embedding(text)
        self.documents[doc_id] = {"text": text, "metadata": metadata or {}}
        self.index.add(vector, doc_id)

    def search(self, query: str, top_k: int = 5) -> list:
        """Search top_k most similar documents."""
        query_vector = self.get_embedding(query)
        results = self.index.search(query_vector, k=top_k)
        return [
            {
                "doc_id": doc_id,
                "text": self.documents[doc_id]["text"],
                "metadata": self.documents[doc_id]["metadata"],
                "distance": dist
            }
            for dist, doc_id in results
            if doc_id in self.documents
        ]
