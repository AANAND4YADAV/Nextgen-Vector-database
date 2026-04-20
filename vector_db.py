import urllib.request
import json
from brute_force import BruteForceIndex
from kdtree import KDTreeIndex
from hnsw import HNSWIndex

class VectorDB:
    def __init__(self, index_type='brute_force', embed_model='all-MiniLM-L6-v2', embed_backend='sentence-transformers'):
        """
        Initialize the Vector Database.
        index_type: 'brute_force', 'kdtree', or 'hnsw'
        embed_backend: 'ollama' or 'sentence-transformers'
        embed_model: embedding model name.
        """
        self.index_type = index_type
        self.embed_model = embed_model
        self.embed_backend = embed_backend
        self.documents = {}  # Store mapping doc_id -> Document

        if embed_backend == 'sentence-transformers':
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(embed_model)
        else:
            self.model = None

        if index_type == 'brute_force':
            self.index = BruteForceIndex(metric='euclidean')
        elif index_type == 'kdtree':
            self.index = KDTreeIndex()
        elif index_type == 'hnsw':
            self.index = HNSWIndex(m=16, ef_construction=100, m0=32)
        else:
            raise ValueError(f"Unknown index type: {index_type}")

    def get_embedding(self, text):
        """
        Get embeddings according to the configured backend.
        """
        if self.embed_backend == 'sentence-transformers':
            vector = self.model.encode(text)
            if hasattr(vector, "tolist"):
                return vector.tolist()
            return vector

        # Fallback to local Ollama API
        url = "http://localhost:11434/api/embed"
        data = {
            "model": self.embed_model,
            "input": text
        }
        
        req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode())
                return result['embeddings'][0]
        except Exception as e:
            raise RuntimeError(f"Failed to get embedding from Ollama: {e}")

    def add(self, doc_id, text, metadata=None):
        """
        Embed the document and add it to the index.
        """
        vector = self.get_embedding(text)
        self.documents[doc_id] = {
            "text": text,
            "metadata": metadata or {}
        }
        self.index.add(vector, doc_id)

    def search(self, query, top_k=5):
        """
        Search for the top_k most similar documents to the query.
        """
        query_vector = self.get_embedding(query)
        results = self.index.search(query_vector, k=top_k)
        
        search_results = []
        for dist, doc_id in results:
            doc = self.documents[doc_id]
            search_results.append({
                "doc_id": doc_id,
                "text": doc["text"],
                "metadata": doc["metadata"],
                "distance": dist
            })
            
        return search_results
