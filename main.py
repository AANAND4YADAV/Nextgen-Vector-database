from vector_db import VectorDB

def main():
    # Make sure Ollama is running and has the model, e.g. run: ollama pull nomic-embed-text
    # You can change the model to whatever you have pulled locally.
    embed_model = 'all-minilm' # usually small and quick, alternatively 'nomic-embed-text' or 'mxbai-embed-large'
    
    documents = [
        {"id": "doc1", "text": "Python is a high-level programming language."},
        {"id": "doc2", "text": "Machine learning represents a major step forward in artificial intelligence."},
        {"id": "doc3", "text": "The quick brown fox jumps over the lazy dog."},
        {"id": "doc4", "text": "Deep learning uses neural networks with many layers."},
        {"id": "doc5", "text": "Vectors are geometric objects that have magnitude and direction."},
        {"id": "doc6", "text": "Database indexing speeds up regular query performance."},
        {"id": "doc7", "text": "A programming language is a system of notation for writing computer programs."}
    ]

    algorithms = ['brute_force', 'kdtree', 'hnsw']
    query = "What is a neural network?"
    
    print("="*60)
    print(f"Testing Vector DB with Ollama embeddings (Model: {embed_model})")
    print("="*60)

    for algo in algorithms:
        print(f"\n--- Initializing VectorDB with {algo} ---")
        # Try-catch inside just in case Ollama relies on a certain model
        db = VectorDB(index_type=algo, embed_model=embed_model)
        
        # Add documents
        try:
            for doc in documents:
                db.add(doc["id"], doc["text"])
                
            # Perform search
            results = db.search(query, top_k=3)
            print(f"\nResults for '{algo}':")
            for rank, res in enumerate(results, 1):
                print(f"[{rank}] ID: {res['doc_id']} | Distance: {res['distance']:.4f} | Text: {res['text']}")
        except RuntimeError as e:
            print(f"Error: {e}")
            print("Please ensure Ollama is running and you have the model installed: `ollama pull all-minilm`")
            break

if __name__ == "__main__":
    main()
