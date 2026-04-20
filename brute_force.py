import math

def euclidean_distance(v1, v2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))

def cosine_similarity(v1, v2):
    dot_product = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a ** 2 for a in v1))
    mag2 = math.sqrt(sum(a ** 2 for a in v2))
    if mag1 == 0 or mag2 == 0:
        return 0
    return dot_product / (mag1 * mag2)

class BruteForceIndex:
    def __init__(self, metric='euclidean'):
        self.vectors = []
        self.metric = metric

    def add(self, vector, doc_id):
        self.vectors.append((vector, doc_id))

    def search(self, query_vector, k=5):
        if not self.vectors:
            return []
            
        distances = []
        for vec, doc_id in self.vectors:
            if self.metric == 'euclidean':
                dist = euclidean_distance(query_vector, vec)
                distances.append((dist, doc_id))
            elif self.metric == 'cosine':
                # Higher cosine similarity is better, so we use negative for sorting 
                # or just sort in descending order
                sim = cosine_similarity(query_vector, vec)
                distances.append((sim, doc_id))
        
        if self.metric == 'euclidean':
            distances.sort(key=lambda x: x[0])
        else:
            distances.sort(key=lambda x: x[0], reverse=True)
            
        return distances[:k]
