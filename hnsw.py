import math
import heapq
import random

def euclidean_distance(v1, v2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))

class HNSWIndex:
    def __init__(self, m=16, ef_construction=100, m0=32, level_mult=1 / math.log(16)):
        self.M = m
        self.M0 = m0
        self.ef_construction = ef_construction
        self.level_mult = level_mult
        
        self.vectors = {} # node_id -> vector
        self.doc_ids = {} # node_id -> doc_id
        self.graphs = []  # list of adjacency list dicts for each layer
        self.enter_point = None
        self.max_level = -1
        self.node_count = 0

    def get_random_level(self):
        f = random.uniform(0, 1)
        if f == 0:
            f = 1e-6
        return int(-math.log(f) * self.level_mult)

    def search_layer(self, query, enter_points, ef, layer):
        visited = set(enter_points)
        candidates = [] # min-heap
        nearest = []    # max-heap
        
        for ep in enter_points:
            dist = euclidean_distance(query, self.vectors[ep])
            heapq.heappush(candidates, (dist, ep))
            heapq.heappush(nearest, (-dist, ep))

        results = []
        
        while candidates:
            c_dist, c = heapq.heappop(candidates)
            if nearest and c_dist > -nearest[0][0]:
                break
                
            for e in self.graphs[layer].get(c, []):
                if e not in visited:
                    visited.add(e)
                    e_dist = euclidean_distance(query, self.vectors[e])
                    
                    if len(nearest) < ef or e_dist < -nearest[0][0]:
                        heapq.heappush(candidates, (e_dist, e))
                        heapq.heappush(nearest, (-e_dist, e))
                        if len(nearest) > ef:
                            heapq.heappop(nearest)
                            
        while nearest:
            dist, node = heapq.heappop(nearest)
            results.append((-dist, node))
        results.reverse()
        return results

    def add(self, vector, doc_id):
        node_id = self.node_count
        self.node_count += 1
        self.vectors[node_id] = vector
        self.doc_ids[node_id] = doc_id
        
        level = self.get_random_level()
        while len(self.graphs) <= level:
            self.graphs.append({})
            
        if self.enter_point is None:
            self.enter_point = node_id
            self.max_level = level
            for l in range(level + 1):
                self.graphs[l][node_id] = []
            return
            
        ep = [self.enter_point]
        # Search down to level+1
        for l in range(self.max_level, level, -1):
            res = self.search_layer(vector, ep, 1, l)
            if res:
                ep = [res[0][1]]
                
        # Connect in layers from `level` down to 0
        for l in range(min(level, self.max_level), -1, -1):
            self.graphs[l][node_id] = []
            res = self.search_layer(vector, ep, self.ef_construction, l)
            ep = [x[1] for x in res]
            
            # Select M nearest neighbors
            max_m = self.M if l > 0 else self.M0
            neighbors = ep[:max_m]
            
            # Add bidirectional edges
            for n in neighbors:
                self.graphs[l][node_id].append(n)
                self.graphs[l][n].append(node_id)
                # prune long edges
                if len(self.graphs[l][n]) > max_m:
                    # Simple pruning: keep M closest
                    n_neighbors = self.graphs[l][n]
                    n_dists = [(euclidean_distance(self.vectors[n], self.vectors[nn]), nn) for nn in n_neighbors]
                    n_dists.sort()
                    self.graphs[l][n] = [nn for _, nn in n_dists[:max_m]]
                    
        if level > self.max_level:
            self.max_level = level
            self.enter_point = node_id

    def search(self, query_vector, k=5, ef_search=50):
        if self.enter_point is None:
            return []
            
        ep = [self.enter_point]
        ef = max(k, ef_search)
        
        for l in range(self.max_level, 0, -1):
            res = self.search_layer(query_vector, ep, 1, l)
            if res:
                ep = [res[0][1]]
                
        res = self.search_layer(query_vector, ep, ef, 0)
        
        # res contains (distance, node_id)
        results = [(dist, self.doc_ids[node_id]) for dist, node_id in res[:k]]
        return results
