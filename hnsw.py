import heapq
import random
import math
import numpy as np


def euclidean_distance(v1, v2) -> float:
    """Fast numpy-based euclidean distance."""
    a = np.asarray(v1, dtype=np.float32)
    b = np.asarray(v2, dtype=np.float32)
    return float(np.linalg.norm(a - b))


class HNSWIndex:
    def __init__(self, m: int = 16, ef_construction: int = 100, m0: int = 32):
        self.M = m
        self.M0 = m0
        self.ef_construction = ef_construction
        self.level_mult = 1.0 / math.log(m) if m > 1 else 1.0

        self.vectors = {}   # node_id -> np.array
        self.doc_ids = {}   # node_id -> doc_id
        self.graphs = []    # list of dicts: layer -> {node_id: [neighbor_ids]}
        self.enter_point = None
        self.max_level = -1
        self.node_count = 0

    def get_random_level(self) -> int:
        f = random.random()
        if f == 0.0:
            f = 1e-10
        return int(-math.log(f) * self.level_mult)

    def search_layer(self, query: np.ndarray, enter_points: list, ef: int, layer: int) -> list:
        """
        Greedy beam search on a single HNSW layer.
        Returns list of (distance, node_id) sorted closest-first.
        """
        visited = set(enter_points)
        candidates = []  # min-heap: (dist, node)
        w = []           # max-heap of nearest: (-dist, node)

        for ep in enter_points:
            dist = euclidean_distance(query, self.vectors[ep])
            heapq.heappush(candidates, (dist, ep))
            heapq.heappush(w, (-dist, ep))

        while candidates:
            c_dist, c = heapq.heappop(candidates)
            # Furthest element in W
            f_dist = -w[0][0] if w else float('inf')
            if c_dist > f_dist:
                break

            for neighbor in self.graphs[layer].get(c, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    n_dist = euclidean_distance(query, self.vectors[neighbor])
                    f_dist = -w[0][0] if w else float('inf')
                    if n_dist < f_dist or len(w) < ef:
                        heapq.heappush(candidates, (n_dist, neighbor))
                        heapq.heappush(w, (-n_dist, neighbor))
                        if len(w) > ef:
                            heapq.heappop(w)

        # Convert max-heap to sorted list (closest first)
        result = [(-d, node) for d, node in w]
        result.sort(key=lambda x: x[0])
        return result

    def _select_neighbors(self, node_id: int, candidates: list, max_m: int, layer: int) -> list:
        """
        Simple neighbor selection: pick max_m closest candidates.
        """
        return [node for _, node in candidates[:max_m]]

    def add(self, vector: list, doc_id: str):
        node_id = self.node_count
        self.node_count += 1
        self.vectors[node_id] = np.asarray(vector, dtype=np.float32)
        self.doc_ids[node_id] = doc_id

        level = self.get_random_level()

        # Ensure enough layers exist
        while len(self.graphs) <= level:
            self.graphs.append({})

        # Init adjacency list for new node on all its layers
        for l in range(level + 1):
            self.graphs[l][node_id] = []

        # First node — just set as entry point
        if self.enter_point is None:
            self.enter_point = node_id
            self.max_level = level
            return

        ep = [self.enter_point]

        # Descend from top layer to level+1 with ef=1 (greedy)
        for l in range(self.max_level, level, -1):
            if l < len(self.graphs):
                res = self.search_layer(self.vectors[node_id], ep, ef=1, layer=l)
                if res:
                    ep = [res[0][1]]

        # Connect in layers from level down to 0
        for l in range(min(level, self.max_level), -1, -1):
            res = self.search_layer(self.vectors[node_id], ep, self.ef_construction, l)
            ep = [x[1] for x in res]

            max_m = self.M0 if l == 0 else self.M
            neighbors = self._select_neighbors(node_id, res, max_m, l)

            self.graphs[l][node_id] = neighbors

            # Bidirectional edges + pruning
            for n in neighbors:
                if n not in self.graphs[l]:
                    self.graphs[l][n] = []
                self.graphs[l][n].append(node_id)

                if len(self.graphs[l][n]) > max_m:
                    # Prune: keep max_m closest neighbors
                    n_vec = self.vectors[n]
                    n_neighbors = self.graphs[l][n]
                    n_dists = sorted(
                        (euclidean_distance(n_vec, self.vectors[nn]), nn)
                        for nn in n_neighbors
                    )
                    self.graphs[l][n] = [nn for _, nn in n_dists[:max_m]]

        if level > self.max_level:
            self.max_level = level
            self.enter_point = node_id

    def search(self, query_vector: list, k: int = 5, ef_search: int = 50) -> list:
        """
        Returns list of (distance, doc_id) sorted by distance ascending.
        """
        if self.enter_point is None:
            return []

        query = np.asarray(query_vector, dtype=np.float32)
        ep = [self.enter_point]
        ef = max(k, ef_search)

        # Greedy descent to layer 1
        for l in range(self.max_level, 0, -1):
            if l < len(self.graphs):
                res = self.search_layer(query, ep, ef=1, layer=l)
                if res:
                    ep = [res[0][1]]

        # Full search at layer 0
        res = self.search_layer(query, ep, ef=ef, layer=0)
        return [(dist, self.doc_ids[node_id]) for dist, node_id in res[:k]]
