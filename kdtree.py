import math
import heapq

def euclidean_distance(v1, v2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))

class KDNode:
    def __init__(self, vector, doc_id, depth, left=None, right=None, parent=None):
        self.vector = vector
        self.doc_id = doc_id
        self.depth = depth
        self.left = left
        self.right = right

class KDTreeIndex:
    def __init__(self):
        self.root = None
        self.k = None

    def _build(self, points, depth=0):
        if not points:
            return None
        
        if self.k is None:
            self.k = len(points[0][0])
            
        axis = depth % self.k
        
        # Sort point indices and choose median as pivot element
        points.sort(key=lambda x: x[0][axis])
        median = len(points) // 2
        
        # Create node and construct subtrees
        node = KDNode(
            vector=points[median][0],
            doc_id=points[median][1],
            depth=depth,
            left=self._build(points[:median], depth + 1),
            right=self._build(points[median + 1:], depth + 1)
        )
        return node

    def add_all(self, vectors, doc_ids):
        points = list(zip(vectors, doc_ids))
        self.root = self._build(points)
        
    def add(self, vector, doc_id):
        # Note: True KD-Tree needs rebalancing. 
        # For simplicity, we just add to the leaf.
        if self.root is None:
            self.root = KDNode(vector, doc_id, 0)
            self.k = len(vector)
            return

        curr = self.root
        depth = 0
        while True:
            axis = depth % self.k
            if vector[axis] < curr.vector[axis]:
                if curr.left is None:
                    curr.left = KDNode(vector, doc_id, depth + 1)
                    break
                curr = curr.left
            else:
                if curr.right is None:
                    curr.right = KDNode(vector, doc_id, depth + 1)
                    break
                curr = curr.right
            depth += 1

    def search(self, query_vector, k=5):
        if self.root is None:
            return []
            
        # Priority queue for neighbors (max heap of size k)
        # We store (-distance, doc_id) because heapq is a min-heap
        heap = []
        
        def nearest(node):
            if node is None:
                return
            
            # Distance from query to current node
            dist = euclidean_distance(query_vector, node.vector)
            
            # Maintain heap of size k
            if len(heap) < k:
                heapq.heappush(heap, (-dist, node.doc_id))
            elif dist < -heap[0][0]:
                heapq.heapreplace(heap, (-dist, node.doc_id))
                
            axis = node.depth % self.k
            diff = query_vector[axis] - node.vector[axis]
            
            # Determine which side to search first
            if diff < 0:
                first, second = node.left, node.right
            else:
                first, second = node.right, node.left
                
            nearest(first)
            
            # Check if we need to search the other side
            # Only search if the distance to the splitting plane is less than the worst distance in our heap
            if len(heap) < k or abs(diff) < -heap[0][0]:
                nearest(second)
                
        nearest(self.root)
        
        # Format results: (distance, doc_id) and sort by distance
        results = [(-dist, doc_id) for dist, doc_id in heap]
        results.sort(key=lambda x: x[0])
        return results
