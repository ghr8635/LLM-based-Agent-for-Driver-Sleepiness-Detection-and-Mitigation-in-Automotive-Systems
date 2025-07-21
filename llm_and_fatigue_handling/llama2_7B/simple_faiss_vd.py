# vector_db.py
import faiss
import numpy as np
import pickle
import os
import time
from collections import Counter

class PrefixVectorDB:
    def __init__(self, dim=9, index_path='index.faiss', metadata_path='metadata.pkl'):
        self.dim = dim
        self.index_path = index_path
        self.metadata_path = metadata_path

        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        else:
            self.index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))

        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            self.metadata = {}

        self.current_id = max(self.metadata.keys(), default=0) + 1

    def normalize_vector(self, vector):
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    def cosine_distance(self, v1, v2):
        v1 = self.normalize_vector(v1)
        v2 = self.normalize_vector(v2)
        return 1 - np.dot(v1, v2)

    def add_vector(self, feature_vector: np.ndarray, intervention: str, source: str):
        assert feature_vector.shape == (9,), f"Expected shape (9,), got {feature_vector.shape}"
        if intervention.lower() in ["none", "", "no action", "driver alert"]:
            return  # Skip irrelevant samples

        vector = self.normalize_vector(feature_vector)

        if len(self.metadata) > 0:
            D, I = self.index.search(np.expand_dims(vector, axis=0), k=min(5, len(self.metadata)))
            for idx, dist in zip(I[0], D[0]):
                if idx == -1:
                    continue
                neighbor = self.metadata.get(int(idx), {})
                try:
                    reconstructed = self.index.reconstruct(int(idx))
                    cosine_dist = self.cosine_distance(vector, reconstructed)
                except Exception:
                    continue
                if neighbor.get("intervention") == intervention and cosine_dist < 0.05:
                    return  # Skip duplicate

        if len(self.metadata) >= 2000:
            D, I = self.index.search(np.expand_dims(vector, axis=0), k=10)
            close_ids = [i for i, d in zip(I[0], D[0]) if i != -1 and d < 0.1]
            if close_ids:
                remove_id = close_ids[0]
            else:
                counter = Counter([v['intervention'] for v in self.metadata.values()])
                most_common = counter.most_common(1)[0][0]
                remove_id = next(k for k, v in self.metadata.items() if v['intervention'] == most_common)

            self.index.remove_ids(np.array([remove_id]))
            del self.metadata[remove_id]

        self.index.add_with_ids(np.expand_dims(vector, axis=0), np.array([self.current_id]))
        self.metadata[self.current_id] = {
            'intervention': intervention,
            'source': source,
            'timestamp': time.time(),
            'last_accessed': time.time()
        }
        self.current_id += 1

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)

    def search(self, query_vector: np.ndarray, k=5):
        assert query_vector.shape == (9,), f"Expected shape (9,), got {query_vector.shape}"
        query_vector = self.normalize_vector(query_vector)
        D, I = self.index.search(np.expand_dims(query_vector, axis=0), k)
        results = []
        for i, d in zip(I[0], D[0]):
            if i == -1:
                continue
            self.metadata[i]['last_accessed'] = time.time()
            results.append((int(i), self.metadata.get(int(i), {}), float(d)))
        return results


# === Inference-Time Helper Functions ===

def runtime_add(feature_vector: np.ndarray, intervention: str):
    db = PrefixVectorDB()
    db.add_vector(feature_vector, intervention, source='inference')
    db.save()

def retrieve_similar_vectors(feature_vector: np.ndarray, k=5):
    db = PrefixVectorDB()
    return db.search(feature_vector, k)