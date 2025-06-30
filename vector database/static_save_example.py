import numpy as np
from faiss_vd import static_dump, retrieve_similar_vectors

# dummy 4096-dimensional vectors
vecs = np.random.rand(3, 4096).astype('float32')
interventions = ["brake alert", "lane departure", "drowsiness alert"]

# Save vectors using static_dump helper
static_dump(vecs, interventions, source_file="example_static_dump")

print("✅ Vectors saved to index.faiss and metadata.pkl")

# Retrieve similar vectors (e.g., search vecs[1])
print("\n🔍 Searching for similar vectors to vecs[1]:")
results = retrieve_similar_vectors(vecs[1], k=3)

for idx, meta, distance in results:
    print(f"ID: {idx}, Distance: {distance:.4f}, Shape: {meta['vector'].shape}, Metadata: {meta},")