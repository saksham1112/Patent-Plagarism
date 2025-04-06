import sys
import io
import pickle

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

try:
    with open("embeddings/trained_embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)
    print(f"✓ Success! Loaded {len(embeddings)} embeddings")
except FileNotFoundError:
    print("✗ Error: Embeddings file not found")
except Exception as e:
    print(f"✗ Error loading embeddings: {str(e)}")