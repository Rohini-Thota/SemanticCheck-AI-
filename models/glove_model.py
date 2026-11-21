"""GloVe text-vector loader. Requires a downloaded GloVe .txt embedding file."""
import os
import numpy as np

class GloVeEncoder:
    def __init__(self, glove_path="embeddings/glove.6B.100d.txt"):
        self.glove_path = glove_path
        self.embeddings = {}
        self.dimension = 0

    def load(self, max_words=None):
        if not os.path.exists(self.glove_path):
            raise FileNotFoundError(
                f"GloVe file not found: {self.glove_path}. Download a GloVe text vector file and place it there."
            )
        with open(self.glove_path, encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                parts = line.rstrip().split()
                if len(parts) < 3:
                    continue
                word = parts[0]
                try:
                    vec = np.asarray(parts[1:], dtype=np.float32)
                except ValueError:
                    continue
                self.embeddings[word] = vec
                self.dimension = len(vec)
                if max_words and i + 1 >= max_words:
                    break
        if not self.embeddings:
            raise ValueError("No valid GloVe vectors were loaded.")
        return self

    def encode(self, text):
        from models.preprocessing import preprocess
        if not self.embeddings:
            self.load()
        vectors = [self.embeddings[w] for w in preprocess(text, remove_stopwords=False, lemmatize=False)
                   if w in self.embeddings]
        if not vectors:
            return np.zeros(self.dimension, dtype=np.float32)
        return np.mean(vectors, axis=0)

    def similarity(self, a, b):
        from sklearn.metrics.pairwise import cosine_similarity
        va, vb = self.encode(a), self.encode(b)
        if not np.any(va) or not np.any(vb):
            return 0.0
        return float(cosine_similarity([va], [vb])[0][0])
