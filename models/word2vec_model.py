"""Word2Vec sentence vectors. Train on corpus text or load a local pretrained .model/.kv file."""
import os
import numpy as np

class Word2VecEncoder:
    def __init__(self, model_path="embeddings/word2vec.model", vector_size=100):
        self.model_path = model_path
        self.vector_size = vector_size
        self.model = None

    def train(self, documents, save=True):
        from gensim.models import Word2Vec
        from models.preprocessing import preprocess
        sentences = [preprocess(s, remove_stopwords=False, lemmatize=False)
                     for doc in documents for s in doc if preprocess(s, remove_stopwords=False, lemmatize=False)]
        if not sentences:
            raise ValueError("Not enough text to train Word2Vec.")
        self.model = Word2Vec(sentences=sentences, vector_size=self.vector_size,
                              window=5, min_count=1, workers=2, epochs=20)
        if save:
            os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)
            self.model.save(self.model_path)
        return self

    def load(self):
        from gensim.models import Word2Vec, KeyedVectors
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(self.model_path)
        try:
            self.model = Word2Vec.load(self.model_path).wv
        except Exception:
            self.model = KeyedVectors.load(self.model_path)
        return self

    def encode(self, text):
        from models.preprocessing import preprocess
        if self.model is None:
            self.load()
        tokens = preprocess(text, remove_stopwords=False, lemmatize=False)
        vectors = [self.model[w] for w in tokens if w in self.model]
        if not vectors:
            return np.zeros(self.model.vector_size, dtype=np.float32)
        return np.mean(vectors, axis=0)

    def similarity(self, a, b):
        from sklearn.metrics.pairwise import cosine_similarity
        va, vb = self.encode(a), self.encode(b)
        if not np.any(va) or not np.any(vb):
            return 0.0
        return float(cosine_similarity([va], [vb])[0][0])
