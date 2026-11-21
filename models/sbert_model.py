"""Sentence-BERT semantic sentence embeddings."""
class SBERTEncoder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def similarity(self, a, b):
        va, vb = self.encode([a, b])
        return float(va @ vb)
