"""Optional BERT mean-pooled sentence embeddings. Downloads bert-base-uncased on first use."""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

class BERTEncoder:
    def __init__(self, model_name="bert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        batch = self.tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            out = self.model(**batch)
        mask = batch["attention_mask"].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return F.normalize(pooled, p=2, dim=1).cpu().numpy()

    def similarity(self, a, b):
        va, vb = self.encode([a, b])
        return float((va * vb).sum())
