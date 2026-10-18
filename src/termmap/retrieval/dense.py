"""SapBERT dense retrieval: CLS embeddings, L2-normalized, FAISS exact inner-product search."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from termmap.retrieval.fusion import Ranking, collapse


class Encoder:
    """Frozen SapBERT encoder (CLS token, L2-normalized)."""

    def __init__(self, name_or_path: str, device: str | None = None, max_len: int = 32):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        local = Path(name_or_path).exists()
        self.tok = AutoTokenizer.from_pretrained(name_or_path, local_files_only=local)
        self.model = AutoModel.from_pretrained(name_or_path, local_files_only=local).eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda":
            self.model = self.model.half()
        self.model.to(self.device)
        self.max_len = max_len

    def save(self, path: str | Path) -> None:
        self.tok.save_pretrained(path)
        self.model.float().save_pretrained(path)
        if self.device == "cuda":
            self.model.half()

    def encode(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        """Embed texts in length-sorted batches; returns float32 [n, 768], rows L2-normalized."""
        order = np.argsort([len(t) for t in texts], kind="stable")
        out = np.zeros((len(texts), self.model.config.hidden_size), dtype=np.float32)
        with self.torch.inference_mode():
            for s in range(0, len(texts), batch_size):
                idx = order[s:s + batch_size]
                enc = self.tok([texts[i] or "[UNK]" for i in idx], padding=True, truncation=True,
                               max_length=self.max_len, return_tensors="pt").to(self.device)
                cls = self.model(**enc).last_hidden_state[:, 0].float()
                cls = self.torch.nn.functional.normalize(cls, dim=-1)
                out[idx] = cls.cpu().numpy()
        return out


class DenseIndex:
    """FAISS IndexFlatIP (exact) over precomputed description embeddings."""

    def __init__(self, emb: np.ndarray, codes: list[str]):
        import faiss

        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(np.ascontiguousarray(emb, dtype=np.float32))
        self.codes = np.asarray(codes, dtype=object)

    def search(self, q_emb: np.ndarray, top_desc: int = 200, top_codes: int = 100) -> list[Ranking]:
        D, I = self.index.search(np.ascontiguousarray(q_emb, dtype=np.float32), top_desc)
        return [collapse(I[r][I[r] >= 0], D[r][I[r] >= 0], self.codes, top_codes) for r in range(len(I))]
