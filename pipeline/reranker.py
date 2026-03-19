"""Cross-encoder 리랭커"""
from sentence_transformers import CrossEncoder
from .retriever import SearchResult


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[SearchResult], top_n: int = 5) -> list[SearchResult]:
        if not candidates:
            return []

        scores = self.model.predict([(query, c.content) for c in candidates])
        reranked = sorted(
            [SearchResult(**{**c.__dict__, "score": float(s)}) for c, s in zip(candidates, scores)],
            key=lambda r: r.score,
            reverse=True,
        )
        return reranked[:top_n]
