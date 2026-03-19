"""Gemini text-embedding-004 배치 임베딩"""
from google import genai
from .chunker import Chunk

EMBED_MODEL = "text-embedding-004"  # 출력 차원: 768
BATCH_SIZE = 100


def embed_chunks(chunks: list[Chunk], client: genai.Client) -> list[list[float]]:
    """청크 리스트를 배치로 임베딩해 벡터 리스트 반환"""
    texts = [c.content for c in chunks]
    vectors = []
    for i in range(0, len(texts), BATCH_SIZE):
        result = client.models.embed_content(
            model=EMBED_MODEL,
            contents=texts[i:i + BATCH_SIZE],
        )
        vectors.extend([e.values for e in result.embeddings])
    return vectors
