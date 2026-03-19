"""하이브리드 검색: 벡터 유사도 + 키워드 + solution 가중치"""
from dataclasses import dataclass
import psycopg2.extensions
import psycopg2.extras

REPO = "langchain-ai/langchain"

@dataclass
class SearchResult:
    issue_number: int
    content: str
    chunk_type: str
    score: float
    is_solution: bool
    issue_url: str


class HybridRetriever:
    def __init__(self, conn: psycopg2.extensions.connection, top_k: int = 10):
        self.conn = conn
        self.top_k = top_k

    def search(self, query_vector: list[float], query_text: str) -> list[SearchResult]:
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT issue_number, content, chunk_type, is_solution, weight,
                       1 - (embedding <=> %s::vector) AS vscore,
                       CASE WHEN content ILIKE %s THEN 1.3 ELSE 1.0 END AS kw
                FROM chunks ORDER BY embedding <=> %s::vector LIMIT %s
            """, (query_vector, f"%{query_text}%", query_vector, self.top_k * 3))
            rows = cur.fetchall()

        results = [
            SearchResult(
                issue_number=r["issue_number"],
                content=r["content"],
                chunk_type=r["chunk_type"],
                is_solution=r["is_solution"],
                issue_url=f"https://github.com/{REPO}/issues/{r['issue_number']}",
                score=r["vscore"] * r["weight"] * r["kw"] * (1.2 if r["is_solution"] else 1.0),
            )
            for r in rows
        ]
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:self.top_k]
