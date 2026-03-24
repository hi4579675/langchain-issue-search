"""
검색 구성별 성능 비교 — 가중치 기여도 분석 + LLM 리랭커 비교
실행: python -m eval.compare

비교 항목:
  A. 벡터 유사도만 (baseline)
  B. + 키워드 매칭
  C. + solution 가중치
  D. + 최신성 점수 (full hybrid)
  E. Gemini 2.5 Flash 리랭커
  F. Gemini 1.5 Flash 리랭커
  G. Groq llama-3.3-70b 리랭커  (GROQ_API_KEY 필요)

최적화:
  - 임베딩은 QA 쌍당 1회만 계산 후 전 구성에서 재사용
  - n_samples=200 으로 통계 신뢰도 향상 (이슈 3600+개 보유)
"""
import math
import datetime
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from typing import Callable
import psycopg2.extensions
import psycopg2.extras
from dotenv import load_dotenv
from google import genai
from tqdm import tqdm

from pipeline.vector_store import VectorStore
from pipeline.retriever import SearchResult
from eval.dataset import build_dataset, QAPair
from eval.metrics import EvalResult, _hit, _rr, _ndcg, _embed_query


def _search_configurable(
    conn: psycopg2.extensions.connection,
    query_vector: list[float],
    query_text: str,
    top_k: int = 10,
    use_keyword: bool = True,
    use_solution: bool = True,
    use_recency: bool = True,
) -> list[SearchResult]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT issue_number, content, chunk_type, is_solution, weight, issue_created_at,
                   1 - (embedding <=> %s::vector) AS vscore,
                   CASE WHEN content ILIKE %s THEN 1.3 ELSE 1.0 END AS kw
            FROM chunks ORDER BY embedding <=> %s::vector LIMIT %s
        """, (query_vector, f"%{query_text}%", query_vector, top_k * 3))
        rows = cur.fetchall()

    now = datetime.datetime.now(datetime.timezone.utc)
    results = []
    for r in rows:
        vscore = float(r["vscore"])
        kw     = float(r["kw"]) if use_keyword else 1.0
        sol    = 1.2 if (use_solution and r["is_solution"]) else 1.0

        if use_recency and r["issue_created_at"]:
            created = r["issue_created_at"]
            if created.tzinfo is None:
                created = created.replace(tzinfo=datetime.timezone.utc)
            age_days = (now - created).days
            recency = 0.8 + 0.2 * math.exp(-age_days / 365)
        else:
            recency = 1.0

        score = vscore * float(r["weight"]) * kw * sol * recency
        results.append(SearchResult(
            issue_number=r["issue_number"],
            content=r["content"],
            chunk_type=r["chunk_type"],
            is_solution=r["is_solution"],
            issue_url=f"https://github.com/langchain-ai/langchain/issues/{r['issue_number']}",
            score=score,
        ))

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k]


def _evaluate_config(conn, dataset, vecs: list[list[float]], **kwargs) -> EvalResult:
    """임베딩은 미리 계산된 vecs 재사용 — API 호출 없음"""
    h3 = h5 = rr = ndcg = 0.0
    for qa, vec in zip(dataset, vecs):
        results = _search_configurable(conn, vec, qa.query, **kwargs)
        ranked = [r.issue_number for r in results]
        h3   += _hit(ranked, qa.ground_truth, 3)
        h5   += _hit(ranked, qa.ground_truth, 5)
        rr   += _rr(ranked, qa.ground_truth)
        ndcg += _ndcg(ranked, qa.ground_truth, 5)
    n = len(dataset)
    return EvalResult(h3/n, h5/n, rr/n, ndcg/n, n)


def _get_rerank_candidates(conn, vec: list[float], query: str) -> list[SearchResult]:
    """벡터 검색 후보 30개 추출 → 이슈 단위 중복 제거 (vec 재사용)"""
    candidates = _search_configurable(
        conn, vec, query, top_k=30,
        use_keyword=False, use_solution=False, use_recency=False,
    )
    seen: set[int] = set()
    unique: list[SearchResult] = []
    for c in candidates:
        if c.issue_number not in seen:
            seen.add(c.issue_number)
            unique.append(c)
    return unique


def _build_rerank_prompt(query: str, candidates: list[SearchResult]) -> str:
    candidate_lines = "\n".join(
        f"Issue #{c.issue_number}: {c.content[:200].strip()}"
        for c in candidates[:15]
    )
    return (
        f"You are a LangChain GitHub issue expert.\n"
        f"Given the user query and candidate issues below, "
        f"return ONLY a comma-separated list of the top 5 most relevant issue numbers, "
        f"most relevant first. No explanation.\n\n"
        f"Query: {query}\n\n"
        f"Candidates:\n{candidate_lines}\n\n"
        f"Answer (e.g. 12345, 67890, 11111):"
    )


def _evaluate_llm_reranker(
    conn,
    dataset: list[QAPair],
    vecs: list[list[float]],
    generate_fn: Callable[[str], str],
    label: str,
) -> EvalResult:
    """
    LLM 리랭커 공통 평가 함수.

    방식:
      1. 미리 계산된 vec 재사용 → 벡터 검색 후보 30개 추출
      2. 이슈 단위 중복 제거 후 상위 15개 선택
      3. generate_fn(prompt) → issue_number 콤마 리스트 파싱
      4. Hit@k / MRR / NDCG@5 측정
    """
    h3 = h5 = rr = ndcg = 0.0

    for qa, vec in tqdm(zip(dataset, vecs), total=len(dataset), desc=f"  {label}", leave=False):
        candidates = _get_rerank_candidates(conn, vec, qa.query)
        prompt = _build_rerank_prompt(qa.query, candidates)
        try:
            raw = generate_fn(prompt)
            ranked = [int(x.strip()) for x in raw.split(",") if x.strip().isdigit()]
        except Exception as e:
            print(f"  {label} 호출 실패 (issue #{qa.ground_truth}): {e}")
            ranked = []

        h3   += _hit(ranked, qa.ground_truth, 3)
        h5   += _hit(ranked, qa.ground_truth, 5)
        rr   += _rr(ranked, qa.ground_truth)
        ndcg += _ndcg(ranked, qa.ground_truth, 5)

    n = len(dataset)
    return EvalResult(h3/n, h5/n, rr/n, ndcg/n, n)


def print_table(results: dict[str, EvalResult]):
    print("\n" + "=" * 66)
    print(f"{'구성':<34} {'Hit@3':>6} {'Hit@5':>6} {'MRR':>6} {'NDCG@5':>8}")
    print("-" * 66)
    for name, r in results.items():
        print(f"{name:<34} {r.hit_at_3:>6.3f} {r.hit_at_5:>6.3f} {r.mrr:>6.3f} {r.ndcg_at_5:>8.3f}")
    print("=" * 66)


if __name__ == "__main__":
    load_dotenv()
    store  = VectorStore(dsn=os.environ["DATABASE_URL"])
    gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    dataset = build_dataset(store.conn, n_samples=200)
    print(f"\n총 {len(dataset)}개 QA 쌍으로 비교 평가 시작...\n")

    # ── 임베딩 1회만 계산 → 전 구성 재사용 ────────────────────────
    print("임베딩 계산 중 (1회만)...")
    vecs = [
        _embed_query(qa.query, gemini)
        for qa in tqdm(dataset, desc="임베딩")
    ]
    print()

    # ── A~D: 검색 구성별 ablation ──────────────────────────────────
    configs = {
        "A. 벡터만 (baseline)":    dict(use_keyword=False, use_solution=False, use_recency=False),
        "B. +키워드 매칭":          dict(use_keyword=True,  use_solution=False, use_recency=False),
        "C. +solution 가중치":      dict(use_keyword=True,  use_solution=True,  use_recency=False),
        "D. +최신성 (full hybrid)": dict(use_keyword=True,  use_solution=True,  use_recency=True),
    }
    results = {}
    for name, cfg in configs.items():
        print(f"평가 중: {name}")
        results[name] = _evaluate_config(store.conn, dataset, vecs, **cfg)

    # ── E: Gemini 2.5 Flash 리랭커 ────────────────────────────────
    print("평가 중: E. Gemini 2.5 Flash 리랭커")
    results["E. Gemini 2.5 Flash 리랭커"] = _evaluate_llm_reranker(
        store.conn, dataset, vecs,
        generate_fn=lambda p: gemini.models.generate_content(
            model="gemini-2.5-flash", contents=p
        ).text.strip(),
        label="Gemini 2.5 Flash",
    )

    # ── F: Gemini 1.5 Flash 리랭커 ────────────────────────────────
    print("평가 중: F. Gemini 1.5 Flash 리랭커")
    results["F. Gemini 1.5 Flash 리랭커"] = _evaluate_llm_reranker(
        store.conn, dataset, vecs,
        generate_fn=lambda p: gemini.models.generate_content(
            model="gemini-2.0-flash", contents=p
        ).text.strip(),
        label="Gemini 1.5 Flash",
    )

    # ── G: Groq llama-3.3-70b 리랭커 (GROQ_API_KEY 필요) ─────────
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if groq_key:
        from openai import OpenAI as OpenAIClient
        groq_client = OpenAIClient(
            api_key=groq_key,
            base_url="https://api.groq.com/openai/v1",
        )
        print("평가 중: G. Groq llama-3.3-70b 리랭커")
        results["G. Groq llama-3.3-70b 리랭커"] = _evaluate_llm_reranker(
            store.conn, dataset, vecs,
            generate_fn=lambda p: groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": p}],
                max_tokens=40,
                temperature=0,
            ).choices[0].message.content.strip(),
            label="Groq llama-3.3-70b",
        )
    else:
        print("GROQ_API_KEY 미설정 — G. Groq 리랭커 스킵")

    print_table(results)
    store.close()
