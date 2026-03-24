"""전체 파이프라인 실행: 수집 → 정제 → 청킹 → 임베딩 → 저장"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from multiprocessing import Process, Queue
from dotenv import load_dotenv
from google import genai
from tqdm import tqdm

from collector.github_client import GitHubClient
from collector.issue_fetcher import fetch_issues, fetch_comments
from collector.cleaner import make_cleaned_issue
from pipeline.chunker import split_into_chunks
from pipeline.embedder import embed_chunks
from pipeline.vector_store import VectorStore

load_dotenv()

ISSUE_TIMEOUT = 180  # 이슈당 최대 처리 시간 (초)


def process_issue_worker(raw_issue, gh_token: str, gemini_key: str, dsn: str, result_q: Queue):
    """
    독립 프로세스에서 실행되는 워커.
    Process.kill()로 OS 레벨 강제 종료 가능 (Thread와 달리 실제로 멈춤).
    프로세스 간 객체 공유 불가 → 토큰/DSN 문자열만 받아 내부에서 클라이언트 재생성.
    """
    try:
        gh     = GitHubClient(token=gh_token)
        gemini = genai.Client(api_key=gemini_key)
        store  = VectorStore(dsn=dsn)

        comments = fetch_comments(gh, raw_issue.number)
        cleaned  = make_cleaned_issue(raw_issue, comments)
        if not cleaned:
            result_q.put(False)
            return

        for text, is_sol in [(cleaned.question, False), (cleaned.solution, True)]:
            chunks  = split_into_chunks(text, cleaned.issue_number)
            vectors = embed_chunks(chunks, gemini)
            store.upsert(chunks, vectors, is_solution=is_sol,
                         issue_created_at=cleaned.created_at)
        store.close()
        result_q.put(True)
    except Exception as e:
        result_q.put(e)


def main():
    gh_token   = os.environ["GITHUB_TOKEN"]
    gemini_key = os.environ["GEMINI_API_KEY"]
    dsn        = os.environ["DATABASE_URL"]

    gh    = GitHubClient(token=gh_token)
    store = VectorStore(dsn=dsn)

    skipped = 0

    try:
        indexed = store.get_indexed_issue_numbers()
        print(f"이미 인덱싱된 이슈: {len(indexed)}개 — 스킵")

        for raw_issue in tqdm(fetch_issues(gh, max_pages=100), desc="이슈 처리"):
            if raw_issue.number in indexed:
                continue

            print(f"Issue #{raw_issue.number} 처리 시작...")
            result_q = Queue()
            p = Process(
                target=process_issue_worker,
                args=(raw_issue, gh_token, gemini_key, dsn, result_q),
            )
            p.start()
            p.join(timeout=ISSUE_TIMEOUT)

            if p.is_alive():
                # 타임아웃: OS 레벨로 프로세스 강제 종료 (실제로 멈춤)
                p.kill()
                p.join()
                skipped += 1
                print(f"Issue #{raw_issue.number} 타임아웃({ISSUE_TIMEOUT}s) — 강제 종료 (누적 {skipped}개)")
            else:
                result = result_q.get() if not result_q.empty() else None
                if isinstance(result, Exception):
                    skipped += 1
                    print(f"Issue #{raw_issue.number} 오류 — 스킵 (누적 {skipped}개): {result}")
                elif result:
                    print(f"Issue #{raw_issue.number} 완료")
                else:
                    print(f"Issue #{raw_issue.number} 스킵 (정제 결과 없음)")
    finally:
        store.close()
        print(f"\n완료. 타임아웃/오류로 스킵된 이슈: {skipped}개")


if __name__ == "__main__":
    main()
