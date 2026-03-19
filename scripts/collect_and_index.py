"""전체 파이프라인 실행: 수집 → 정제 → 청킹 → 임베딩 → 저장"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

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


def main():
    gh     = GitHubClient(token=os.environ["GITHUB_TOKEN"])
    gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    store  = VectorStore(dsn=os.environ["DATABASE_URL"])

    try:
        for raw_issue in tqdm(fetch_issues(gh, max_pages=20), desc="이슈 처리"):
            comments = fetch_comments(gh, raw_issue.number)
            cleaned  = make_cleaned_issue(raw_issue, comments)
            if not cleaned:
                continue

            for text, is_sol in [(cleaned.question, False), (cleaned.solution, True)]:
                chunks  = split_into_chunks(text, cleaned.issue_number)
                vectors = embed_chunks(chunks, gemini)
                store.upsert(chunks, vectors, is_solution=is_sol)

            print(f"Issue #{cleaned.issue_number} 완료")
    finally:
        store.close()


if __name__ == "__main__":
    main()
