"""
F. Gemini 1.5 Flash 리랭커만 단독 평가
실행: python scripts/eval_f_only.py

임베딩은 eval_cache.pkl 에 저장 — 재실행 시 스킵
"""
import os
import sys
import pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
from google import genai
from tqdm import tqdm

from pipeline.vector_store import VectorStore
from eval.dataset import build_dataset
from eval.metrics import _embed_query
from eval.compare import _evaluate_llm_reranker, print_table

CACHE_FILE = os.path.join(os.path.dirname(__file__), "eval_cache.pkl")

load_dotenv()

store  = VectorStore(dsn=os.environ["DATABASE_URL"])
gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# 캐시가 있으면 dataset + vecs 재사용
if os.path.exists(CACHE_FILE):
    print("캐시 로드 중...")
    with open(CACHE_FILE, "rb") as f:
        dataset, vecs = pickle.load(f)
    print(f"총 {len(dataset)}개 QA 쌍 (캐시)\n")
else:
    dataset = build_dataset(store.conn, n_samples=200)
    print(f"총 {len(dataset)}개 QA 쌍\n")
    print("임베딩 계산 중...")
    vecs = [_embed_query(qa.query, gemini) for qa in tqdm(dataset, desc="임베딩")]
    with open(CACHE_FILE, "wb") as f:
        pickle.dump((dataset, vecs), f)
    print("캐시 저장 완료\n")

result = _evaluate_llm_reranker(
    store.conn, dataset, vecs,
    generate_fn=lambda p: gemini.models.generate_content(
        model="gemini-flash-lite-latest", contents=p
    ).text.strip(),
    label="Gemini 1.5 Flash",
)

print_table({"F. Gemini 1.5 Flash 리랭커": result})
store.close()
