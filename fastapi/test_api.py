"""Test the running API with requests. Start the server first: python fastapi/main.py"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests

from common.config import Config

conf = Config()

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else conf.API_BASE_URL
USER_ID = conf.API_TEST_USER_ID

QUESTIONS = [
    "What are the effects of ginseng?",
    "What are the ingredients of Mahuang Decoction?",
    "How's the weather today?",
]


def test_health_check():
    response = requests.get(f"{BASE_URL}/", timeout=conf.API_HEALTH_TIMEOUT)
    print(f"[GET /] {response.status_code} {response.json()}")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_process(question: str):
    start = time.time()
    response = requests.post(
        f"{BASE_URL}/process",
        json={"input": question, "user_id": USER_ID},
        timeout=conf.API_PROCESS_TIMEOUT,
    )
    result = response.json()
    print(f"[POST /process] {response.status_code} ({time.time() - start:.1f}s)")
    print(f"  Question: {result['input']}")
    print(f"  Answer: {result['output']}")
    assert response.status_code == 200
    assert result["input"] == question
    assert result["output"]


def test_process_missing_field():
    # user_id is required, so FastAPI should reject the request with 422
    response = requests.post(
        f"{BASE_URL}/process",
        json={"input": "What are the effects of ginseng?"},
        timeout=conf.API_MISSING_FIELD_TIMEOUT,
    )
    print(f"[POST /process missing user_id] {response.status_code}")
    assert response.status_code == 422


if __name__ == "__main__":
    test_health_check()
    test_process_missing_field()
    for q in QUESTIONS:
        print("=" * 80)
        test_process(q)
    print("=" * 80)
    print("All tests passed")
