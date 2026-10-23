"""Test POST /process_stream. Start the server first: python fastapi/main.py"""
import json
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
    "How's the weather today?",
]


def read_stream(question: str) -> list:
    """POST the question and print each message as it arrives; return all parsed messages."""
    start = time.time()
    messages = []
    with requests.post(
        f"{BASE_URL}/process_stream",
        json={"input": question, "user_id": USER_ID},
        stream=True,
        timeout=conf.API_PROCESS_TIMEOUT,
    ) as response:
        print(f"[POST /process_stream] {response.status_code} question={question!r}")
        assert response.status_code == 200
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            message = json.loads(line)
            messages.append(message)
            elapsed = time.time() - start
            if message["type"] == "thinking":
                detail = f" {message['data']}" if message["status"] == "update" else ""
                print(f"  {elapsed:6.1f}s thinking {message['node']} [{message['status']}]{detail}")
            else:
                print(f"  {elapsed:6.1f}s {message['type']}: {message['output']}")
    return messages


def check_order(messages: list, question: str):
    """Every node must go start -> update -> end, and the stream must end with exactly one final."""
    assert messages, "stream was empty"
    *thinking, last = messages
    assert last["type"] == "final", f"last message should be final, got {last}"
    assert last["input"] == question
    assert last["output"]
    assert all(m["type"] == "thinking" for m in thinking), "final/error must only appear at the end"
    assert len(thinking) % 3 == 0, f"expected start/update/end triples, got {len(thinking)} thinking messages"
    for i in range(0, len(thinking), 3):
        triple = thinking[i:i + 3]
        nodes = {m["node"] for m in triple}
        statuses = [m["status"] for m in triple]
        assert len(nodes) == 1, f"triple mixes nodes: {triple}"
        assert statuses == ["start", "update", "end"], f"bad order for {nodes}: {statuses}"
        assert isinstance(triple[1]["data"], dict)


def test_process_stream_missing_field():
    response = requests.post(
        f"{BASE_URL}/process_stream",
        json={"input": "What are the effects of ginseng?"},
        timeout=conf.API_MISSING_FIELD_TIMEOUT,
    )
    print(f"[POST /process_stream missing user_id] {response.status_code}")
    assert response.status_code == 422


if __name__ == "__main__":
    test_process_stream_missing_field()
    for q in QUESTIONS:
        print("=" * 80)
        check_order(read_stream(q), q)
    print("=" * 80)
    print("All stream tests passed")
