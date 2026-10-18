import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.llm import my_llm
from common.path_utils import get_file_path
from extract_information.extract_graph_data_utils import extract_tcm_knowledge

INPUT_DIR = get_file_path("clawler/medicinal_herbs")
OUTPUT_DIR = get_file_path("extract_information/medicinal_herbs")
MAX_WORKERS = 5
DEFAULT_LIMIT = 20
MAX_RETRIES = 3
CJK_PATTERN = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")
TRANSLATE_PROMPT = (
    "You are a translator for Traditional Chinese Medicine data. "
    "Translate every Chinese string value in the given JSON into English. "
    "Use pinyin in parentheses for herb and formula names, e.g. Bupleurum (Chai Hu). "
    "Keep the JSON structure, keys and enum values unchanged. "
    "Return only the JSON, without markdown fences or explanations."
)


def has_cjk(data):
    return bool(CJK_PATTERN.search(json.dumps(data, ensure_ascii=False)))


def parse_json(text):
    """Parse the LLM reply, tolerating ```json fences."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    return json.loads(text)


def translate_to_english(data):
    """Ask the LLM to translate any Chinese left in the extraction result."""
    messages = [
        SystemMessage(content=TRANSLATE_PROMPT),
        HumanMessage(content=json.dumps(data, ensure_ascii=False)),
    ]
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            translated = parse_json(my_llm.invoke(messages).content)
        except json.JSONDecodeError:
            if attempt == MAX_RETRIES:
                raise
            continue
        if not has_cjk(translated):
            return translated
        data = translated
        messages[1] = HumanMessage(content=json.dumps(data, ensure_ascii=False))
    raise RuntimeError("Chinese text still present after translation")


def extract_one(input_path, output_path):
    """Extract one file and save it. Writes to a temp file first so an
    interrupted run never leaves a half-written file that would be skipped later."""
    with open(input_path, encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        raise ValueError("empty file")

    result = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = extract_tcm_knowledge(text)
            break
        except Exception:
            if attempt == MAX_RETRIES:
                raise
    if has_cjk(result):
        result = translate_to_english(result)

    data = {"filename": os.path.basename(input_path), "extract_dict": result}
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, output_path)


def run(input_dir, output_dir, limit, workers):
    """
    Extract entities and relations from every txt in input_dir, one json per txt.
    Files that already have a non-empty json are skipped, so the run can be resumed.

    Args:
        input_dir: directory of crawled txt files
        output_dir: directory for the json files
        limit: only extract the first N files (0 or None = all)
        workers: number of threads
    """
    os.makedirs(output_dir, exist_ok=True)
    filenames = sorted(f for f in os.listdir(input_dir) if f.endswith(".txt"))
    if limit:
        filenames = filenames[:limit]

    tasks = [
        (os.path.join(input_dir, name), os.path.join(output_dir, Path(name).stem + ".json"))
        for name in filenames
    ]
    pending = [t for t in tasks if not (os.path.exists(t[1]) and os.path.getsize(t[1]) > 0)]
    print(f"Total {len(tasks)}, done {len(tasks) - len(pending)}, pending {len(pending)}")

    failed = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(extract_one, *t): t for t in pending}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting", unit="herb"):
            try:
                future.result()
            except Exception as e:
                name = os.path.basename(futures[future][0])
                failed.append(name)
                tqdm.write(f"Failed: {name}: {e}")

    print(f"Finished. Failed {len(failed)}; rerun the script to retry them.")


def main():
    parser = argparse.ArgumentParser(description="Extract a knowledge graph from crawled herb files.")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="extract the first N herbs (0 = all)")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args()
    run(INPUT_DIR, OUTPUT_DIR, args.limit, args.workers)


if __name__ == "__main__":
    main()
