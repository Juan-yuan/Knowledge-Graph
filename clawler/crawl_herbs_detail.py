import argparse
import csv
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from langchain_core.messages import HumanMessage, SystemMessage
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clawler.crawl_herbs_list import HEADERS, OUTPUT_CSV
from common.config import Config
from common.llm import my_llm
from common.path_utils import get_file_path

_conf = Config()
OUTPUT_DIR = get_file_path("clawler/medicinal_herbs")
MAX_WORKERS = int(_conf.require("CRAWLER_DETAIL_WORKERS"))
DEFAULT_LIMIT = 100
MAX_RETRIES = int(_conf.require("CRAWLER_MAX_RETRIES"))
REQUEST_DELAY = (
    float(_conf.require("CRAWLER_REQUEST_DELAY_MIN")),
    float(_conf.require("CRAWLER_REQUEST_DELAY_MAX")),
)
CHUNK_SIZE = int(_conf.require("CRAWLER_CHUNK_SIZE"))
CONTENT_SELECTOR = _conf.require("CRAWLER_CONTENT_SELECTOR")
DETAIL_REMOVE_SELECTORS = _conf.require("CRAWLER_DETAIL_REMOVE_SELECTORS")
DETAIL_UNWRAP_SELECTORS = _conf.require("CRAWLER_DETAIL_UNWRAP_SELECTORS")
CJK_PATTERN = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")
SYSTEM_PROMPT = _conf.require("CRAWLER_HERBS_DETAIL_PROMPT")

_local = threading.local()


def get_session():
    """One requests.Session per thread (Session is not thread-safe)."""
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
        _local.session.headers.update(HEADERS)
    return _local.session


def safe_filename(name):
    """Replace characters that are not allowed in file names."""
    return re.sub(r'[\\/:*?"<>|]+', "-", name).strip(" .-")


def build_tasks(rows, name_column, output_dir):
    """
    Map each csv row to an output file path. Names that collide get a numeric suffix;
    the csv order is fixed, so the mapping is stable across runs (needed for resuming).

    Returns:
        List of (name, url, file_path)
    """
    tasks, used = [], set()
    for row in rows:
        base = safe_filename(row[name_column])
        filename, n = base, 2
        while filename.lower() in used:
            filename, n = f"{base} ({n})", n + 1
        used.add(filename.lower())
        tasks.append((row[name_column], row["Detail URL"], os.path.join(output_dir, f"{filename}.txt")))
    return tasks


def fetch_detail(url):
    """
    Fetch a detail page (herb or formula) and extract its text content.

    Returns:
        Lines of Chinese text, without the publish time or blank lines
    """
    for attempt in range(1, MAX_RETRIES + 1):
        time.sleep(random.uniform(*REQUEST_DELAY))
        try:
            response = get_session().get(url, timeout=30)
            if response.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {response.status_code}")
            response.raise_for_status()
            break
        except requests.RequestException:
            if attempt == MAX_RETRIES:
                raise
            # Back off longer each time in case we tripped rate limiting
            time.sleep(2 ** attempt * 2)
    response.encoding = "utf-8"
    soup = BeautifulSoup(response.text, "html.parser")

    content = soup.select_one(CONTENT_SELECTOR)
    # Drop images, ads and scripts; the publish time lives in the header, which we skip
    for tag in content.select(DETAIL_REMOVE_SELECTORS):
        tag.decompose()
    # Unwrap inline tags so e.g. a linked ingredient stays on the same line as its dosage
    for tag in content.select(DETAIL_UNWRAP_SELECTORS):
        tag.unwrap()
    content.smooth()

    title = soup.select_one(_conf.require("CRAWLER_DETAIL_TITLE_SELECTOR"))
    lines = [title.get_text(strip=True)] if title else []
    for line in content.get_text("\n").splitlines():
        line = line.strip()
        if line:
            lines.append(line)
    return lines


def split_chunks(lines):
    """Group lines into chunks of at most CHUNK_SIZE characters."""
    chunks, current, size = [], [], 0
    for line in lines:
        if current and size + len(line) > CHUNK_SIZE:
            chunks.append(current)
            current, size = [], 0
        current.append(line)
        size += len(line)
    if current:
        chunks.append(current)
    return chunks


def translate_text(lines, system_prompt=SYSTEM_PROMPT):
    """
    Translate lines of Chinese text into English with the LLM, chunk by chunk.

    Returns:
        Translated, non-blank English lines
    """
    result = []
    for chunk in split_chunks(lines):
        messages = [SystemMessage(content=system_prompt), HumanMessage(content="\n".join(chunk))]
        for attempt in range(1, MAX_RETRIES + 1):
            text = my_llm.invoke(messages).content
            # Retry if Chinese slipped through, otherwise accept the last attempt
            if not CJK_PATTERN.search(text) or attempt == MAX_RETRIES:
                break
        result.extend(line.strip() for line in text.splitlines() if line.strip())
    return result


def crawl_one(task, system_prompt=SYSTEM_PROMPT):
    """Crawl, translate and save one item. Writes to a temp file first so an
    interrupted run never leaves a half-written file that would be skipped later."""
    name, url, file_path = task
    english_lines = translate_text(fetch_detail(url), system_prompt)
    tmp_path = file_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write("\n".join([name, f"Source: {url}"] + english_lines[1:]) + "\n")
    os.replace(tmp_path, file_path)


def run(csv_path, name_column, output_dir, system_prompt, limit, workers, unit):
    """
    Crawl the detail pages listed in a csv, translate them and save one txt per row.
    Rows that already have a non-empty file are skipped, so the run can be resumed.

    Args:
        csv_path: list csv with a name column and a "Detail URL" column
        name_column: header of the English name column
        output_dir: directory for the txt files
        system_prompt: translation instructions for the LLM
        limit: only crawl the first N rows (0 or None = all)
        workers: number of threads
        unit: label for the progress bar
    """
    os.makedirs(output_dir, exist_ok=True)
    with open(csv_path, encoding="utf-8") as f:
        tasks = build_tasks(list(csv.DictReader(f)), name_column, output_dir)
    if limit:
        tasks = tasks[:limit]

    pending = [t for t in tasks if not (os.path.exists(t[2]) and os.path.getsize(t[2]) > 0)]
    print(f"Total {len(tasks)}, done {len(tasks) - len(pending)}, pending {len(pending)}")

    failed = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(crawl_one, t, system_prompt): t for t in pending}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Crawling", unit=unit):
            try:
                future.result()
            except Exception as e:
                name, url, _ = futures[future]
                failed.append(name)
                tqdm.write(f"Failed: {name} ({url}): {e}")

    print(f"Finished. Failed {len(failed)}; rerun the script to retry them.")


def main():
    parser = argparse.ArgumentParser(description="Crawl herb detail pages and save them in English.")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="crawl the first N herbs (0 = all)")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args()
    run(OUTPUT_CSV, "Herb Name", OUTPUT_DIR, SYSTEM_PROMPT, args.limit, args.workers, "herb")


if __name__ == "__main__":
    main()
