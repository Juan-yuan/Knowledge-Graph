import csv
import json
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote, urljoin

import requests
from bs4 import BeautifulSoup
from langchain_core.messages import HumanMessage, SystemMessage
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import Config
from common.llm import my_llm
from common.path_utils import get_file_path

_conf = Config()
BASE_URL = _conf.HERBS_BASE_URL
LIST_URL = _conf.HERBS_LIST_URL
OUTPUT_CSV = get_file_path("clawler/medicinal_herbs_list.csv")
HEADERS = {"User-Agent": _conf.CRAWLER_USER_AGENT}
BATCH_SIZE = int(_conf.require("CRAWLER_BATCH_SIZE"))
MAX_RETRIES = int(_conf.require("CRAWLER_MAX_RETRIES"))
TRANSLATE_WORKERS = int(_conf.require("CRAWLER_TRANSLATE_WORKERS"))
CONTENT_SELECTOR = _conf.require("CRAWLER_CONTENT_SELECTOR")
LIST_LINK_SELECTOR = _conf.require("CRAWLER_LIST_LINK_SELECTOR")
SYSTEM_PROMPT = _conf.require("CRAWLER_HERBS_LIST_PROMPT")


def fetch_herb_list(list_url=LIST_URL):
    """
    Fetch a list page (herbs by default) and extract item names and detail links.

    Args:
        list_url: url of the list page

    Returns:
        List of (chinese_name, detail_url) tuples, de-duplicated by url
    """
    response = requests.get(list_url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    response.encoding = "utf-8"
    soup = BeautifulSoup(response.text, "html.parser")

    content = soup.select_one(CONTENT_SELECTOR)
    herbs, seen = [], set()
    for a in content.select(LIST_LINK_SELECTOR):
        name = a.get_text(strip=True)
        # Percent-encode the path so the stored link is pure ASCII
        url = urljoin(BASE_URL, quote(a["href"]))
        if name and url not in seen:
            seen.add(url)
            herbs.append((name, url))
    return herbs


def to_ascii(text):
    """Fold LLM output (e.g. non-breaking hyphens, accented pinyin) into plain ASCII."""
    text = unicodedata.normalize("NFKD", text).replace("\u2011", "-").replace("\u2010", "-")
    return text.encode("ascii", "ignore").decode().strip()


def translate_batch(names, system_prompt=SYSTEM_PROMPT):
    """
    Translate a batch of Chinese names (herbs by default) into English with the LLM.

    Args:
        names: Chinese names
        system_prompt: translation instructions for the LLM

    Returns:
        English names in the same order as the input
    """
    messages = [SystemMessage(content=system_prompt),
                HumanMessage(content=json.dumps(names, ensure_ascii=False))]
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            content = my_llm.invoke(messages).content.strip()
            content = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            result = json.loads(content)
            if isinstance(result, list) and len(result) == len(names):
                return [to_ascii(str(r)) for r in result]
            print(f"Length mismatch ({len(result)} vs {len(names)}), retry {attempt}")
        except Exception as e:
            print(f"Translate failed: {e}, retry {attempt}")
    # Fall back to translating one by one so a single bad batch doesn't stop the run
    return [translate_batch([n], system_prompt)[0] for n in names] if len(names) > 1 else names


def main():
    herbs = fetch_herb_list()
    print(f"Found {len(herbs)} herbs")

    names = [name for name, _ in herbs]
    batches = [names[i:i + BATCH_SIZE] for i in range(0, len(names), BATCH_SIZE)]
    # Translate batches concurrently; map() keeps the original order
    with ThreadPoolExecutor(max_workers=TRANSLATE_WORKERS) as executor:
        results = list(tqdm(executor.map(translate_batch, batches), total=len(batches), desc="Translating"))
    english_names = [name for batch in results for name in batch]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Herb Name", "Detail URL"])
        for english_name, (_, url) in zip(english_names, herbs):
            writer.writerow([english_name, url])
    print(f"Saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
