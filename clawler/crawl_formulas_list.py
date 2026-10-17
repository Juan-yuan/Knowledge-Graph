import csv
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clawler.crawl_herbs_list import BATCH_SIZE, TRANSLATE_WORKERS, fetch_herb_list, translate_batch
from common.config import Config
from common.path_utils import get_file_path

LIST_URL = Config().FORMULAS_LIST_URL
OUTPUT_CSV = get_file_path("clawler/herbal_formulas_list.csv")

SYSTEM_PROMPT = (
    "You are a translator specializing in Traditional Chinese Medicine. "
    "Translate each Chinese herbal formula name into English. Use the standard English "
    "name when one exists, followed by the Pinyin in parentheses, e.g. "
    "\"Ephedra Decoction (Ma Huang Tang)\". Return ONLY a JSON array of strings, with the "
    "same length and order as the input."
)


def main():
    # The formula list page has the same layout as the herb list page
    formulas = fetch_herb_list(LIST_URL)
    print(f"Found {len(formulas)} formulas")

    names = [name for name, _ in formulas]
    batches = [names[i:i + BATCH_SIZE] for i in range(0, len(names), BATCH_SIZE)]
    # Translate batches concurrently; map() keeps the original order
    translate = partial(translate_batch, system_prompt=SYSTEM_PROMPT)
    with ThreadPoolExecutor(max_workers=TRANSLATE_WORKERS) as executor:
        results = list(tqdm(executor.map(translate, batches), total=len(batches), desc="Translating"))
    english_names = [name for batch in results for name in batch]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Formula Name", "Detail URL"])
        for english_name, (_, url) in zip(english_names, formulas):
            writer.writerow([english_name, url])
    print(f"Saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
