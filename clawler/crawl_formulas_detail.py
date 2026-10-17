import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clawler.crawl_formulas_list import OUTPUT_CSV
from clawler.crawl_herbs_detail import MAX_WORKERS, run
from common.path_utils import get_file_path

OUTPUT_DIR = get_file_path("clawler/herbal_formulas")
DEFAULT_LIMIT = 10

SYSTEM_PROMPT = (
    "You are a translator specializing in Traditional Chinese Medicine. "
    "Translate the following Chinese text about an herbal formula into English. "
    "Keep the line structure: one output line per input line, no blank lines. "
    "Use standard English names for formulas and herbs with Pinyin in parentheses, "
    "translate book titles with Pinyin in parentheses, keep dosages and units exact, "
    "and translate every Chinese character - the output must contain no Chinese. "
    "Return only the translation."
)


def main():
    parser = argparse.ArgumentParser(description="Crawl formula detail pages and save them in English.")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="crawl the first N formulas (0 = all)")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args()
    # Formula pages share the herb page layout, so the herb crawler is reused
    run(OUTPUT_CSV, "Formula Name", OUTPUT_DIR, SYSTEM_PROMPT, args.limit, args.workers, "formula")


if __name__ == "__main__":
    main()
