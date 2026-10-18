"""
Extract entities and relations from crawled formula txt files.

Reads:  clawler/herbal_formulas/*.txt
Writes: extract_information/herbal_formulas/<same-stem>.json

Features:
  - progress bar
  - resume (skips non-empty existing json files)
  - multithreaded extraction (default concurrency = 5)
  - limit=0 extracts all files
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.path_utils import get_file_path
from extract_information.extract_herbs import DEFAULT_LIMIT, MAX_WORKERS, run

INPUT_DIR = get_file_path("clawler/herbal_formulas")
OUTPUT_DIR = get_file_path("extract_information/herbal_formulas")


def main():
    parser = argparse.ArgumentParser(
        description="Extract a knowledge graph from crawled formula files."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="extract the first N formulas (0 = all)",
    )
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args()
    run(INPUT_DIR, OUTPUT_DIR, args.limit, args.workers)


if __name__ == "__main__":
    main()
