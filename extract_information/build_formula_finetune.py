"""
Build Alpaca fine-tuning JSON Lines from formula extractions.

Pairs:
  input  <- clawler/herbal_formulas/<name>.txt
  output <- extract_information/herbal_formulas/<name>.json  (extract_dict)

Writes one JSON object per line to extract_information/herb_fine_tuning.json.
Records already present (matched by input text) are skipped for resume support.

Instruction text is loaded from KG_FINETUNE_INSTRUCTION in common/.env.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.path_utils import get_file_path
from extract_information.build_herb_finetune import run

INPUT_DIR = get_file_path("clawler/herbal_formulas")
EXTRACT_DIR = get_file_path("extract_information/herbal_formulas")
OUTPUT_PATH = get_file_path("extract_information/herb_fine_tuning.json")


def main():
    parser = argparse.ArgumentParser(
        description="Build an Alpaca fine-tuning file from formula extractions."
    )
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args()
    run(INPUT_DIR, EXTRACT_DIR, args.output)


if __name__ == "__main__":
    main()
