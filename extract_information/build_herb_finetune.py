import argparse
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.path_utils import get_file_path

INPUT_DIR = get_file_path("clawler/medicinal_herbs")
EXTRACT_DIR = get_file_path("extract_information/medicinal_herbs")
OUTPUT_PATH = get_file_path("extract_information/herb_fine_tuning.json")
INSTRUCTION = (
    "Please extract the knowledge graph structure from the following "
    "Traditional Chinese Medicine text, including entities and relations."
)


def load_existing_records(output_path):
    """
    Load existing fine-tune records from a JSON array file.
    Also accepts legacy JSON Lines (.jsonl-style) content and normalizes it.
    """
    if not os.path.exists(output_path):
        return []

    raw = Path(output_path).read_text(encoding="utf-8").strip()
    if not raw:
        return []

    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        pass

    # Legacy JSON Lines: one object per line
    records = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def save_records(records, output_path):
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
        f.write("\n")


def build_record(txt_path, json_path):
    with open(txt_path, encoding="utf-8") as f:
        text = f.read().strip()
    with open(json_path, encoding="utf-8") as f:
        extract_dict = json.load(f)["extract_dict"]
    return {
        "instruction": INSTRUCTION,
        "input": text,
        "output": json.dumps(extract_dict, ensure_ascii=False),
    }


def run(input_dir, extract_dir, output_path):
    """
    Pair every extraction json with its source txt and write one Alpaca record
    each into a JSON array. Records already present (by input text) are skipped.
    """
    json_names = sorted(f for f in os.listdir(extract_dir) if f.endswith(".json"))
    records = load_existing_records(output_path)
    done_inputs = {record["input"] for record in records if isinstance(record, dict)}
    print(f"Found {len(json_names)} extraction file(s), {len(done_inputs)} record(s) already written")

    added, skipped, failed = 0, 0, []
    for name in tqdm(json_names, desc="Building", unit="herb"):
        txt_path = os.path.join(input_dir, Path(name).stem + ".txt")
        try:
            record = build_record(txt_path, os.path.join(extract_dir, name))
        except Exception as e:
            failed.append(name)
            tqdm.write(f"Failed: {name}: {e}")
            continue
        if not record["input"] or record["input"] in done_inputs:
            skipped += 1
            continue
        records.append(record)
        done_inputs.add(record["input"])
        added += 1
        # Persist incrementally so an interrupted run can resume
        save_records(records, output_path)

    # Ensure a valid array file exists even when nothing was added
    save_records(records, output_path)
    print(f"Finished. Added {added}, skipped {skipped}, failed {len(failed)}.")


def main():
    parser = argparse.ArgumentParser(description="Build an Alpaca fine-tuning file from herb extractions.")
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args()
    run(INPUT_DIR, EXTRACT_DIR, args.output)


if __name__ == "__main__":
    main()
