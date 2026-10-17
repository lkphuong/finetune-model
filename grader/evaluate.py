"""Score submission_task1.json against labelled held-out samples."""
import argparse
import json
from pathlib import Path

from common import read_manifest
from config import SUBMISSION_NAME
from data import load_records
from metrics import grading_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--predictions_dir", type=Path, required=True)
    parser.add_argument("--adapter_dir", type=Path, required=True)
    args = parser.parse_args()

    held_out = set(read_manifest(args.adapter_dir)["held_out_sample_ids"])
    if not held_out:
        raise SystemExit("No held-out samples: retrain with --val_ratio > 0 to evaluate honestly.")
    preds = {p["sample_id"]: p["output"] for p in json.loads((args.predictions_dir / SUBMISSION_NAME).read_text(encoding="utf-8"))}
    records = [r for r in load_records(args.data_root) if r.sample_id in held_out]
    report = grading_metrics([r.output for r in records], [preds[r.sample_id] for r in records])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
