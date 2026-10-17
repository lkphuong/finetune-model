"""Dataset loading for Task 1: resolves code files and exam metadata per sample."""
import json
import random
from dataclasses import dataclass
from pathlib import Path

from config import GRADING_FILE


@dataclass(frozen=True)
class Record:
    sample_id: str
    sample_input: dict
    output: dict | None
    exam: dict
    code: str


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_records(root: Path, with_labels: bool = True) -> list[Record]:
    """Load every Task 1 sample from a dataset root directory."""
    exams = {e["exam_id"]: e for e in _read_json(root / "exams.json")["exams"]}
    records = []
    for sample in _read_json(root / GRADING_FILE)["samples"]:
        inp = sample["input"]
        output = sample.get("output") if with_labels else None
        if with_labels and output is None:
            raise ValueError(f"{sample['sample_id']} has no label in {root}")
        code = (root / inp["code_file"]).read_text(encoding="utf-8", errors="replace")
        records.append(Record(sample["sample_id"], inp, output, exams[inp["exam_id"]], code))
    return records


def split_records(records: list[Record], val_ratio: float, seed: int):
    ids = sorted(r.sample_id for r in records)
    random.Random(seed).shuffle(ids)
    val_ids = set(ids[: int(len(ids) * val_ratio)])
    return [r for r in records if r.sample_id not in val_ids], [r for r in records if r.sample_id in val_ids]
