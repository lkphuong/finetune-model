"""Fully automatic batch inference with greedy decoding. Writes submission_task1.json."""
import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import read_manifest, select_device, select_dtype, set_global_seed
from config import SUBMISSION_NAME, TrainConfig
from data import Record, load_records
from policy import apply_grading_policy
from postprocess import parse_rubric, to_submission_output
from prompts import fit_prompt_ids


def load_model(adapter_dir: Path, manifest: dict, device, dtype):
    local_path = snapshot_download(manifest["base_model_id"], revision=manifest["base_model_revision"])
    tokenizer = AutoTokenizer.from_pretrained(local_path)
    base = AutoModelForCausalLM.from_pretrained(local_path, dtype=dtype).to(device)
    return tokenizer, PeftModel.from_pretrained(base, adapter_dir).eval()


@torch.no_grad()
def generate(tokenizer, model, record: Record, device, cfg: TrainConfig) -> str:
    prompt_ids = fit_prompt_ids(
        tokenizer, record.sample_input, record.exam, record.code, cfg.max_seq_len - cfg.max_new_tokens
    )
    input_ids = torch.tensor([prompt_ids], device=device)
    output = model.generate(
        input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
        max_new_tokens=cfg.max_new_tokens, do_sample=False, num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(output[0, len(prompt_ids):], skip_special_tokens=True)


def grade(tokenizer, model, record: Record, manifest: dict, device, cfg: TrainConfig) -> dict:
    text = generate(tokenizer, model, record, device, cfg)
    rubric = parse_rubric(text, manifest["fallback_rubric"])
    return to_submission_output(apply_grading_policy(rubric, record.exam, record.code))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_root", type=Path, required=True)
    parser.add_argument("--adapter_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--with_labels", action="store_true", help="test_root is labelled (validation run)")
    args = parser.parse_args()

    set_global_seed()
    manifest = read_manifest(args.adapter_dir)
    cfg = TrainConfig(**manifest["train_config"])
    device = select_device()
    tokenizer, model = load_model(args.adapter_dir, manifest, device, select_dtype(device))

    results = [
        {"sample_id": r.sample_id, "output": grade(tokenizer, model, r, manifest, device, cfg)}
        for r in load_records(args.test_root, with_labels=args.with_labels)
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / SUBMISSION_NAME).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(results)} predictions")


if __name__ == "__main__":
    main()
