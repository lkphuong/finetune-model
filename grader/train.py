"""LoRA supervised fine-tuning of Qwen on rubric grading (Task 1). Fully automatic and seeded."""
import argparse
import json
import random
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import environment_versions, select_device, select_dtype, set_global_seed, write_manifest
from config import BASE_MODEL_ID, RUBRIC_MAX, SEED, TrainConfig
from data import Record, load_records, split_records
from prompts import build_target_text, fit_prompt_ids

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
IGNORE_INDEX = -100


def encode_record(tokenizer, record: Record, cfg: TrainConfig) -> dict:
    """Tokenize one example; loss is computed on the target tokens only."""
    target_ids = tokenizer(
        build_target_text(record.output) + "<|im_end|>", add_special_tokens=False
    )["input_ids"]
    prompt_ids = fit_prompt_ids(
        tokenizer, record.sample_input, record.exam, record.code,
        cfg.max_seq_len - len(target_ids),
    )
    return {
        "input_ids": torch.tensor(prompt_ids + target_ids),
        "labels": torch.tensor([IGNORE_INDEX] * len(prompt_ids) + target_ids),
    }


def mean_rubric(records: list[Record]) -> dict:
    """Rounded mean rubric used as a fallback when a generation cannot be parsed."""
    graded = [r.output["rubric"] for r in records]
    return {k: round(sum(g[k] for g in graded) / len(graded)) for k in RUBRIC_MAX}


def run_training(model, examples, cfg: TrainConfig, device) -> list[float]:
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.learning_rate)
    total_steps = max(1, cfg.epochs * len(examples) // cfg.grad_accum_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: max(0.0, 1 - s / total_steps))
    order_rng = random.Random(SEED)
    losses, step_loss = [], 0.0
    model.train()
    for epoch in range(cfg.epochs):
        order = list(range(len(examples)))
        order_rng.shuffle(order)
        for i, idx in enumerate(order, 1):
            batch = {k: v.unsqueeze(0).to(device) for k, v in examples[idx].items()}
            loss = model(**batch).loss / cfg.grad_accum_steps
            loss.backward()
            step_loss += loss.item()
            if i % cfg.grad_accum_steps == 0 or i == len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                losses.append(step_loss)
                print(json.dumps({"epoch": epoch + 1, "step_loss": round(step_loss, 4)}))
                step_loss = 0.0
    return losses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_root", type=Path, required=True, help="dataset dir with labels")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--val_ratio", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    args = parser.parse_args()

    set_global_seed()
    cfg = TrainConfig(epochs=args.epochs)
    device, dtype = select_device(), select_dtype(select_device())

    # Resolve and pin the exact model commit so the run can be reproduced
    local_path = snapshot_download(BASE_MODEL_ID)
    revision = Path(local_path).name
    tokenizer = AutoTokenizer.from_pretrained(local_path)
    model = AutoModelForCausalLM.from_pretrained(local_path, dtype=dtype).to(device)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=LORA_TARGETS, task_type="CAUSAL_LM",
    ))

    records = load_records(args.train_root)
    train_records, val_records = split_records(records, args.val_ratio, SEED)
    examples = [encode_record(tokenizer, r, cfg) for r in train_records]
    losses = run_training(model, examples, cfg, device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    write_manifest(args.output_dir, {
        "base_model_id": BASE_MODEL_ID,
        "base_model_revision": revision,
        "seed": SEED,
        "train_config": cfg.__dict__,
        "n_train_samples": len(train_records),
        "held_out_sample_ids": sorted(r.sample_id for r in val_records),
        "fallback_rubric": mean_rubric(train_records),
        "final_step_loss": losses[-1] if losses else None,
        "versions": environment_versions(),
    })


if __name__ == "__main__":
    main()
