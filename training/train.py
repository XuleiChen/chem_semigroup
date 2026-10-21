"""
Step 5 — LoRA SFT with HuggingFace Trainer

Usage:
    python train.py --config config.yaml            # full training
    python train.py --config config.yaml --debug    # 10 reactions, quick run
    python train.py --config config.yaml --overfit  # overfit check on 8 samples
"""

import argparse
import json
import logging
import os
import random
from pathlib import Path

import torch
import yaml
from torch.utils.data import Dataset
from transformers import (
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from training.model_utils import load_tokenizer_and_model
from data_pipeline.prompts import build_sample

logging.basicConfig(format="%(levelname)s | %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)


# ── dataset ──────────────────────────────────────────────────────────────────

class SmilesDataset(Dataset):
    """Reads pre-tokenized jsonl produced by prompts.py."""

    def __init__(self, jsonl_path: Path):
        self.samples = [
            json.loads(l)
            for l in jsonl_path.read_text().splitlines()
            if l.strip()
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "input_ids": s["input_ids"],
            "labels":    s["labels"],
        }


class RawSmilesDataset(Dataset):
    """
    Builds tokenized samples on-the-fly from raw jsonl samples.
    Used for the overfit check where we want a tiny hand-picked slice.
    """

    def __init__(self, samples: list[dict], tokenizer, cfg: dict):
        self.tokenized = [build_sample(tokenizer, cfg, s) for s in samples]

    def __len__(self):
        return len(self.tokenized)

    def __getitem__(self, idx):
        t = self.tokenized[idx]
        return {"input_ids": t["input_ids"], "labels": t["labels"]}


# ── LoRA setup ───────────────────────────────────────────────────────────────

def attach_lora(model, cfg: dict):
    from peft import LoraConfig, TaskType, get_peft_model

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["lora_target_modules"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)

    trainable, total = 0, 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    log.info(
        "LoRA attached | trainable params: %d / %d (%.2f%%)",
        trainable, total, 100 * trainable / total,
    )
    return model


# ── overfit check ─────────────────────────────────────────────────────────────

def overfit_check(model, tokenizer, cfg: dict, device: str) -> None:
    """
    Train on 8 samples until loss ~ 0, then greedily decode and verify
    each target is reproduced exactly.  Catches tokenization / label bugs.
    """
    log.info("═" * 60)
    log.info("OVERFIT CHECK — 8 samples, up to 200 steps")

    data_dir = Path(cfg["data_dir"])
    raw_samples = [
        json.loads(l)
        for l in (data_dir / "train_samples.jsonl").read_text().splitlines()
        if l.strip()
    ][:8]

    ds = RawSmilesDataset(raw_samples, tokenizer, cfg)

    # verify label assertion for each sample
    for item in ds:
        non_masked = [t for t in item["labels"] if t != -100]
        assert len(non_masked) >= 2, "labels look wrong: too few non-masked tokens"

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
    )

    # tiny training run
    args = TrainingArguments(
        output_dir=str(Path(cfg["output_dir"]) / "overfit_check"),
        num_train_epochs=200,
        per_device_train_batch_size=8,
        learning_rate=1e-3,
        lr_scheduler_type="constant",
        logging_steps=20,
        save_strategy="no",
        bf16=cfg.get("bf16", True) and torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=False,
        report_to="none",
        seed=cfg["seed"],
        dataloader_pin_memory=False,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=collator,
    )
    trainer.train()

    # generation check
    log.info("Overfit check — generation verification:")
    model.eval()
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos_token_id = [im_end_id, tokenizer.eos_token_id]

    n_correct = 0
    for raw_s in raw_samples:
        from data_pipeline.prompts import build_prompt_text
        prompt_str = tokenizer.apply_chat_template(
            [{"role": "user", "content": build_prompt_text(cfg, raw_s["source"], raw_s["k"])}],
            add_generation_prompt=True,
            tokenize=False,
        )
        input_ids = tokenizer.encode(prompt_str, add_special_tokens=False, return_tensors="pt")
        input_ids = input_ids.to(device)

        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=256,
                do_sample=False,
                eos_token_id=eos_token_id,
            )
        new_tokens = out[0][input_ids.shape[1]:]
        pred = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        match = pred == raw_s["target"]
        n_correct += int(match)
        log.info("  target=%r  pred=%r  match=%s", raw_s["target"][:60], pred[:60], match)

    log.info("Overfit check: %d / %d targets reproduced exactly", n_correct, len(raw_samples))
    if n_correct < len(raw_samples):
        log.warning("Some targets not reproduced — check tokenization/labels/generation.")
    log.info("═" * 60)


# ── main training ─────────────────────────────────────────────────────────────

def main(cfg: dict, debug: bool = False, overfit: bool = False) -> None:
    # reproducibility
    seed = cfg["seed"]
    random.seed(seed)
    torch.manual_seed(seed)

    output_dir = Path(cfg["output_dir"])
    data_dir   = Path(cfg["data_dir"])

    # ── load model + tokenizer ───────────────────────────────────────────────
    tokenizer, model = load_tokenizer_and_model(cfg, load_model=True)

    assert tokenizer.pad_token is not None, "tokenizer.pad_token is None"
    # training uses right-padding; infer.py will switch to left
    tokenizer.padding_side = "right"

    model.enable_input_require_grads()   # needed before gradient checkpointing + LoRA
    model.gradient_checkpointing_enable()
    model = attach_lora(model, cfg)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    log.info("Device: %s", device)

    # ── overfit check ────────────────────────────────────────────────────────
    if overfit:
        overfit_check(model, tokenizer, cfg, device)
        return

    # ── datasets ─────────────────────────────────────────────────────────────
    if debug:
        # rebuild from raw samples (small enough to not need pre-tokenized files)
        n = cfg.get("debug_n_reactions", 10)
        raw_train = [
            json.loads(l)
            for l in (data_dir / "train_samples.jsonl").read_text().splitlines()
            if l.strip()
        ][:n * 3]   # rough cap
        raw_val = [
            json.loads(l)
            for l in (data_dir / "val_samples.jsonl").read_text().splitlines()
            if l.strip()
        ][:n]
        train_ds = RawSmilesDataset(raw_train, tokenizer, cfg)
        val_ds   = RawSmilesDataset(raw_val,   tokenizer, cfg)
        log.info("DEBUG | train=%d  val=%d samples", len(train_ds), len(val_ds))
    else:
        train_ds = SmilesDataset(data_dir / "train_tokenized.jsonl")
        val_ds   = SmilesDataset(data_dir / "val_tokenized.jsonl")
        log.info("train=%d  val=%d samples", len(train_ds), len(val_ds))

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
    )

    # ── training arguments ───────────────────────────────────────────────────
    adapter_dir = output_dir / "best_adapter"
    args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch"],
        gradient_accumulation_steps=cfg["gradient_accumulation"],
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_steps=int(cfg["warmup_ratio"] * cfg["num_train_epochs"] * 190),
        weight_decay=0.01,
        bf16=cfg.get("bf16", True) and torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=False,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        logging_steps=20,
        report_to="wandb",
        run_name=cfg.get("wandb_run_name") or None,
        seed=seed,
        dataloader_pin_memory=False,
    )

    import os
    os.environ["WANDB_PROJECT"] = cfg.get("wandb_project", "chem_semigroup")

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    log.info("Starting training …")
    trainer.train()

    # ── save best LoRA adapter only ──────────────────────────────────────────
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    log.info("Best adapter saved → %s", adapter_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="config.yaml")
    parser.add_argument("--debug",   action="store_true")
    parser.add_argument("--overfit", action="store_true",
                        help="Run overfit sanity check on 8 samples then exit")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main(cfg, debug=args.debug, overfit=args.overfit)
