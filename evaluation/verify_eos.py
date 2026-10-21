"""
Verify stopping condition after overfitting on 8 samples.

Reports for each sample:
  - eos_token_ids actually passed to generate()
  - terminated_by_eos
  - n_generated_tokens
  - correctness (pred == target)
  - raw decoded output (with special tokens)
"""

import json
import logging
from pathlib import Path

import torch
import yaml

from data_pipeline.data import canon
from training.model_utils import load_tokenizer_and_model
from data_pipeline.prompts import build_prompt_text
from training.train import attach_lora, RawSmilesDataset
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

logging.basicConfig(format="%(levelname)s | %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["data_dir"])

    # ── load 8 training samples ───────────────────────────────────────────
    raw_samples = [
        json.loads(l)
        for l in (data_dir / "train_samples.jsonl").read_text().splitlines()
        if l.strip()
    ][:8]

    # ── load model + tokenizer ────────────────────────────────────────────
    tokenizer, model = load_tokenizer_and_model(cfg, load_model=True)
    tokenizer.padding_side = "right"
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model = attach_lora(model, cfg)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    log.info("Device: %s", device)

    # ── train to near-zero loss ───────────────────────────────────────────
    ds = RawSmilesDataset(raw_samples, tokenizer, cfg)
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, padding=True, label_pad_token_id=-100
    )
    args = TrainingArguments(
        output_dir="outputs/verify_eos_tmp",
        num_train_epochs=200,
        per_device_train_batch_size=8,
        learning_rate=1e-3,
        lr_scheduler_type="constant",
        logging_steps=40,
        save_strategy="no",
        bf16=cfg.get("bf16", True) and torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=False,
        report_to="none",
        seed=cfg["seed"],
        dataloader_pin_memory=False,
    )
    Trainer(model=model, args=args, train_dataset=ds, data_collator=collator).train()

    # ── confirm eos ids ───────────────────────────────────────────────────
    im_end_id    = tokenizer.convert_tokens_to_ids("<|im_end|>")
    endoftext_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    eos_ids      = list({im_end_id, endoftext_id, tokenizer.eos_token_id})
    # filter out None / unk
    eos_ids = [i for i in eos_ids if i is not None and i != tokenizer.unk_token_id]

    log.info("=" * 60)
    log.info("eos_ids passed to generate(): %s", eos_ids)
    log.info("  <|im_end|>    id = %s", im_end_id)
    log.info("  <|endoftext|> id = %s", endoftext_id)
    log.info("  tokenizer.eos_token_id = %s  (%r)", tokenizer.eos_token_id, tokenizer.eos_token)
    log.info("=" * 60)

    # ── inference on the same 8 samples ──────────────────────────────────
    model.eval()
    tokenizer.padding_side = "left"

    n_correct = 0
    n_eos = 0

    for i, s in enumerate(raw_samples):
        user_msg = build_prompt_text(cfg, s["source"], s["k"])
        prompt_str = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
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
                eos_token_id=eos_ids,
            )

        new_tokens = out[0][input_ids.shape[1]:]
        n_gen = len(new_tokens)
        raw_decoded = tokenizer.decode(new_tokens, skip_special_tokens=False)
        last_tok = new_tokens[-1].item() if n_gen > 0 else None
        terminated_by_eos = last_tok in eos_ids if last_tok is not None else False
        hit_max = (n_gen >= 256)

        # parse and compare
        import re
        text = re.sub(r"```[^\n]*\n?", "", raw_decoded).strip()
        pred = text.splitlines()[0].strip() if text.splitlines() else ""
        correct = (canon(pred) == canon(s["target"])) if pred else False

        n_eos     += int(terminated_by_eos)
        n_correct += int(correct)

        log.info(
            "sample %d | k=%d | terminated_by_eos=%s | hit_max=%s | n_gen=%d | correct=%s",
            i + 1, s["k"], terminated_by_eos, hit_max, n_gen, correct,
        )
        log.info(
            "  last_token_id=%s  raw=%r",
            last_tok,
            raw_decoded[:80],
        )

    log.info("=" * 60)
    log.info(
        "SUMMARY: terminated_by_eos %d/8 | correct %d/8",
        n_eos, n_correct,
    )


if __name__ == "__main__":
    main()
