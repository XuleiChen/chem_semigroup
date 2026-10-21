"""
Model / tokenizer loading with three-priority fallback:
  1. local path  (cfg["model_path"] non-empty)
  2. HuggingFace mirror  (HF_ENDPOINT=cfg["hf_mirror"])
  3. ModelScope

Also exposes get_hidden_states(model, tokenizer, texts, layer) for white-box analysis.
"""

import logging
import os
from typing import Optional

import torch

log = logging.getLogger(__name__)


def _load_from_hf(model_name: str, hf_mirror: str, load_model: bool):
    os.environ["HF_ENDPOINT"] = hf_mirror
    log.info("Trying HF mirror: %s", hf_mirror)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if not load_model:
        return tokenizer, None
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    return tokenizer, model


def _load_from_modelscope(model_name: str, load_model: bool):
    log.info("Trying ModelScope for %s", model_name)
    from modelscope import AutoModelForCausalLM, AutoTokenizer, snapshot_download
    local_dir = snapshot_download(model_name)
    tokenizer = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)
    if not load_model:
        return tokenizer, None
    model = AutoModelForCausalLM.from_pretrained(
        local_dir, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    return tokenizer, model


def load_tokenizer_and_model(cfg: dict, load_model: bool = True):
    """
    Returns (tokenizer, model).  model is None when load_model=False.
    Priority: local path → HF mirror → ModelScope.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name: str = cfg["model_name"]
    model_path: str = cfg.get("model_path", "")
    hf_mirror:  str = cfg.get("hf_mirror", "https://hf-mirror.com")

    # ── 1. local path ────────────────────────────────────────────────────────
    if model_path:
        log.info("Loading from local path: %s", model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if not load_model:
            return tokenizer, None
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        return tokenizer, model

    # ── 2. HF mirror ─────────────────────────────────────────────────────────
    try:
        return _load_from_hf(model_name, hf_mirror, load_model)
    except Exception as e:
        log.warning("HF mirror failed (%s), trying ModelScope …", e)

    # ── 3. ModelScope ────────────────────────────────────────────────────────
    return _load_from_modelscope(model_name, load_model)


def get_hidden_states(
    model,
    tokenizer,
    texts: list[str],
    layer: int,
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Return hidden states at `layer` for each text in `texts`.
    Shape: (batch, seq_len, hidden_size).
    Uses the last token's hidden state as a fixed-size representation
    if you want one vector per text; otherwise returns the full tensor.
    """
    if device is None:
        device = next(model.parameters()).device
    enc = tokenizer(texts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    # hidden_states is a tuple of (n_layers+1) tensors
    return out.hidden_states[layer]
