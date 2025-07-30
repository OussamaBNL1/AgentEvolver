#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_adv_dist.py – Visualise advantage distributions (before vs after
normalisation) for a BeyondAgent checkpoint.

Example
-------
    python plot_adv_dist.py \
        --ckpt /path/to/global_step_20/actor \
        --adv-before advantage_dump/adv_before.pt \
        --adv-after  advantage_dump/adv_after.pt \
        --bins 200 --show-logy

The script displays a two‑panel Matplotlib figure:
1. Overlayed histograms of raw & normalised advantage.
2. A dedicated histogram for the raw (pre‑norm) distribution to inspect long
   tails more clearly.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

plt.rcParams["figure.dpi"] = 120  # crisper inline in notebooks


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _load_tensor(path: str | Path) -> torch.Tensor:
    tensor = torch.load(path, map_location="cpu")
    if tensor.ndim > 1:
        tensor = tensor.reshape(-1)
    return tensor


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot advantage histograms")
    parser.add_argument("--ckpt", required=True, help="Actor checkpoint directory (for completeness)")
    parser.add_argument("--adv-before", required=True, help="Raw advantage .pt file")
    parser.add_argument("--adv-after", required=True, help="Normalised advantage .pt file")
    parser.add_argument("--bins", type=int, default=200, help="Number of histogram bins")
    parser.add_argument("--show-logy", action="store_true", help="Use log scale on y‑axis")
    args = parser.parse_args()

    # (Optional) load tokenizer/model just to verify ckpt directory works
    try:
        AutoTokenizer.from_pretrained(args.ckpt, trust_remote_code=True)
        AutoModelForCausalLM.from_pretrained(args.ckpt, trust_remote_code=True, device_map="cpu")
    except Exception as e:
        print(f"[WARN] Could not load model/tokenizer from {args.ckpt}: {e}. Continuing …")

    adv_before = _load_tensor(args.adv_before).numpy()
    adv_after = _load_tensor(args.adv_after).numpy()

    print(f"Loaded {adv_before.size} raw tokens and {adv_after.size} normalised tokens")

    plt.figure(figsize=(10, 4))

    # Panel 1 – overlay
    plt.subplot(1, 2, 1)
    plt.hist(adv_before, bins=args.bins, alpha=0.5, label="Before norm")
    plt.hist(adv_after, bins=args.bins, alpha=0.5, label="After norm")
    plt.title("Advantage distribution (overlay)")
    plt.xlabel("Advantage value")
    plt.ylabel("Token count")
    if args.show_logy:
        plt.yscale("log")
    plt.legend()

    # Panel 2 – raw only
    plt.subplot(1, 2, 2)
    plt.hist(adv_before, bins=args.bins, alpha=0.8, color="grey")
    plt.title("Before normalisation")
    plt.xlabel("Advantage value")
    if args.show_logy:
        plt.yscale("log")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
