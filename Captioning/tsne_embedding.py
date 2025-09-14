#!/usr/bin/env python
# tsne_gemma3_visual_embeddings.py
"""
t-SNE of visual embeddings from a Gemma-3 (vision) checkpoint.

- Labels are inferred from parent folder names of images.
- Works with PEFT/LoRA (optional).
- Robust extractor:
    1) model.get_image_features(pixel_values=...)
    2) model.vision_model / vision_tower / vision_encoder / image_encoder / vit
    3) Hook fallback on a vision module

Usage:
  python tsne_gemma3_visual_embeddings.py \
      --image_dir /path/to/images_root \
      --model_id your-org/your-gemma3-vision \
      --peft_path /optional/lora \
      --batch_size 8 \
      --save_csv gemma3_tsne.csv \
      --out_png gemma3_tsne.png
"""

import os, argparse
from pathlib import Path
from typing import List, Sequence
from PIL import Image
import numpy as np
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import pandas as pd

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig
)


# ----------------------------- IO helpers ---------------------------------
def find_images(root: str,
                exts={".jpg",".jpeg",".png",".webp",".bmp",".tif",".tiff"}) -> List[str]:
    paths = [str(p) for p in sorted(Path(root).rglob("*")) if p.suffix.lower() in exts]
    if not paths:
        raise FileNotFoundError(f"No images found under: {root}")
    return paths

def load_pil_batch(paths: Sequence[str], max_side: int=None) -> List[Image.Image]:
    ims = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        if max_side:
            im.thumbnail((max_side, max_side))
        ims.append(im)
    return ims


# ------------------------ Visual embedding extractor -----------------------
@torch.no_grad()
def get_image_embeddings(model, processor, pil_batch, device, pooling="mean"):
    """
    Returns [B, D] normalized embeddings (torch.FloatTensor, on CPU).
    Handles Gemma-3's processor that expects a 'text' argument.
    """
    if not isinstance(pil_batch, (list, tuple)):
        pil_batch = [pil_batch]

    # Gemma-3 processors expect text alongside images; pass empty strings.
    try:
        proc = processor(
            images=pil_batch,
            text=[""] * len(pil_batch),   # <-- critical to avoid NoneType crash
            return_tensors="pt",
            padding=True
        )
    except TypeError:
        # Fallback: try image_processor if present
        if hasattr(processor, "image_processor"):
            proc = processor.image_processor(images=pil_batch, return_tensors="pt")
        else:
            raise

    # Move tensors to device
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in proc.items()}
    pixel_values = inputs.get("pixel_values", None)
    if pixel_values is None:
        raise RuntimeError("processor did not return 'pixel_values'.")

    # 1) Preferred: CLIP-like API
    if hasattr(model, "get_image_features"):
        feats = model.get_image_features(pixel_values=pixel_values)
        return F.normalize(feats.float().cpu(), dim=-1)

    # 2) Direct call to a vision tower
    for name in ["vision_model", "vision_tower", "vision_encoder", "image_encoder", "vit"]:
        if hasattr(model, name):
            vt = getattr(model, name)
            if hasattr(vt, "model"):
                vt = vt.model
            try:
                out = vt(pixel_values=pixel_values, output_hidden_states=True)
                last = getattr(out, "last_hidden_state", out[0])
            except TypeError:
                # Some encoders may not accept output_hidden_states
                out = vt(pixel_values=pixel_values)
                last = getattr(out, "last_hidden_state", out[0])
            pooled = last.mean(dim=1) if pooling == "mean" else last[:, 0]
            return F.normalize(pooled.float().cpu(), dim=-1)

    # 3) Hook fallback: attach to a likely vision module and forward it
    vision_feats = {}

    def hook_fn(_m, _inp, out):
        x = out[0] if isinstance(out, (tuple, list)) else out
        pooled = x.mean(dim=1) if pooling == "mean" else x[:, 0]
        vision_feats["x"] = pooled

    target = None
    for n, m in model.named_modules():
        if any(t in n.lower() for t in ["vision", "vit", "image"]):
            target = m
    if target is None:
        raise RuntimeError("No vision submodule found on model to hook.")

    h = target.register_forward_hook(hook_fn)
    try:
        _ = target(pixel_values=pixel_values)
    finally:
        h.remove()

    if "x" not in vision_feats:
        raise RuntimeError("Vision hook did not capture features; adjust target module.")

    return F.normalize(vision_feats["x"].float().cpu(), dim=-1)


# ----------------------------- Plot helpers --------------------------------
def plot_tsne(xy: np.ndarray, labels: List[str], title: str, out_png: str):
    uniq = sorted(set(labels))
    plt.figure(figsize=(8, 7), dpi=140)
    # Plot per label (let matplotlib assign distinct colors)
    for u in uniq:
        mask = [l == u for l in labels]
        arr = xy[mask]
        plt.scatter(arr[:, 0], arr[:, 1], s=14, alpha=0.85, label=u)
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    # Avoid unreadable legends if too many classes
    if len(uniq) <= 25:
        plt.legend(loc="best", fontsize=8, frameon=True)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    print(f"[OK] Saved plot: {out_png}")


# --------------------------------- Main ------------------------------------
def main():
    ap = argparse.ArgumentParser(description="t-SNE of Gemma-3 visual embeddings")
    ap.add_argument("--image_dir", required=True, help="Root folder; subfolders are labels.")
    ap.add_argument("--model_id", required=True, help="HF model id or local path.")
    ap.add_argument("--peft_path", default=None, help="Optional PEFT/LoRA path.")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    ap.add_argument("--max_images", type=int, default=0, help="0 = all images.")
    ap.add_argument("--max_side", type=int, default=512, help="Resize longer side to this.")
    ap.add_argument("--quant4", action="store_true", help="Load model in 4-bit (bitsandbytes).")
    ap.add_argument("--save_csv", default="embeddings_tsne.csv")
    ap.add_argument("--out_png", default="tsne.png")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # Gather images and labels
    paths = find_images(args.image_dir)
    if args.max_images and args.max_images > 0:
        paths = paths[:args.max_images]
    labels = [Path(p).parent.name for p in paths]
    print(f"[INFO] Found {len(paths)} images across {len(set(labels))} labels.")

    # Processor & model
    quant = None
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    if args.quant4:
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

    print("[INFO] Loading processor/model...")
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        quantization_config=quant,
        trust_remote_code=True
    )

    # Optional PEFT merge
    if args.peft_path:
        from peft import PeftModel
        print(f"[INFO] Loading PEFT from {args.peft_path}")
        model = PeftModel.from_pretrained(model, args.peft_path)
        try:
            model = model.merge_and_unload()
            print("[INFO] PEFT merged into base model.")
        except Exception as e:
            print(f"[WARN] merge_and_unload failed: {e}")

    model.eval()

    # Extract embeddings
    print("[INFO] Extracting embeddings...")
    feats_list = []
    B = args.batch_size
    for i in range(0, len(paths), B):
        batch_paths = paths[i:i+B]
        pil_batch = load_pil_batch(batch_paths, max_side=args.max_side)
        feats = get_image_embeddings(model, processor, pil_batch, device, pooling=args.pooling)
        feats_list.append(feats)
    feats = torch.cat(feats_list, dim=0).numpy()  # [N, D]
    print(f"[INFO] Embeddings shape: {feats.shape}")

    # Standardize before t-SNE (often improves layout)
    feats_std = StandardScaler(with_mean=True, with_std=True).fit_transform(feats)

    # Robust perplexity choice
    n = feats_std.shape[0]
    if n < 3:
        raise ValueError("Need at least 3 samples for t-SNE.")
    # perplexity must be < n; choose conservatively
    perplexity = min(30, max(2, min(n - 1, n // 3)))

    print(f"[INFO] Running t-SNE (N={n}, perplexity={perplexity})...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        metric="cosine",
        random_state=42,
        n_iter=1500
    )
    xy = tsne.fit_transform(feats_std)  # [N, 2]

    # Save CSV with embeddings + 2D coords
    df = pd.DataFrame({
        "path": paths,
        "label": labels,
        "tsne_x": xy[:, 0],
        "tsne_y": xy[:, 1],
    })
    for j in range(feats.shape[1]):
        df[f"f{j}"] = feats[:, j]
    df.to_csv(args.save_csv, index=False)
    print(f"[OK] Saved embeddings and t-SNE: {args.save_csv}")

    # Plot
    title = f"t-SNE of Gemma-3 visual embeddings (perplexity={perplexity}, N={n})"
    plot_tsne(xy, labels, title, args.out_png)


if __name__ == "__main__":
    main()
