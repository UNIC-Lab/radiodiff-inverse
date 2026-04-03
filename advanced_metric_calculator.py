import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from lpips import LPIPS

from util.metrics import evaluate_metrics

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MASK_NOISE_RE = re.compile(r"mask(?P<mask>[0-9.]+)_noise(?P<noise>[0-9.]+)")


def load_image(path: str) -> torch.Tensor:
    """Load image into [1, C, H, W] tensor normalized to [0, 1]."""
    img = Image.open(path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.CenterCrop((256, 256)),
    ])
    return transform(img).unsqueeze(0)


def _index_label_files(label_dir: str) -> Dict[str, str]:
    label_map: Dict[str, str] = {}
    for name in os.listdir(label_dir):
        ext = os.path.splitext(name)[1].lower()
        if ext not in IMAGE_EXTS:
            continue
        stem = os.path.splitext(name)[0]
        key = stem
        if stem.endswith("_label"):
            key = stem[:-6]
        label_map[key] = os.path.join(label_dir, name)
    return label_map


def get_image_pairs(exp_dir: str) -> List[Tuple[str, str, str]]:
    """
    Return image pairs as (sample_id, recon_path, label_path).

    Supports both naming conventions:
    - xxxx_recon.png + xxxx_label.png
    - same filename in recon/ and label/
    """
    recon_dir = os.path.join(exp_dir, "recon")
    label_dir = os.path.join(exp_dir, "label")
    if not (os.path.isdir(recon_dir) and os.path.isdir(label_dir)):
        return []

    label_map = _index_label_files(label_dir)
    pairs: List[Tuple[str, str, str]] = []

    for recon_name in sorted(os.listdir(recon_dir)):
        ext = os.path.splitext(recon_name)[1].lower()
        if ext not in IMAGE_EXTS:
            continue

        recon_stem = os.path.splitext(recon_name)[0]
        sample_id = recon_stem[:-6] if recon_stem.endswith("_recon") else recon_stem

        label_path = label_map.get(sample_id)
        if label_path is None:
            continue

        pairs.append((sample_id, os.path.join(recon_dir, recon_name), label_path))

    return pairs


def compute_experiment_metrics(exp_dir: str, device: torch.device, lpips_fn: LPIPS) -> Dict[str, float]:
    pairs = get_image_pairs(exp_dir)
    if not pairs:
        return {}

    per_sample_metrics: List[Dict[str, float]] = []

    for sample_id, recon_path, label_path in pairs:
        try:
            recon_img = load_image(recon_path).to(device)
            gt_img = load_image(label_path).to(device)
            metrics = evaluate_metrics(
                recon_img=recon_img,
                gt_img=gt_img,
                degraded_img=recon_img,
                device=device,
                lpips_fn=lpips_fn,
            )
            metrics["sample_id"] = sample_id
            per_sample_metrics.append(metrics)
        except Exception as exc:  # Keep batch processing robust.
            print(f"[WARN] Failed sample in {exp_dir}: {sample_id} ({exc})")

    if not per_sample_metrics:
        return {}

    metric_keys = [k for k in per_sample_metrics[0].keys() if k != "sample_id"]
    avg_metrics: Dict[str, float] = {}
    for key in metric_keys:
        avg_metrics[key] = float(np.mean([float(m[key]) for m in per_sample_metrics]))

    with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
        json.dump(avg_metrics, f, indent=2)

    with open(os.path.join(exp_dir, "per_sample_metrics.json"), "w") as f:
        json.dump(per_sample_metrics, f, indent=2)

    return avg_metrics


def discover_experiment_dirs(base_dir: str) -> List[str]:
    discovered: List[str] = []
    for root, dirs, _ in os.walk(base_dir):
        if "recon" in dirs and "label" in dirs:
            discovered.append(root)
    return sorted(discovered)


def _ensure_nested(d: dict, keys: List[str]) -> dict:
    cur = d
    for key in keys:
        if key not in cur:
            cur[key] = {}
        cur = cur[key]
    return cur


def _mask_noise_key(exp_name: str) -> Tuple[str, str]:
    match = MASK_NOISE_RE.search(exp_name)
    if not match:
        return "", ""
    mask_key = f"mask{float(match.group('mask')):.2f}"
    noise_key = f"noise{float(match.group('noise')):.3f}"
    return mask_key, noise_key


def build_summary(base_dir: str, metrics_by_exp: Dict[str, Dict[str, float]]) -> Dict[str, dict]:
    """
    Build summary in two forms:
    - `all_result.json`: best effort nested layout (compatible with old tooling when possible)
    - fallback entries stored under `experiments` with relative paths.
    """
    summary: Dict[str, dict] = {"experiments": {}}

    for exp_dir, metrics in metrics_by_exp.items():
        rel = str(Path(exp_dir).relative_to(base_dir))
        parts = Path(rel).parts

        if len(parts) >= 3:
            cond_type = parts[0]
            mask_type = parts[1]
            exp_name = parts[-1]
            mask_key, noise_key = _mask_noise_key(exp_name)
            if mask_key and noise_key:
                node = _ensure_nested(summary, [cond_type, mask_type, mask_key])
                node[noise_key] = metrics
                continue

        summary["experiments"][rel] = metrics

    return summary


def run(base_dir: str, exp_dir: str, gpu: int, skip_summary: bool) -> None:
    base_dir = os.path.abspath(base_dir)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    lpips_fn = LPIPS(net="alex").to(device)

    if exp_dir:
        targets = [os.path.abspath(exp_dir)]
    else:
        targets = discover_experiment_dirs(base_dir)

    if not targets:
        print(f"[WARN] No experiment directories found under: {base_dir}")
        return

    metrics_by_exp: Dict[str, Dict[str, float]] = {}

    for idx, target in enumerate(targets, start=1):
        print(f"[{idx}/{len(targets)}] Processing: {target}")
        avg = compute_experiment_metrics(target, device=device, lpips_fn=lpips_fn)
        if avg:
            metrics_by_exp[target] = avg

    if skip_summary or not metrics_by_exp:
        return

    summary = build_summary(base_dir=base_dir, metrics_by_exp=metrics_by_exp)

    all_result_path = os.path.join(base_dir, "all_result.json")
    with open(all_result_path, "w") as f:
        json.dump(summary, f, indent=2)

    ts_path = os.path.join(base_dir, f"summary_metrics_v{datetime.now().strftime('%m%d_%H%M')}.json")
    with open(ts_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary written to: {all_result_path}")
    print(f"Timestamped summary written to: {ts_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute image reconstruction metrics for experiment outputs.")
    parser.add_argument("--base_dir", type=str, default="results", help="Base directory containing experiment outputs")
    parser.add_argument("--exp_dir", type=str, default="", help="Optional single experiment directory")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id")
    parser.add_argument("--skip_summary", action="store_true", help="Skip writing summary JSON files")
    args = parser.parse_args()

    run(base_dir=args.base_dir, exp_dir=args.exp_dir, gpu=args.gpu, skip_summary=args.skip_summary)
