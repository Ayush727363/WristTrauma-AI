"""
03_evaluate_test.py
--------------------
FINAL, one-time evaluation of WristNet on the held-out TEST split. This split
has never been used for training or for picking the best checkpoint or the
per-class thresholds -- those were both selected on VAL only, then frozen
(see models/wristnet/tuned_thresholds.json). Here we apply them UNCHANGED to
test and report whatever comes out -- that is what makes this evaluation
honest.

Produces, under results/wristnet_test_eval/:
  - overall_test_metrics.json        (macro AUROC, macro AUPRC, macro F1)
  - per_class_test_metrics.csv       (AUROC, AUPRC, precision/recall/F1 @ frozen threshold)
  - cam_samples/                     a handful of annotated test images (heatmap + boxes)
  - final_report.md                  human-readable summary, same style as the old YOLO report

Why AUPRC is reported alongside AUROC: AUROC stays optimistic for very rare
classes (bone_lesion is ~1% prevalence) because it's computed over ALL
positive-negative pairs, most of which are "easy". AUPRC's random baseline
IS the class prevalence, so it's the metric that actually tells you whether
the model beats "always guess negative" in a way that matches real-world
usefulness. Report both; lead with AUPRC for bone_lesion in the writeup.

Usage:
    python scripts\\03_evaluate_test.py
    python scripts\\03_evaluate_test.py --weights models\\wristnet\\best.pt --n-cam-samples 12
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dataset import CLASS_NAMES, N_CLASSES, WristMultiLabelDataset  # noqa: E402
from model import WristNet  # noqa: E402
from cam_utils import analyze_image, make_heatmap_overlay  # noqa: E402


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, default=None,
                   help="Path to best.pt. Default: models/wristnet/best.pt")
    p.add_argument("--thresholds", type=str, default=None,
                   help="Path to tuned_thresholds.json. Default: models/wristnet/tuned_thresholds.json")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--n-cam-samples", type=int, default=10,
                   help="How many test images to save annotated CAM visualizations for.")
    p.add_argument("--name", type=str, default="wristnet_test_eval")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def run_inference(model, loader, device):
    """Single pass over the loader -> (probs, targets, stems) as numpy arrays / list."""
    model.eval()
    all_logits, all_targets, all_stems = [], [], []
    for imgs, labels, stems in loader:
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        all_logits.append(logits.cpu())
        all_targets.append(labels)
        all_stems.extend(stems)
    logits = torch.cat(all_logits).numpy()
    targets = torch.cat(all_targets).numpy()
    probs = 1 / (1 + np.exp(-logits))
    return probs, targets, all_stems


def compute_metrics(probs: np.ndarray, targets: np.ndarray, thresholds: dict) -> dict:
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                  precision_recall_fscore_support)

    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        y_true, y_prob = targets[:, i], probs[:, i]
        t = thresholds.get(name, 0.5)
        y_pred = (y_prob >= t).astype(np.float32)

        try:
            auroc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
        except ValueError:
            auroc = float("nan")
        try:
            auprc = average_precision_score(y_true, y_prob) if y_true.sum() > 0 else float("nan")
        except ValueError:
            auprc = float("nan")

        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        prevalence = float(y_true.mean())

        per_class[name] = {
            "auroc": auroc, "auprc": auprc,
            "auprc_random_baseline": prevalence,  # AUPRC of a model that guesses at random -- compare against this, not 0
            "threshold_used": t,
            "precision": prec, "recall": rec, "f1": f1,
            "n_positive": int(y_true.sum()), "n_total": int(len(y_true)),
        }

    valid_auroc = [v["auroc"] for v in per_class.values() if not np.isnan(v["auroc"])]
    valid_auprc = [v["auprc"] for v in per_class.values() if not np.isnan(v["auprc"])]
    overall = {
        "macro_auroc": float(np.mean(valid_auroc)) if valid_auroc else float("nan"),
        "macro_auprc": float(np.mean(valid_auprc)) if valid_auprc else float("nan"),
        "macro_f1": float(np.mean([v["f1"] for v in per_class.values()])),
    }
    return {"overall": overall, "per_class": per_class}


def save_cam_samples(model, test_ds, device, thresholds, out_dir: Path, n_samples: int, seed: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)

    # bias the sample toward images that actually have findings, so the
    # visualizations are informative (not mostly-empty normal wrists)
    has_finding = test_ds.df[test_ds.df[test_ds.label_cols].sum(axis=1) >= 1]
    pool = list(has_finding.index)
    chosen = random.sample(pool, k=min(n_samples, len(pool)))
    positions = [test_ds.df.index.get_loc(i) for i in chosen]

    colors = {"fracture": (0, 0, 255), "foreign_material": (0, 255, 255),
              "softtissue_indirect": (255, 0, 0), "bone_lesion": (0, 255, 0)}

    saved_paths = []
    for pos in positions:
        img_t, label_t, stem = test_ds[pos]
        result = analyze_image(model, img_t, device, CLASS_NAMES, thresholds)

        base_np = (img_t[0].numpy() * 255).astype(np.uint8)
        base_bgr = cv2.cvtColor(base_np, cv2.COLOR_GRAY2BGR)

        # heatmap overlay for the class with highest predicted probability
        top_class = max(CLASS_NAMES, key=lambda n: result[n]["probability"])
        heatmap_vis = make_heatmap_overlay(base_np, result["_cams_normalized"][top_class])

        boxed = base_bgr.copy()
        for name in CLASS_NAMES:
            for (x1, y1, x2, y2) in result[name]["boxes"]:
                cv2.rectangle(boxed, (x1, y1), (x2, y2), colors[name], 2)
                label_text = f"{name} {result[name]['probability']:.2f}"
                cv2.putText(boxed, label_text, (x1, max(0, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[name], 1)

        # classes we don't box (merged, multi-finding) are annotated as text
        # in the corner so the visualization still shows they were detected
        y_text = 14
        for name in CLASS_NAMES:
            r = result[name]
            if r["present"] and not r["localizable"]:
                cv2.putText(boxed, f"{name}: present ({r['probability']:.2f})",
                            (4, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.38, colors[name], 1)
                y_text += 14

        combined = np.hstack([base_bgr, heatmap_vis, boxed])
        path = out_dir / f"{stem}.png"
        cv2.imwrite(str(path), combined)
        saved_paths.append(path)

    return saved_paths


def main():
    args = parse_args()
    root = project_root()

    weights_path = Path(args.weights) if args.weights else root / "models" / "wristnet" / "best.pt"
    thresholds_path = Path(args.thresholds) if args.thresholds else root / "models" / "wristnet" / "tuned_thresholds.json"
    csv_path = root / "data" / "classification" / "labels_multilabel.csv"
    images_root = root / "data" / "processed" / "images"
    out_dir = root / "results" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    if not weights_path.exists():
        sys.exit(f"[FATAL] weights not found: {weights_path}")
    if not thresholds_path.exists():
        sys.exit(f"[FATAL] thresholds not found: {thresholds_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device: {device}")

    print(f"[info] loading weights: {weights_path}")
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    # Rebuild the architecture with the SAME options the checkpoint was trained
    # with (dropout/fine_cam are stored in the checkpoint's args), otherwise
    # load_state_dict will fail or silently mismatch.
    ck_args = ckpt.get("args", {})
    model = WristNet(
        n_classes=N_CLASSES,
        dropout=ck_args.get("dropout", 0.4),
        fine_cam=not ck_args.get("coarse_cam", False),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    img_size = ckpt["args"]["img_size"]
    print(f"[info] checkpoint from epoch {ckpt.get('epoch')}, img_size={img_size}")

    with open(thresholds_path) as f:
        thresholds = json.load(f)
    print(f"[info] frozen per-class thresholds (tuned on VAL, applied unchanged to TEST): {thresholds}")

    print("\n[info] loading TEST split (held out, never used for training/selection)...")
    test_ds = WristMultiLabelDataset(csv_path, images_root / "test", "test",
                                      img_size=img_size, augment=False)
    print(f"[ok] test images: {len(test_ds)}")

    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, pin_memory=(device.type == "cuda"))

    print("\n[info] running inference on TEST split...")
    probs, targets, stems = run_inference(model, test_loader, device)

    print("[info] computing metrics...")
    metrics = compute_metrics(probs, targets, thresholds)

    with open(out_dir / "overall_test_metrics.json", "w") as f:
        json.dump(metrics["overall"], f, indent=2)
    print(f"[saved] {out_dir / 'overall_test_metrics.json'}")

    import csv as _csv
    with open(out_dir / "per_class_test_metrics.csv", "w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=["class", *next(iter(metrics["per_class"].values())).keys()])
        writer.writeheader()
        for name, m in metrics["per_class"].items():
            writer.writerow({"class": name, **m})
    print(f"[saved] {out_dir / 'per_class_test_metrics.csv'}")

    print(f"\n[info] saving {args.n_cam_samples} annotated CAM sample visualizations...")
    cam_dir = out_dir / "cam_samples"
    saved = save_cam_samples(model, test_ds, device, thresholds, cam_dir, args.n_cam_samples, args.seed)
    print(f"[ok] saved {len(saved)} visualizations to {cam_dir}")

    # ---- markdown report ----
    lines = []
    lines.append("# WristTrauma-AI -- WristNet (Custom CNN) Test-Set Evaluation\n")
    lines.append(f"- Checkpoint: `{weights_path}` (epoch {ckpt.get('epoch')})")
    lines.append(f"- Test set size: {len(test_ds)} images (patient-level held-out split, never used in training)")
    lines.append(f"- Thresholds: tuned on VAL split, frozen and applied unchanged here (`{thresholds_path.name}`)")
    lines.append("")
    lines.append("## Overall")
    lines.append(f"- Macro AUROC: {metrics['overall']['macro_auroc']:.4f}")
    lines.append(f"- Macro AUPRC: {metrics['overall']['macro_auprc']:.4f}")
    lines.append(f"- Macro F1 (at tuned thresholds): {metrics['overall']['macro_f1']:.4f}")
    lines.append("")
    lines.append("## Per-class")
    lines.append("| Class | AUROC | AUPRC | AUPRC random baseline | Precision | Recall | F1 | Threshold | n_positive |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for name, m in metrics["per_class"].items():
        def fmt(x):
            return "n/a" if x != x else f"{x:.4f}"
        lines.append(f"| {name} | {fmt(m['auroc'])} | {fmt(m['auprc'])} | {fmt(m['auprc_random_baseline'])} | "
                      f"{fmt(m['precision'])} | {fmt(m['recall'])} | {fmt(m['f1'])} | "
                      f"{m['threshold_used']:.2f} | {m['n_positive']}/{m['n_total']} |")
    lines.append("")
    lines.append("**Reading AUPRC**: compare each class's AUPRC to its own random baseline (= prevalence), "
                  "not to 0 or to other classes' AUPRC. A class with 1% prevalence and AUPRC 0.15 is "
                  "meaningfully better than random even though 0.15 looks low next to a common class's AUPRC.")
    lines.append("")
    lines.append(f"CAM sample visualizations (original | heatmap | boxes): `{cam_dir}`")

    report_path = out_dir / "final_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[saved] {report_path}")

    print("\n" + "=" * 70)
    print("TEST-SET EVALUATION COMPLETE")
    print("=" * 70)
    print(f"  Macro AUROC : {metrics['overall']['macro_auroc']:.4f}")
    print(f"  Macro AUPRC : {metrics['overall']['macro_auprc']:.4f}")
    print(f"  Macro F1    : {metrics['overall']['macro_f1']:.4f}")
    print("\n  per-class:")
    for name, m in metrics["per_class"].items():
        auroc_s = "n/a" if m["auroc"] != m["auroc"] else f"{m['auroc']:.3f}"
        auprc_s = "n/a" if m["auprc"] != m["auprc"] else f"{m['auprc']:.3f}"
        print(f"    {name:22s}: AUROC={auroc_s}  AUPRC={auprc_s} "
              f"(baseline={m['auprc_random_baseline']:.3f})  F1={m['f1']:.3f}  "
              f"P={m['precision']:.3f}  R={m['recall']:.3f}")
    print("=" * 70)
    print(f"\nFull report: {report_path}")


if __name__ == "__main__":
    main()