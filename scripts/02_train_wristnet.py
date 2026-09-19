"""
02_train_wristnet.py
---------------------
Trains WristNet (src/model.py) on the 4 merged classes using the multilabel
CSV produced by 01_prepare_labels.py and the dataset loader in src/dataset.py.

What this script does, in order:
  1. Loads train/val datasets (test is NEVER touched here -- held out for
     Phase 2 evaluation only).
  2. Computes per-class positive counts from the TRAIN split only, and builds
     a FocalLoss with alpha weights from those counts (never from val/test --
     that would leak information about the eval sets into training).
  3. Trains with AMP (mixed precision) since your RTX 5060 supports it, same
     as your YOLO runs.
  4. After every epoch, evaluates on val: per-class AUROC, precision,
     recall, F1 at a 0.5 threshold, and macro-F1.
  5. Saves the checkpoint with the best val macro-F1 to
     models/wristnet/best.pt, and the last epoch to last.pt.
  6. Writes a training curve CSV (models/wristnet/training_log.csv) and a
     final JSON summary, mirroring the style of your old YOLO scripts.

data/raw/, data/processed/labels/*.txt and the YOLO weights are never
touched by this script.

Usage (from an activated venv, project root):
    python scripts\\02_train_wristnet.py
    python scripts\\02_train_wristnet.py --epochs 40 --batch 32 --img-size 256
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dataset import CLASS_NAMES, N_CLASSES, build_datasets  # noqa: E402
from model import FocalLoss, WristNet  # noqa: E402


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-4,
                   help="Lowered from 1e-3 -- the model was converging/overfitting fast "
                        "(train_loss kept dropping while val stalled), a smaller LR learns "
                        "more gradually and should generalize better.")
    p.add_argument("--weight-decay", type=float, default=1e-2,
                   help="Raised from 1e-4 to 1e-2 (AdamW's normal working range) -- extra "
                        "regularization to fight overfitting, paired with the dropout "
                        "increase and the much stronger augmentation in dataset.py.")
    p.add_argument("--dropout", type=float, default=0.4,
                   help="Dropout before the classifier (raised from 0.3 to fight overfitting).")
    p.add_argument("--coarse-cam", action="store_true", default=False,
                   help="Use the original 8x8 final feature map instead of the default 16x16. "
                        "16x16 gives 4x finer CAM heatmaps/boxes; pass this flag only to "
                        "reproduce the older, coarser localization behaviour.")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--gamma", type=float, default=2.0, help="Focal loss focusing parameter.")
    p.add_argument("--patience", type=int, default=20, help="Early-stop patience (epochs with no val improvement on --select-metric).")
    p.add_argument("--select-metric", type=str, default="stable_auroc",
                   choices=["macro_auroc", "stable_auroc", "macro_tuned_f1", "macro_f1"],
                   help="Metric used to pick the best checkpoint and drive early stopping. "
                        "Default stable_auroc: PRIORITY-WEIGHTED mean AUROC over fracture "
                        "(0.50), softtissue_indirect (0.30) and foreign_material (0.20), "
                        "excluding bone_lesion, whose ~31 val positives make its AUROC swing "
                        "0.1+ epoch to epoch from pure noise and previously caused early "
                        "stopping driven by that noise rather than genuine convergence. "
                        "bone_lesion is still fully evaluated and logged every epoch -- it's "
                        "only excluded from the SELECTION metric.")
    p.add_argument("--name", type=str, default="wristnet")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--use-cache", action="store_true", default=False,
                   help="Cache preprocessed images to disk as .npy after first load "
                        "(faster after epoch 1, uses extra disk space under data/classification/cache_<size>/).")
    return p.parse_args()


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device, threshold: float = 0.5) -> dict:
    """Runs the model over a loader, returns per-class + macro metrics.

    Reports metrics TWO ways for each class:
      - at the fixed `threshold` (default 0.5)
      - at the threshold that maximizes that class's F1 on THIS split
        ("tuned"), found by sweeping candidate thresholds

    Why both: for rare classes (bone_lesion is ~1.2% prevalence), a fixed 0.5
    threshold is close to meaningless -- the model can rank positives above
    negatives well (good AUROC) while almost never outputting >0.5 confidence
    (terrible F1@0.5). Tuning the threshold on the VALIDATION split is the
    standard, legitimate fix; the tuned thresholds are then frozen and applied
    unchanged to the test split in Phase 2. AUROC is reported because it is
    threshold-free and therefore the stable signal of whether the model is
    actually learning.
    """
    from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

    model.eval()
    all_logits, all_targets = [], []
    for imgs, labels, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        all_logits.append(logits.cpu())
        all_targets.append(labels)

    logits = torch.cat(all_logits).numpy()
    targets = torch.cat(all_targets).numpy()
    probs = 1 / (1 + np.exp(-logits))  # sigmoid
    preds = (probs >= threshold).astype(np.float32)

    per_class = {}
    aurocs = []
    for i, name in enumerate(CLASS_NAMES):
        y_true, y_prob, y_pred = targets[:, i], probs[:, i], preds[:, i]
        try:
            auroc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
        except ValueError:
            auroc = float("nan")
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )

        # --- threshold sweep: find the threshold maximising this class's F1 ---
        best_t, best_f1, best_p, best_r = threshold, f1, prec, rec
        if len(np.unique(y_true)) > 1:
            for t in np.arange(0.01, 0.99, 0.01):
                yp = (y_prob >= t).astype(np.float32)
                p_t, r_t, f_t, _ = precision_recall_fscore_support(
                    y_true, yp, average="binary", zero_division=0
                )
                if f_t > best_f1:
                    best_t, best_f1, best_p, best_r = float(t), f_t, p_t, r_t

        per_class[name] = {
            "auroc": auroc,
            "precision": prec, "recall": rec, "f1": f1,          # at fixed threshold
            "tuned_threshold": best_t,
            "tuned_precision": best_p, "tuned_recall": best_r, "tuned_f1": best_f1,
            "n_positive": int(y_true.sum()),
        }
        if not np.isnan(auroc):
            aurocs.append(auroc)

    macro_f1 = float(np.mean([v["f1"] for v in per_class.values()]))
    macro_tuned_f1 = float(np.mean([v["tuned_f1"] for v in per_class.values()]))
    macro_auroc = float(np.mean(aurocs)) if aurocs else float("nan")

    # "stable" AUROC = WEIGHTED mean over the 3 classes that actually matter
    # for this project (bone_lesion excluded -- only ~31 val positives, its
    # AUROC swings 0.1+ epoch to epoch from pure noise and previously caused
    # early stopping driven by that noise rather than genuine convergence).
    # Weighted by priority: fracture > softtissue_indirect > foreign_material
    # (the priority order given for this project), so the checkpoint that
    # gets selected is the one that's best on what actually matters most,
    # not just an unweighted average of the three. bone_lesion is still
    # fully computed and reported every epoch above -- only left out of the
    # metric used to PICK checkpoints / drive early stopping.
    stable_weights = {"fracture": 0.5, "softtissue_indirect": 0.3, "foreign_material": 0.2}
    weighted_sum, weight_total = 0.0, 0.0
    for name, w in stable_weights.items():
        auroc_val = per_class[name]["auroc"]
        if not np.isnan(auroc_val):
            weighted_sum += w * auroc_val
            weight_total += w
    stable_auroc = weighted_sum / weight_total if weight_total > 0 else float("nan")

    return {"per_class": per_class, "macro_f1": macro_f1,
            "macro_tuned_f1": macro_tuned_f1, "macro_auroc": macro_auroc,
            "stable_auroc": stable_auroc}


def main():
    args = parse_args()
    set_seed(args.seed)

    root = project_root()
    csv_path = root / "data" / "classification" / "labels_multilabel.csv"
    images_root = root / "data" / "processed" / "images"
    out_dir = root / "models" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device: {device}")
    if device.type == "cuda":
        print(f"[info] GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[warn] CUDA not available -- training on CPU will be very slow.")

    print("\n[info] building datasets...")
    cache_root = (root / "data" / "classification") if args.use_cache else None
    if args.use_cache:
        print(f"[info] preprocessing cache ENABLED -> {cache_root / f'cache_{args.img_size}'}")
        print("       (first epoch will be slower while the cache fills; every epoch after is much faster)")
    train_ds, val_ds, _test_ds = build_datasets(csv_path, images_root, img_size=args.img_size,
                                                 use_cache=args.use_cache, cache_root=cache_root)
    print(f"[ok] train={len(train_ds)}  val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                               num_workers=args.workers, pin_memory=(device.type == "cuda"),
                               drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                             num_workers=args.workers, pin_memory=(device.type == "cuda"))

    # --- class weights from TRAIN split only ---
    pos_counts = [int(train_ds.df[c].sum()) for c in train_ds.label_cols]
    print(f"\n[info] train positive counts: {dict(zip(CLASS_NAMES, pos_counts))}")
    loss_fn = FocalLoss.from_class_frequencies(pos_counts, total=len(train_ds), gamma=args.gamma)
    loss_fn = loss_fn.to(device)
    print(f"[info] focal loss alpha weights: {loss_fn.alpha.tolist()}  (gamma={args.gamma})")

    model = WristNet(n_classes=N_CLASSES, dropout=args.dropout, fine_cam=not args.coarse_cam).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[info] model parameters: {n_params:,}")
    print(f"[info] dropout={args.dropout}  fine_cam={not args.coarse_cam} "
          f"(final feature map {'16x16' if not args.coarse_cam else '8x8'} for 256px input)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device="cuda", enabled=(device.type == "cuda"))

    log_rows = []
    best_score = -1.0
    best_epoch = -1
    epochs_since_improve = 0
    sel = args.select_metric

    print(f"\n[info] starting training: {args.epochs} epochs, batch={args.batch}, img_size={args.img_size}")
    print(f"[info] checkpoint selection / early stopping metric: {sel}\n")
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        t_ep = time.time()
        running_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch",
                    leave=False, dynamic_ncols=True)
        for imgs, labels, _ in pbar:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                logits = model(imgs)
                loss = loss_fn(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{running_loss / n_batches:.4f}")

        scheduler.step()
        train_loss = running_loss / max(n_batches, 1)

        val_metrics = evaluate(model, val_loader, device)
        macro_f1 = val_metrics["macro_f1"]
        macro_tuned_f1 = val_metrics["macro_tuned_f1"]
        macro_auroc = val_metrics["macro_auroc"]
        stable_auroc = val_metrics["stable_auroc"]
        score = val_metrics[sel]
        ep_time = time.time() - t_ep

        print(f"[epoch {epoch:3d}/{args.epochs}] "
              f"train_loss={train_loss:.4f}  val_AUROC={macro_auroc:.4f}  "
              f"val_stable_AUROC={stable_auroc:.4f}  "
              f"val_F1@0.5={macro_f1:.4f}  val_F1@tuned={macro_tuned_f1:.4f}  ({ep_time:.1f}s)")
        for name, m in val_metrics["per_class"].items():
            auroc_s = f"{m['auroc']:.3f}" if not np.isnan(m["auroc"]) else "n/a"
            print(f"    {name:22s}: AUROC={auroc_s}  "
                  f"F1@0.5={m['f1']:.3f}  |  tuned t={m['tuned_threshold']:.2f} "
                  f"P={m['tuned_precision']:.3f} R={m['tuned_recall']:.3f} F1={m['tuned_f1']:.3f} "
                  f"(n_pos={m['n_positive']})")

        log_rows.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_macro_auroc": macro_auroc,
            "val_stable_auroc": stable_auroc,
            "val_macro_f1": macro_f1,
            "val_macro_tuned_f1": macro_tuned_f1,
            "epoch_time_s": round(ep_time, 1),
            "lr": optimizer.param_groups[0]["lr"],
        })

        torch.save({"model_state": model.state_dict(), "epoch": epoch,
                    "val_metrics": val_metrics, "args": vars(args)}, out_dir / "last.pt")

        if score > best_score:
            best_score = score
            best_epoch = epoch
            epochs_since_improve = 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch,
                        "select_metric": sel, "select_score": score,
                        "val_metrics": val_metrics,
                        "args": vars(args)}, out_dir / "best.pt")
            print(f"    -> new best ({sel}={score:.4f}), saved to {out_dir / 'best.pt'}")
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= args.patience:
                print(f"\n[info] early stopping: no val {sel} improvement for {args.patience} epochs.")
                break

    total_time = time.time() - t_start

    with open(out_dir / "training_log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"\n[saved] {out_dir / 'training_log.csv'}")

    # Reload the best checkpoint's metrics so the summary/thresholds describe
    # the model we actually keep (best.pt), not whatever the last epoch was.
    best_ckpt = torch.load(out_dir / "best.pt", map_location="cpu", weights_only=False)
    best_metrics = best_ckpt["val_metrics"]
    tuned_thresholds = {name: m["tuned_threshold"] for name, m in best_metrics["per_class"].items()}

    # These thresholds were tuned on VAL only. Phase 2 freezes them and applies
    # them unchanged to TEST -- tuning on test would invalidate the evaluation.
    with open(out_dir / "tuned_thresholds.json", "w") as f:
        json.dump(tuned_thresholds, f, indent=2)
    print(f"[saved] {out_dir / 'tuned_thresholds.json'}")

    summary = {
        "best_epoch": best_epoch,
        "select_metric": sel,
        "best_select_score": best_score,
        "best_val_macro_auroc": best_metrics["macro_auroc"],
        "best_val_stable_auroc": best_metrics["stable_auroc"],
        "best_val_macro_f1_at_0.5": best_metrics["macro_f1"],
        "best_val_macro_f1_tuned": best_metrics["macro_tuned_f1"],
        "tuned_thresholds": tuned_thresholds,
        "per_class_at_best_epoch": best_metrics["per_class"],
        "total_epochs_run": len(log_rows),
        "total_training_time_s": round(total_time, 1),
        "total_training_time_hms": time.strftime("%H:%M:%S", time.gmtime(total_time)),
        "img_size": args.img_size,
        "batch_size": args.batch,
        "lr": args.lr,
        "focal_gamma": args.gamma,
        "focal_alpha": loss_fn.alpha.tolist(),
        "n_params": n_params,
        "best_pt": str(out_dir / "best.pt"),
        "last_pt": str(out_dir / "last.pt"),
    }
    with open(out_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"  best epoch           : {best_epoch}  (selected on {sel})")
    print(f"  val macro AUROC      : {best_metrics['macro_auroc']:.4f}")
    print(f"  val stable AUROC     : {best_metrics['stable_auroc']:.4f}  "
          f"(weighted: fracture 50%, softtissue 30%, foreign_material 20%; bone_lesion excluded)")
    print(f"  val macro F1 @0.5    : {best_metrics['macro_f1']:.4f}")
    print(f"  val macro F1 @tuned  : {best_metrics['macro_tuned_f1']:.4f}")
    print(f"  total epochs run     : {len(log_rows)}")
    print(f"  training time        : {summary['total_training_time_hms']}")
    print("\n  per-class at best epoch:")
    for name, m in best_metrics["per_class"].items():
        auroc_s = f"{m['auroc']:.3f}" if not np.isnan(m["auroc"]) else "n/a"
        print(f"    {name:22s}: AUROC={auroc_s}  F1@0.5={m['f1']:.3f}  "
              f"F1@tuned={m['tuned_f1']:.3f} (t={m['tuned_threshold']:.2f}, n_pos={m['n_positive']})")
    print("=" * 70)
    print(f"\nNext: evaluate best.pt on the held-out TEST split (Phase 2 script),")
    print(f"applying the frozen thresholds in {out_dir / 'tuned_thresholds.json'}.")


if __name__ == "__main__":
    main()