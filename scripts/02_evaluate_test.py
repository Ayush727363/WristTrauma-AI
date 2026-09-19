"""
02_evaluate_test.py
--------------------
Final, one-time evaluation of the trained baseline on the held-out TEST split
(data/processed/images/test + labels/test). This split is never used for
training or for picking the best checkpoint - only for reporting the final
numbers.

Produces, under results/<name>/:
  - overall_test_metrics.json          (precision, recall, mAP50, mAP50-95)
  - per_class_test_metrics.csv         (same 4 metrics, per class)
  - confusion_matrix.png               (written automatically by ultralytics, plots=True)
  - a handful of representative annotated test predictions (images)
  - final_report.md                    (human-readable summary of the whole run)

Usage:
    python scripts\\02_evaluate_test.py --weights "models\\baseline_yolo26s_640\\best.pt"
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils_report as U


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, required=True, help="Path to best.pt from training.")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--name", type=str, default="test_evaluation")
    p.add_argument("--n_sample_predictions", type=int, default=20, help="How many test images to save annotated predictions for.")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for the sample prediction images.")
    return p.parse_args()


def main():
    args = parse_args()
    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"[FATAL] weights file not found: {weights_path}")
        sys.exit(1)

    U.ensure_ultralytics_installed()
    U.assert_raw_data_untouched_guard()

    env = U.gpu_report()
    U.print_gpu_report(env)
    if not env.get("cuda_available"):
        print("[FATAL] CUDA not available - run 00_sanity_check.py first to diagnose.")
        sys.exit(1)

    from ultralytics import YOLO

    print(f"\nLoading trained model: {weights_path}")
    model = YOLO(str(weights_path))
    class_names = [model.names[i] for i in sorted(model.names.keys())]

    print("\n### Evaluating on the untouched TEST split ###")
    t0 = time.time()
    metrics = model.val(
        data=str(U.dataset_yaml_path()),
        imgsz=args.imgsz,
        device=0,
        split="test",
        project=str(U.results_dir()),
        name=args.name,
        exist_ok=True,
        plots=True,        # writes confusion_matrix.png, PR curves, etc. into the run dir
        save_json=True,    # COCO-style predictions json, useful for further analysis
    )
    eval_elapsed = time.time() - t0
    run_dir = Path(metrics.save_dir)

    overall = U.overall_metrics_dict(metrics)
    per_class = U.per_class_metrics_table(metrics, class_names)

    U.save_json(overall, run_dir / "overall_test_metrics.json")
    U.save_metrics_csv(per_class, run_dir / "per_class_test_metrics.csv")

    # ---- representative annotated predictions on a random sample of test images ----
    test_img_dir = U.processed_data_path() / "images" / "test"
    all_test_images = sorted(p for p in test_img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    random.seed(0)
    sample = random.sample(all_test_images, k=min(args.n_sample_predictions, len(all_test_images)))

    print(f"\n### Saving {len(sample)} representative annotated test predictions ###")
    model.predict(
        source=[str(p) for p in sample],
        imgsz=args.imgsz,
        device=0,
        conf=args.conf,
        save=True,
        project=str(U.results_dir()),
        name=f"{args.name}_sample_predictions",
        exist_ok=True,
    )

    # ---- pull training-time info back in for a single combined report ----
    train_summary_path = weights_path.parent / "training_summary.json"
    train_summary = {}
    if train_summary_path.exists():
        with open(train_summary_path) as f:
            train_summary = json.load(f)

    report = {
        "weights_evaluated": str(weights_path),
        "test_eval_run_dir": str(run_dir),
        "test_eval_wall_time_seconds": round(eval_elapsed, 1),
        "gpu": env.get("gpu_name"),
        "overall_test_metrics": overall,
        "per_class_test_metrics": per_class,
        "training_summary": train_summary,
    }
    U.save_json(report, run_dir / "final_report.json")

    # ---- human-readable markdown report ----
    lines = []
    lines.append("# GRAZPEDWRI-DX - YOLO26s Baseline: Final Report\n")
    lines.append("## Training")
    if train_summary:
        lines.append(f"- Best epoch: {train_summary.get('best_epoch')}")
        lines.append(f"- Best model path: `{train_summary.get('best_pt')}`")
        lines.append(f"- Exact config used: `{train_summary.get('config_used')}`")
        lines.append(f"- Training time: {train_summary.get('training_time_hms')} ({train_summary.get('training_time_seconds')}s)")
        lines.append(f"- GPU: {train_summary.get('gpu')}")
        lines.append(f"- Epochs requested: {train_summary.get('epochs_requested')}, imgsz: {train_summary.get('imgsz')}, batch: {train_summary.get('batch_setting')}")
        lines.append(f"- Best val mAP50: {train_summary.get('best_val_mAP50')}")
        lines.append(f"- Best val mAP50-95: {train_summary.get('best_val_mAP50-95')}")
    else:
        lines.append("_(training_summary.json not found next to weights - run 01_train_baseline.py's output alongside this evaluation to include it)_")
    lines.append("")
    lines.append("## Test-set evaluation (held-out, never used in training)")
    lines.append(f"- Precision (mean): {overall['precision_mean']:.4f}")
    lines.append(f"- Recall (mean): {overall['recall_mean']:.4f}")
    lines.append(f"- mAP50: {overall['mAP50']:.4f}")
    lines.append(f"- mAP50-95: {overall['mAP50-95']:.4f}")
    lines.append("")
    lines.append("### Per-class test metrics")
    lines.append("| Class | Precision | Recall | mAP50 | mAP50-95 |")
    lines.append("|---|---|---|---|---|")
    for r in per_class:
        def fmt(x):
            return "n/a" if x != x else f"{x:.4f}"  # x != x checks for NaN
        lines.append(f"| {r['class_name']} | {fmt(r['precision'])} | {fmt(r['recall'])} | {fmt(r['mAP50'])} | {fmt(r['mAP50-95'])} |")
    lines.append("")
    lines.append(f"Confusion matrix, PR curves, and other plots: `{run_dir}`")
    lines.append(f"Sample annotated test predictions: `{U.results_dir() / (args.name + '_sample_predictions')}`")

    report_md = run_dir / "final_report.md"
    with open(report_md, "w") as f:
        f.write("\n".join(lines))
    print(f"[saved] {report_md}")

    print("\n" + "=" * 70)
    print("TEST-SET EVALUATION COMPLETE")
    print("=" * 70)
    print(f"  Precision (mean) : {overall['precision_mean']:.4f}")
    print(f"  Recall (mean)    : {overall['recall_mean']:.4f}")
    print(f"  mAP50            : {overall['mAP50']:.4f}")
    print(f"  mAP50-95         : {overall['mAP50-95']:.4f}")
    print(f"\n  Full report: {report_md}")
    print("=" * 70)


if __name__ == "__main__":
    main()
