"""
00_sanity_check.py
-------------------
Run this FIRST. It verifies, in order:

  1. ultralytics is importable in this venv (installs it if missing)
  2. CUDA / the RTX 5060 is visible to torch, and AMP is usable
  3. data/processed/dataset.yaml exists, class list/order matches spec exactly,
     and every split's images/labels are paired correctly (no missing label
     files, counts roughly match the documented split sizes)
  4. YOLO26s weights load correctly
  5. A tiny 1-epoch, small-batch smoke-test train + val actually runs end to
     end on the REAL data (not synthetic data), so any path/format/CUDA
     problem shows up now instead of after hours of real training

data/raw/ is never opened or written to by this script.

Usage (from an activated WristTrauma-AI venv):
    python scripts\\00_sanity_check.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils_report as U


def main():
    print("\n### STEP 1/5: dependency check ###")
    U.ensure_ultralytics_installed()
    U.assert_raw_data_untouched_guard()

    print("\n### STEP 2/5: GPU / CUDA / AMP check ###")
    env = U.gpu_report()
    U.print_gpu_report(env)
    if not env.get("cuda_available"):
        print(
            "\n[FATAL] CUDA is not available to torch in this environment.\n"
            "        Check that: (a) you're running the WristTrauma-AI venv's python,\n"
            "        (b) it has a CUDA-enabled torch build installed (not the CPU-only\n"
            "        wheel), and (c) `nvidia-smi` shows the RTX 5060 outside Python."
        )
        sys.exit(1)
    print(f"[ok] GPU detected: {env['gpu_name']} ({env['gpu_total_memory_gb']} GB)")

    print("\n### STEP 3/5: dataset layout / label pairing check ###")
    try:
        summary = U.validate_dataset_layout()
    except AssertionError as e:
        print(f"\n[FATAL] Dataset validation failed:\n{e}")
        sys.exit(1)

    print(f"[ok] dataset.yaml classes match spec exactly (9 classes, correct order)")
    for split, s in summary["splits"].items():
        print(
            f"  {split:5s}: {s['n_images']:6d} images | {s['n_label_files']:6d} labels "
            f"| empty labels: {s['empty_label_files']:3d} | "
            f"diff from expected: {s['pct_diff_from_expected']}%"
        )
    U.save_json(summary, U.results_dir() / "00_sanity_check" / "dataset_layout_summary.json")

    print("\n### STEP 4/5: load YOLO26s pretrained weights ###")
    from ultralytics import YOLO

    model = YOLO("yolo26s.pt")
    print(f"[ok] Loaded model: {model.model.__class__.__name__}, task={model.task}")

    print("\n### STEP 5/5: 1-epoch smoke-test train + val on REAL data ###")
    print("(small batch, small subset behaviour is NOT used - this runs on the")
    print(" real train/val split for one epoch purely to catch pipeline bugs)")

    t0 = time.time()
    results = model.train(
        data=str(U.dataset_yaml_path()),
        epochs=1,
        imgsz=640,
        batch=-1,               # deliberately small/safe for the smoke test
        device=0,
        workers=U.recommended_workers(),
        amp=True,
        seed=0,
        deterministic=True,
        project=str(U.results_dir()),
        name="00_sanity_check_run",
        exist_ok=True,
        verbose=True,
        plots=False,           # skip plot generation, this run is throwaway
    )
    elapsed = time.time() - t0

    metrics = model.val(
        data=str(U.dataset_yaml_path()),
        imgsz=640,
        device=0,
        split="val",
        project=str(U.results_dir()),
        name="00_sanity_check_val",
        exist_ok=True,
        plots=False,
    )

    overall = U.overall_metrics_dict(metrics)
    print("\n" + "=" * 70)
    print("SANITY CHECK PASSED")
    print("=" * 70)
    print(f"  1-epoch smoke train+val wall time : {elapsed:.1f}s")
    print(f"  (numbers below are meaningless after 1 epoch - this only proves")
    print(f"   the pipeline runs end-to-end, not that the model is trained)")
    print(f"  val precision (mean)  : {overall['precision_mean']:.4f}")
    print(f"  val recall (mean)     : {overall['recall_mean']:.4f}")
    print(f"  val mAP50             : {overall['mAP50']:.4f}")
    print(f"  val mAP50-95          : {overall['mAP50-95']:.4f}")
    print("=" * 70)
    print(
        "\nEverything checks out. You can now run:\n"
        "    python scripts\\01_train_baseline.py\n"
    )


if __name__ == "__main__":
    main()
