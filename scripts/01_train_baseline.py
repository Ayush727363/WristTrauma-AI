"""
01_train_baseline.py
---------------------
YOLO26s training run on GRAZPEDWRI-DX - used both for the 640px clean
baseline AND for performance experiments (e.g. 512px, fixed batch, compile)
by passing different CLI flags. Whatever you pass on the command line is
exactly what gets logged, printed, and saved to training_summary.json - there
is no hardcoded value inside main() that can silently diverge from the flags.

Design choices:
  - No class weighting, no focal loss, no extra/aggressive augmentation on
    top of what ultralytics ships as its standard training recipe. Only
    training-loop performance knobs are exposed here (imgsz, batch, workers,
    amp, cache, compile, channels_last, determinism) - never accuracy knobs.
  - batch: pass a float in (0,1) for AutoBatch targeting that fraction of
    free VRAM (default 0.80, i.e. ~80%, vs ultralytics' stock 60%), '-1' for
    stock 60%-target AutoBatch, or a plain int (e.g. '8') to force a fixed
    batch size. Fixed batch is recommended once you've watched nvidia-smi
    during a run and know AutoBatch's estimate is off for your setup.
  - amp=True -> mixed precision (fp16/bf16 autocast) training.
  - cache: 'disk' (default) caches decoded images to disk, not RAM - safer
    with 15.7GB system RAM shared across OS/process/workers than cache=True
    (full RAM cache). 'none' disables caching entirely (useful for A/B
    testing against 'disk' when system RAM is already under external
    pressure from other apps). 'ram' forces full RAM caching if you've
    confirmed you have headroom - not recommended on this machine by default.
  - compile: off by default. torch.compile can meaningfully speed up steady-
    state training, but 'reduce-overhead' uses CUDA graphs which are more
    likely to misbehave with detection models (variable per-image box counts,
    mosaic/affine augmentation) and cost extra VRAM on an 8GB card. Always
    smoke-test any non-default compile setting with a couple epochs before
    committing to a full run.
  - channels_last: off by default, safe to enable (NHWC memory format speeds
    up Tensor Core convolutions on this GPU with no accuracy impact).
  - workers: capped well below a desktop default because Windows spawns
    (not forks) worker processes, and 15.7GB total RAM has to cover the OS,
    this process, any disk-cache I/O buffers, and all workers at once.
  - deterministic=False + cudnn.benchmark=True: trades exact bit-for-bit
    reproducibility for throughput by letting cuDNN autotune/pick the fastest
    convolution algorithms for the fixed input size. seed is still set, so
    initialization/shuffling remain seeded - only low-level GPU kernel
    selection is no longer forced deterministic.
  - TF32 matmul/cudnn enabled: free throughput on Tensor Core GPUs, negligible
    precision impact, standard practice for training (not inference/eval).
  - pin_memory for the training dataloader is handled internally by
    ultralytics whenever device != 'cpu' - there's no separate flag to set.
  - optimizer="auto" passed explicitly (also ultralytics' own default) so
    it's visible in args.yaml without changing behavior.
  - Ultralytics writes the FULL resolved training config to
    <run_dir>/args.yaml automatically; a copy is also placed in models/
    next to the final weights so the exact config that produced them
    travels with the weights.
  - The patient-level train/val/test split and all preprocessing are
    untouched - this script only touches training-loop performance knobs.

Run AFTER 00_sanity_check.py has passed. Nothing here executes on import -
only when run directly.

Usage examples:
    REM 640px clean baseline (AutoBatch @ ~80% VRAM target)
    python scripts\\01_train_baseline.py --name baseline_yolo26s_640

    REM 512px performance experiment, fixed batch, cache disabled, compile off
    python scripts\\01_train_baseline.py --imgsz 512 --batch 8 --workers 4 ^
        --cache none --compile false --channels-last --name yolo26s_512_exp1 --epochs 150

    REM quick smoke test of a config before committing to the full run
    python scripts\\01_train_baseline.py --imgsz 512 --batch 8 --epochs 3 ^
        --compile default --channels-last --name yolo26s_512_smoketest
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils_report as U

VALID_COMPILE_CHOICES = ["false", "true", "default", "reduce-overhead", "max-autotune-no-cudagraphs"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=150, help="Max training epochs.")
    p.add_argument("--imgsz", type=int, default=640, help="Train/val image size.")
    p.add_argument(
        "--batch",
        type=str,
        default="0.80",
        help="Batch size. A float in (0,1) (default '0.80') = AutoBatch targeting that "
        "fraction of free VRAM. '-1' = AutoBatch's stock 60%% target. A plain int forces "
        "a fixed batch size (e.g. '8').",
    )
    p.add_argument(
        "--patience", type=int, default=100, help="Early-stop patience (epochs with no val improvement)."
    )
    p.add_argument("--name", type=str, default="baseline_yolo26s_640", help="Run name under results/ and models/.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override dataloader worker count. Default: auto-sized (capped at 6) for "
        "this machine's CPU count and RAM (see utils_report.recommended_workers).",
    )
    p.add_argument(
        "--cache",
        type=str,
        default="disk",
        choices=["disk", "ram", "none"],
        help="Image caching mode. 'disk' (default, safe on 15.7GB system RAM), "
        "'ram' (full RAM cache - only if you've confirmed headroom), "
        "'none' (no caching - useful for A/B testing against 'disk').",
    )
    p.add_argument(
        "--compile",
        type=str,
        default="false",
        choices=VALID_COMPILE_CHOICES,
        help="torch.compile mode. 'false' (default, safest), 'true'/'default' (standard "
        "compile), 'reduce-overhead' (CUDA graphs - smoke-test first, risk of instability/"
        "extra VRAM use with detection models), 'max-autotune-no-cudagraphs'.",
    )
    p.add_argument(
        "--channels-last",
        action="store_true",
        default=False,
        help="Enable NHWC (channels_last) memory format - safe Tensor Core speedup.",
    )
    return p.parse_args()


def _parse_batch(raw: str):
    """'-1' or an int string -> int. A float string in (0,1) -> float (AutoBatch fraction)."""
    val = float(raw)
    if val == -1:
        return -1
    if 0 < val < 1:
        return val
    return int(val)


def _parse_cache(raw: str):
    return {"disk": "disk", "ram": True, "none": False}[raw]


def _parse_compile(raw: str):
    if raw == "false":
        return False
    if raw == "true":
        return True
    return raw  # "default" / "reduce-overhead" / "max-autotune-no-cudagraphs" pass through as-is


def main():
    args = parse_args()
    batch = _parse_batch(args.batch)
    cache = _parse_cache(args.cache)
    compile_mode = _parse_compile(args.compile)

    U.ensure_ultralytics_installed()
    U.assert_raw_data_untouched_guard()

    env = U.gpu_report()
    U.print_gpu_report(env)
    if not env.get("cuda_available"):
        print("[FATAL] CUDA not available - run 00_sanity_check.py first to diagnose.")
        sys.exit(1)

    print("\nValidating dataset layout before committing to a full run...")
    try:
        U.validate_dataset_layout()
    except AssertionError as e:
        print(f"[FATAL] Dataset validation failed:\n{e}")
        sys.exit(1)
    print("[ok] dataset layout valid.\n")

    from ultralytics import YOLO
    import torch

    # deterministic=False -> let cuDNN autotune + pick fastest conv algorithms
    # for our fixed input size (this only matters for GPU kernel selection;
    # seed below still seeds init/shuffling).
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Windows uses 'spawn' (not 'fork') for DataLoader workers, so each worker
    # is a fresh Python process (re-imports torch/ultralytics/cv2). Combined
    # with only 15.7GB system RAM shared across the OS, this process, any
    # disk-cache I/O buffers, and worker processes, a desktop-sized worker
    # count (e.g. 8+) risks RAM pressure/paging rather than helping once the
    # GPU - not CPU decode/augment - is the bottleneck. Cap at 6 by default.
    workers = args.workers if args.workers is not None else U.recommended_workers(cap=6)

    if isinstance(batch, float):
        batch_display = f"auto (AutoBatch, target ~{int(batch * 100)}% free VRAM)"
    elif batch == -1:
        batch_display = "auto (AutoBatch, stock ~60% free VRAM)"
    else:
        batch_display = str(batch)

    cache_display = {"disk": "disk", True: "ram (full)", False: "none"}[cache]
    compile_display = {False: "off", True: "on (default mode)"}.get(compile_mode, compile_mode)

    print(f"Config for this run:")
    print(f"  model        : yolo26s.pt")
    print(f"  data         : {U.dataset_yaml_path()}")
    print(f"  epochs       : {args.epochs}")
    print(f"  imgsz        : {args.imgsz}")
    print(f"  batch        : {batch_display}")
    print(f"  device       : 0 ({env.get('gpu_name')})")
    print(f"  workers      : {workers} (Windows-safe cap for 15.7GB system RAM)")
    print(f"  amp          : True")
    print(f"  cache        : {cache_display}")
    print(f"  compile      : {compile_display}")
    print(f"  channels_last: {args.channels_last}")
    print(f"  optimizer    : auto")
    print(f"  seed         : {args.seed}")
    print(f"  deterministic: False (cudnn.benchmark=True, TF32 enabled, for throughput)")
    print(f"  results dir  : {U.results_dir() / args.name}\n")

    model = YOLO("yolo26s.pt")

    t0 = time.time()
    model.train(
        data=str(U.dataset_yaml_path()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch,
        device=0,
        workers=workers,
        amp=True,
        cache=cache,
        compile=compile_mode,
        channels_last=args.channels_last,
        optimizer="auto",          # explicit for visibility in args.yaml; same as ultralytics default
        seed=args.seed,
        deterministic=False,       # throughput: allow cuDNN autotuning/non-deterministic kernels
        patience=args.patience,
        project=str(U.results_dir()),
        name=args.name,
        exist_ok=False,             # never silently overwrite a previous run with the same name
        plots=True,                  # training curves, confusion matrix, PR curves, etc.
        save=True,
        val=True,                    # validate every epoch - required for correct best.pt / patience behavior
        verbose=True,
        # Deliberately NOT set: cls weighting, fl_gamma, extra mixup/copy_paste
        # beyond the framework default recipe - this is the clean baseline.
        # Dataset/splits/preprocessing are untouched by this script.
    )
    elapsed = time.time() - t0

    save_dir = Path(model.trainer.save_dir)
    weights_dir = save_dir / "weights"
    best_pt = weights_dir / "best.pt"
    last_pt = weights_dir / "last.pt"
    results_csv = save_dir / "results.csv"
    args_yaml = save_dir / "args.yaml"

    # ---- figure out the best epoch from results.csv (max mAP50-95) ----
    best_epoch, best_map5095, best_map50 = None, None, None
    if results_csv.exists():
        import csv as _csv

        with open(results_csv, "r") as f:
            reader = _csv.DictReader(f)
            rows = list(reader)
        map_key = next((k for k in rows[0].keys() if "mAP50-95" in k), None)
        map50_key = next((k for k in rows[0].keys() if "mAP50" in k and "95" not in k), None)
        epoch_key = next((k for k in rows[0].keys() if k.strip().lower() == "epoch"), None)
        if map_key:
            best_row = max(rows, key=lambda r: float(r[map_key]))
            best_epoch = int(float(best_row[epoch_key])) if epoch_key else None
            best_map5095 = float(best_row[map_key])
            best_map50 = float(best_row[map50_key]) if map50_key else None

    # ---- copy final artifacts into models/ so they're easy to find ----
    models_out = U.models_dir() / args.name
    models_out.mkdir(parents=True, exist_ok=True)
    if best_pt.exists():
        shutil.copy2(best_pt, models_out / "best.pt")
    if last_pt.exists():
        shutil.copy2(last_pt, models_out / "last.pt")
    if args_yaml.exists():
        shutil.copy2(args_yaml, models_out / "train_args.yaml")  # exact config used
    if results_csv.exists():
        shutil.copy2(results_csv, models_out / "results.csv")

    summary = {
        "run_dir": str(save_dir),
        "best_pt": str(models_out / "best.pt"),
        "last_pt": str(models_out / "last.pt"),
        "config_used": str(models_out / "train_args.yaml"),
        "training_time_seconds": round(elapsed, 1),
        "training_time_hms": time.strftime("%H:%M:%S", time.gmtime(elapsed)),
        "best_epoch": best_epoch,
        "best_val_mAP50": best_map50,
        "best_val_mAP50-95": best_map5095,
        "gpu": env.get("gpu_name"),
        "epochs_requested": args.epochs,
        "imgsz": args.imgsz,
        "batch_setting": batch_display,
        "cache": cache_display,
        "compile": compile_display,
        "channels_last": args.channels_last,
        "deterministic": False,
        "workers": workers,
    }
    U.save_json(summary, U.models_dir() / args.name / "training_summary.json")

    print("\n" + "=" * 70)
    print("TRAINING RUN COMPLETE")
    print("=" * 70)
    for k, v in summary.items():
        print(f"  {k:22s}: {v}")
    print("=" * 70)
    print(
        f"\nTraining curves, confusion matrix and validation prediction samples\n"
        f"are saved under: {save_dir}\n"
        f"\nNext: python scripts\\02_evaluate_test.py --weights \"{models_out / 'best.pt'}\"\n"
    )


if __name__ == "__main__":
    main()