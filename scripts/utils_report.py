"""
utils_report.py
----------------
Shared helper functions used by 00_sanity_check.py, 01_train_baseline.py and
02_evaluate_test.py.

Nothing in this file touches data/raw/. All paths are resolved relative to
the WristTrauma-AI project root, which is assumed to be the parent of the
`scripts/` folder this file lives in (override with the WRISTTRAUMA_ROOT
environment variable if you move things around).
"""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def project_root() -> Path:
    """Root of the WristTrauma-AI project (parent of this scripts/ folder)."""
    env_override = os.environ.get("WRISTTRAUMA_ROOT")
    if env_override:
        return Path(env_override).resolve()
    return Path(__file__).resolve().parent.parent


def dataset_yaml_path() -> Path:
    return project_root() / "data" / "processed" / "dataset.yaml"


def raw_data_path() -> Path:
    return project_root() / "data" / "raw"


def processed_data_path() -> Path:
    return project_root() / "data" / "processed"


def results_dir() -> Path:
    d = project_root() / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def models_dir() -> Path:
    d = project_root() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Environment / dependency setup
# ---------------------------------------------------------------------------

def ensure_ultralytics_installed() -> None:
    """Install ultralytics into the *current* interpreter's environment if missing.

    Run every script with the WristTrauma-AI venv's python
    (`venv\\Scripts\\python.exe script.py`) so this installs into that venv,
    not into a global Python.
    """
    try:
        import ultralytics  # noqa: F401

        return
    except ImportError:
        print("[setup] 'ultralytics' not found in this environment - installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "ultralytics"])
        print("[setup] ultralytics installed.")


def assert_raw_data_untouched_guard():
    """Defensive reminder: nothing in this pipeline should ever write into data/raw/.

    This is only a guard against accidental misuse (e.g. copy-pasting a path).
    It does not scan/modify data/raw/ itself.
    """
    raw = raw_data_path()
    if not raw.exists():
        print(f"[warn] Expected raw data dir not found at {raw} (continuing anyway).")


# ---------------------------------------------------------------------------
# GPU / environment reporting
# ---------------------------------------------------------------------------

def gpu_report() -> dict:
    """Collect GPU/CUDA/AMP-relevant environment info for the final report."""
    info = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            idx = 0
            props = torch.cuda.get_device_properties(idx)
            info["cuda_device_index"] = idx
            info["gpu_name"] = props.name
            info["gpu_total_memory_gb"] = round(props.total_memory / (1024 ** 3), 2)
            info["cuda_version"] = torch.version.cuda
            info["cudnn_version"] = torch.backends.cudnn.version()
            info["amp_supported"] = True
        else:
            info["amp_supported"] = False
    except ImportError:
        info["torch_version"] = None
        info["cuda_available"] = False
        info["amp_supported"] = False

    try:
        import ultralytics

        info["ultralytics_version"] = ultralytics.__version__
    except ImportError:
        info["ultralytics_version"] = None

    return info


def recommended_workers(cap: int = 8) -> int:
    """Pick a sensible number of DataLoader workers for this machine.

    Leaves a couple of cores free for the OS / main process, capped at `cap`
    because more workers rarely helps once the GPU (not the CPU decode/aug
    pipeline) is the bottleneck at 640px with a small/nano-scale model.
    """
    cpu_count = os.cpu_count() or 4
    return max(1, min(cap, cpu_count - 2))


def print_gpu_report(info: dict) -> None:
    print("=" * 70)
    print("ENVIRONMENT / GPU REPORT")
    print("=" * 70)
    for k, v in info.items():
        print(f"  {k:22s}: {v}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Dataset sanity checks (paths + rough counts from the handoff doc)
# ---------------------------------------------------------------------------

EXPECTED_IMAGE_COUNTS = {
    "train": 14170,
    "val": 3165,
    "test": 2992,
}

EXPECTED_CLASS_NAMES = [
    "boneanomaly",
    "bonelesion",
    "foreignbody",
    "fracture",
    "metal",
    "periostealreaction",
    "pronatorsign",
    "softtissue",
    "text",
]


def validate_dataset_layout() -> dict:
    """Verify processed/ has the expected structure, roughly the expected
    image/label counts, and that labels/images pair up 1:1 per split.

    Returns a dict summary; raises AssertionError on hard failures
    (missing dataset.yaml, missing split folders, class name/order mismatch).
    """
    yaml_path = dataset_yaml_path()
    assert yaml_path.exists(), f"dataset.yaml not found at {yaml_path}"

    import yaml as pyyaml

    with open(yaml_path, "r") as f:
        cfg = pyyaml.safe_load(f)

    names = cfg.get("names")
    if isinstance(names, dict):
        names_list = [names[i] for i in sorted(names.keys())]
    else:
        names_list = list(names)

    assert names_list == EXPECTED_CLASS_NAMES, (
        f"Class name/order mismatch.\nExpected: {EXPECTED_CLASS_NAMES}\nFound:    {names_list}"
    )

    summary = {"dataset_yaml": str(yaml_path), "classes": names_list, "splits": {}}

    processed = processed_data_path()
    for split in ("train", "val", "test"):
        img_dir = processed / "images" / split
        lbl_dir = processed / "labels" / split
        assert img_dir.exists(), f"Missing images dir: {img_dir}"
        assert lbl_dir.exists(), f"Missing labels dir: {lbl_dir}"

        img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in img_exts)
        labels = sorted(p for p in lbl_dir.iterdir() if p.suffix.lower() == ".txt")

        image_stems = {p.stem for p in images}
        label_stems = {p.stem for p in labels}
        missing_labels = image_stems - label_stems
        orphan_labels = label_stems - image_stems
        empty_labels = sum(1 for p in labels if p.stat().st_size == 0)

        n_images = len(images)
        expected = EXPECTED_IMAGE_COUNTS[split]
        pct_diff = abs(n_images - expected) / expected * 100 if expected else 0.0

        summary["splits"][split] = {
            "n_images": n_images,
            "n_label_files": len(labels),
            "expected_images": expected,
            "pct_diff_from_expected": round(pct_diff, 2),
            "images_missing_label_file": len(missing_labels),
            "orphan_label_files": len(orphan_labels),
            "empty_label_files": empty_labels,
        }

        assert len(missing_labels) == 0, (
            f"{split}: {len(missing_labels)} images have no matching label file "
            f"(YOLO expects an empty .txt for background images, not a missing file). "
            f"Examples: {list(missing_labels)[:5]}"
        )
        assert pct_diff < 5, (
            f"{split}: image count {n_images} differs from expected {expected} "
            f"by {pct_diff:.1f}% - re-check the split before training."
        )

    return summary


# ---------------------------------------------------------------------------
# Metrics extraction (works with ultralytics DetMetrics objects)
# ---------------------------------------------------------------------------

def per_class_metrics_table(metrics, class_names: list[str]) -> list[dict]:
    """Build a list of per-class metric dicts from an ultralytics DetMetrics
    object (as returned by model.val() / the val step of model.train()).
    """
    box = metrics.box
    rows = []
    # ap_class_index tells us which class each row of p/r/ap/ap50 corresponds to
    class_indices = list(box.ap_class_index)
    for row_i, cls_idx in enumerate(class_indices):
        rows.append(
            {
                "class_id": int(cls_idx),
                "class_name": class_names[int(cls_idx)],
                "precision": float(box.p[row_i]),
                "recall": float(box.r[row_i]),
                "mAP50": float(box.ap50[row_i]),
                "mAP50-95": float(box.ap[row_i]),
            }
        )
    # classes with zero instances in this split won't appear in ap_class_index;
    # list them explicitly with NaN so the table always has all 9 rows
    present = {r["class_id"] for r in rows}
    for cls_idx, name in enumerate(class_names):
        if cls_idx not in present:
            rows.append(
                {
                    "class_id": cls_idx,
                    "class_name": name,
                    "precision": float("nan"),
                    "recall": float("nan"),
                    "mAP50": float("nan"),
                    "mAP50-95": float("nan"),
                }
            )
    rows.sort(key=lambda r: r["class_id"])
    return rows


def overall_metrics_dict(metrics) -> dict:
    box = metrics.box
    return {
        "precision_mean": float(box.mp),
        "recall_mean": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50-95": float(box.map),
    }


def save_metrics_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[saved] {path}")


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"[saved] {path}")


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
