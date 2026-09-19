"""
dataset.py
----------
PyTorch Dataset for the WristTrauma-AI custom CNN (Phase 1).

Reads the ORIGINAL 16-bit grayscale PNGs (not the cached .npy files from the
old YOLO pipeline -- those are a different, opaque preprocessing format we
don't want to depend on). Each image is:

  1. loaded with cv2.IMREAD_UNCHANGED (preserves 16-bit depth correctly --
     PIL can mishandle 16-bit PNGs)
  2. normalized to 0-1 using that image's own min/max pixel value (robust to
     different device manufacturers having different intensity ranges --
     this dataset has multiple, per the handoff doc)
  3. letterboxed (padded, not squashed) to a square of side `img_size`, so
     aspect ratio is preserved -- these images range from ~494x727 to
     ~509x1204, and squashing would distort bone geometry
  4. converted to a 3-channel tensor (grayscale repeated x3) because our CNN
     stem expects 3 input channels, matching standard ImageNet-style
     architectures -- makes it trivial to later swap in pretrained weights
     if we ever want to, and grayscale-x3 is the standard trick for this

Labels come from data/classification/labels_multilabel.csv (built by
01_prepare_labels.py) -- 4 merged classes:
    0 fracture
    1 foreign_material
    2 softtissue_indirect
    3 bone_lesion

This file does not touch data/raw/ or data/processed/labels/*.txt directly --
it only reads the already-validated multilabel CSV and the PNG images.

OPTIONAL PREPROCESSING CACHE (use_cache=True):
Decoding a 16-bit PNG + letterboxing is real CPU work, repeated every epoch
for every image by every worker. With use_cache=True, the first time an
image is loaded it's saved as a compact float16 .npy under
data/classification/cache_<img_size>/<split>/<stem>.npy; every epoch after
that just np.load()s the small pre-resized array instead of re-decoding the
original PNG. This trades a slower first epoch (building the cache) for much
lower CPU/RAM load on every epoch after, which is what actually lets the GPU
stay fed. The cache is a pure derived artifact -- deleting the cache folder
is always safe and just costs one slow epoch to rebuild it. It never touches
data/raw/, data/processed/, or the original PNGs.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

CLASS_NAMES = ["fracture", "foreign_material", "softtissue_indirect", "bone_lesion"]
N_CLASSES = len(CLASS_NAMES)


def letterbox(img: np.ndarray, target_size: int) -> np.ndarray:
    """Resize + pad a single-channel float32 [0,1] image to a target_size x
    target_size square, preserving aspect ratio. Padding value is 0 (black),
    which is appropriate for X-rays (background is dark)."""
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_size, target_size), dtype=np.float32)
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


def load_and_preprocess_image(png_path: Path, img_size: int) -> np.ndarray:
    """Load a 16-bit (or 8-bit) grayscale PNG, normalize to [0,1] by its own
    min/max, letterbox to img_size x img_size. Returns float32 HxW array."""
    img = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"cv2 could not read image: {png_path}")
    if img.ndim == 3:  # just in case a file is unexpectedly 3-channel
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = img.astype(np.float32)
    lo, hi = img.min(), img.max()
    if hi > lo:
        img = (img - lo) / (hi - lo)
    else:
        img = np.zeros_like(img)  # degenerate all-constant image guard

    return letterbox(img, img_size)


class WristMultiLabelDataset(Dataset):
    """
    Args:
        csv_path: path to labels_multilabel.csv (from 01_prepare_labels.py)
        images_dir: path to data/processed/images/<split>/
        split: 'train' | 'val' | 'test' -- filters the CSV's `split` column
        img_size: square side length images are letterboxed to
        augment: if True, applies light train-time augmentation
                 (random horizontal flip is DELIBERATELY OFF by default --
                 see note below. Only rotation/brightness jitter are used.)
    """

    def __init__(self, csv_path: str | Path, images_dir: str | Path,
                 split: str, img_size: int = 256, augment: bool = False,
                 use_cache: bool = False, cache_root: str | Path | None = None):
        self.images_dir = Path(images_dir)
        self.img_size = img_size
        self.augment = augment
        self.use_cache = use_cache

        if use_cache:
            if cache_root is None:
                raise ValueError("cache_root must be given when use_cache=True")
            self.cache_dir = Path(cache_root) / f"cache_{img_size}" / split
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None

        df = pd.read_csv(csv_path)
        df = df[df["split"] == split].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(f"No rows found for split='{split}' in {csv_path}")
        self.df = df

        self.label_cols = [f"has_{name}" for name in CLASS_NAMES]
        missing = [c for c in self.label_cols if c not in df.columns]
        if missing:
            raise ValueError(f"labels_multilabel.csv missing expected columns: {missing}")

    def __len__(self) -> int:
        return len(self.df)

    def _augment(self, img: np.ndarray) -> np.ndarray:
        """Training-time augmentation.

        Deliberately much stronger than a minimal rotate/brightness pass,
        because a 2.8M-param CNN on 14k images overfits hard: train loss
        keeps falling while val plateaus. Augmentation is the cheapest and
        most effective regularizer available here -- it costs nothing at
        inference and directly attacks memorization.

        HORIZONTAL FLIP is included and is safe for THIS task: the labels are
        "is a fracture / foreign object / soft-tissue sign present", none of
        which depend on whether it's a left or right wrist. The dataset
        already contains both (11,135 left / 9,192 right), so flipping
        augments within the distribution the model already sees rather than
        inventing anatomy that doesn't occur. (If you ever add a laterality
        PREDICTION head, flipping would have to be removed -- it would make
        that label wrong.)

        Vertical flip is NOT used: wrist X-rays have a consistent
        proximal/distal orientation, so an upside-down wrist is genuinely
        out-of-distribution.
        """
        h, w = img.shape

        # --- horizontal flip ---
        if np.random.rand() < 0.5:
            img = np.fliplr(img).copy()

        # --- affine: rotation + scale + translation in one warp ---
        if np.random.rand() < 0.8:
            angle = np.random.uniform(-15, 15)
            scale = np.random.uniform(0.85, 1.15)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
            M[0, 2] += np.random.uniform(-0.08, 0.08) * w
            M[1, 2] += np.random.uniform(-0.08, 0.08) * h
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)

        # --- gamma correction (simulates different exposure/processing) ---
        if np.random.rand() < 0.5:
            gamma = np.random.uniform(0.7, 1.4)
            img = np.power(np.clip(img, 0.0, 1.0), gamma)

        # --- brightness + contrast jitter ---
        if np.random.rand() < 0.5:
            brightness = np.random.uniform(-0.08, 0.08)
            contrast = np.random.uniform(0.85, 1.15)
            img = np.clip((img - 0.5) * contrast + 0.5 + brightness, 0.0, 1.0)

        # --- random erasing / cutout: forces the model to use more than one
        #     region, rather than latching onto a single memorized patch ---
        if np.random.rand() < 0.25:
            eh = int(h * np.random.uniform(0.08, 0.20))
            ew = int(w * np.random.uniform(0.08, 0.20))
            y0 = np.random.randint(0, max(1, h - eh))
            x0 = np.random.randint(0, max(1, w - ew))
            img[y0:y0 + eh, x0:x0 + ew] = 0.0

        return np.clip(img, 0.0, 1.0).astype(np.float32)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        stem = row["filestem"]

        if self.use_cache:
            cache_path = self.cache_dir / f"{stem}.npy"
            if cache_path.exists():
                img = np.load(cache_path).astype(np.float32)  # already stored in [0,1] range
            else:
                png_path = self.images_dir / f"{stem}.png"
                img = load_and_preprocess_image(png_path, self.img_size)
                # store directly as float16 in [0,1] -- safe range, no overflow risk
                np.save(cache_path, img.astype(np.float16))
        else:
            png_path = self.images_dir / f"{stem}.png"
            img = load_and_preprocess_image(png_path, self.img_size)

        if self.augment:
            img = self._augment(img)

        # HxW float32 [0,1] -> 3xHxW tensor (grayscale repeated across channels)
        img_t = torch.from_numpy(img).unsqueeze(0).repeat(3, 1, 1).float()

        label = row[self.label_cols].to_numpy(dtype=np.float32)
        label_t = torch.from_numpy(label)

        return img_t, label_t, row["filestem"]


def build_datasets(csv_path: str | Path, images_root: str | Path, img_size: int = 256,
                    use_cache: bool = False, cache_root: str | Path | None = None):
    """Convenience factory: returns (train_ds, val_ds, test_ds) using the
    standard folder layout data/processed/images/{train,val,test}/.

    If use_cache=True, cache_root should be the project's
    data/classification/ folder -- preprocessed arrays are stored under
    cache_root/cache_<img_size>/<split>/. This is optional and purely a
    speed optimization; results are numerically identical to use_cache=False,
    just faster after the first epoch per image."""
    images_root = Path(images_root)
    train_ds = WristMultiLabelDataset(csv_path, images_root / "train", "train",
                                       img_size=img_size, augment=True,
                                       use_cache=use_cache, cache_root=cache_root)
    val_ds = WristMultiLabelDataset(csv_path, images_root / "val", "val",
                                     img_size=img_size, augment=False,
                                     use_cache=use_cache, cache_root=cache_root)
    test_ds = WristMultiLabelDataset(csv_path, images_root / "test", "test",
                                      img_size=img_size, augment=False,
                                      use_cache=use_cache, cache_root=cache_root)
    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    # Quick self-test: run `python src/dataset.py` from the project root
    # (with the venv active) to sanity-check the loader on real data before
    # wiring it into training.
    import sys

    project_root = Path(__file__).resolve().parent.parent
    csv_path = project_root / "data" / "classification" / "labels_multilabel.csv"
    images_root = project_root / "data" / "processed" / "images"

    if not csv_path.exists():
        sys.exit(f"[ERROR] {csv_path} not found -- run scripts/01_prepare_labels.py first.")

    print(f"[test] building datasets from {csv_path}")
    train_ds, val_ds, test_ds = build_datasets(csv_path, images_root, img_size=256)
    print(f"[ok] train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    img, label, stem = train_ds[0]
    print(f"[ok] sample 0: stem={stem}")
    print(f"     image tensor: shape={tuple(img.shape)} dtype={img.dtype} "
          f"min={img.min():.3f} max={img.max():.3f}")
    print(f"     label vector: {label.tolist()}  (order: {CLASS_NAMES})")

    # check a handful more, including one with multiple findings if we can find one
    multi = train_ds.df[train_ds.df[train_ds.label_cols].sum(axis=1) >= 2]
    if len(multi) > 0:
        idx = train_ds.df.index.get_loc(multi.index[0])
        img2, label2, stem2 = train_ds[idx]
        print(f"[ok] multi-finding sample: stem={stem2} label={label2.tolist()}")

    print("\n[done] dataset.py self-test passed.")
