"""
cam_utils.py
------------
Class Activation Mapping (CAM) utilities for WristNet.

Given a trained model and an input image, produces, per class:
  - a heatmap (resized to the input image size, normalized to [0,1])
  - a bounding box derived from thresholding that heatmap (or None if the
    class's predicted probability is below its tuned threshold -- i.e. we
    only draw a box for a class the model actually thinks is present)

This is intentionally simple and inspectable:
    CAM = sum_c( classifier.weight[class, c] * feature_map[c] )
then upsampled from the feature map's spatial size (8x8 for a 256 input,
see model.py) to the full image size with bilinear interpolation.

Box extraction: threshold the normalized heatmap at `box_threshold`
(default 0.5 -- i.e. keep pixels above 50% of that heatmap's own peak
activation), take the largest connected component, and return its
axis-aligned bounding box. This is standard weakly-supervised localization
practice (this is literally the original CAM paper's approach, Zhou et al.
2016) -- not a novel trick, but the correct/expected one to cite.

Does not touch data/raw/ or any label file.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def get_probs_and_cams(model, img_tensor: torch.Tensor, device) -> tuple[np.ndarray, np.ndarray]:
    """
    img_tensor: single image, shape (3, H, W) (NOT batched)
    Returns:
        probs: (n_classes,) sigmoid probabilities
        cams: (n_classes, H, W) raw (un-normalized) CAM heatmaps, already
              upsampled to the input image's H, W
    """
    model.eval()
    x = img_tensor.unsqueeze(0).to(device)  # 1x3xHxW
    logits = model(x)  # 1 x n_classes
    probs = torch.sigmoid(logits)[0].cpu().numpy()

    H, W = img_tensor.shape[-2:]
    n_classes = logits.shape[1]
    cams = []
    for c in range(n_classes):
        cam = model.compute_cam(class_idx=c)  # 1 x h x w (feature-map resolution)
        cam = F.interpolate(cam.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False)
        cams.append(cam[0, 0].cpu().numpy())
    cams = np.stack(cams, axis=0)  # n_classes x H x W

    return probs, cams


def normalize_cam(cam: np.ndarray) -> np.ndarray:
    """Min-max normalize a single CAM to [0,1]. If the CAM is flat
    (max == min), returns all-zeros rather than dividing by zero."""
    lo, hi = cam.min(), cam.max()
    if hi - lo < 1e-8:
        return np.zeros_like(cam)
    return (cam - lo) / (hi - lo)


def cam_to_boxes(cam_norm: np.ndarray, box_threshold: float = 0.6,
                 max_area_fraction: float = 0.85, max_boxes: int = 1,
                 min_area_fraction: float = 0.005) -> list[tuple[int, int, int, int]]:
    """
    cam_norm: HxW array in [0,1] (already normalized)
    Returns a list of (x1, y1, x2, y2) boxes, one per connected component
    above box_threshold, largest first, capped at max_boxes. Empty list if
    nothing usable survives thresholding.

    max_boxes: some findings genuinely occur more than once in one image --
    in this dataset ~4,100 images have 2 fracture boxes and ~200 have 3+, so
    returning only the single largest hot region under-reports those. Classes
    that are effectively single-instance can keep max_boxes=1.

    max_area_fraction: a box covering more than this fraction of the image is
    treated as a failed localization (a near-flat CAM at a borderline
    probability can survive min-max normalization and span the whole frame)
    and is dropped. The class can still be reported as present from its
    probability -- this only suppresses a box that would be a useless border
    around the entire X-ray.

    min_area_fraction: drops specks too small to be a real finding.
    """
    mask = (cam_norm >= box_threshold).astype(np.uint8)
    if mask.sum() == 0:
        return []

    n_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:  # background only
        return []

    H, W = cam_norm.shape
    img_area = H * W

    # stats[0] is background; sort the rest by component area, largest first
    comps = sorted(range(1, n_labels), key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)

    boxes = []
    for i in comps:
        x, y, w, h, _area = stats[i]
        frac = (w * h) / img_area
        if frac > max_area_fraction:
            continue  # diffuse/borderline CAM -- not a meaningful localization
        if frac < min_area_fraction:
            continue  # too small to be a real finding
        boxes.append((int(x), int(y), int(x + w), int(y + h)))
        if len(boxes) >= max_boxes:
            break

    return boxes


def cam_to_box(cam_norm: np.ndarray, box_threshold: float = 0.6,
               max_area_fraction: float = 0.85) -> tuple[int, int, int, int] | None:
    """Backwards-compatible single-box wrapper around cam_to_boxes."""
    boxes = cam_to_boxes(cam_norm, box_threshold=box_threshold,
                          max_area_fraction=max_area_fraction, max_boxes=1)
    return boxes[0] if boxes else None


def make_heatmap_overlay(base_img_u8: np.ndarray, cam_norm: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    base_img_u8: HxW or HxWx3 uint8 grayscale/RGB image (0-255)
    cam_norm: HxW float in [0,1]
    Returns HxWx3 uint8 BGR image with a jet-colormap heatmap blended on top.
    """
    if base_img_u8.ndim == 2:
        base_rgb = cv2.cvtColor(base_img_u8, cv2.COLOR_GRAY2BGR)
    else:
        base_rgb = base_img_u8

    heat_u8 = (cam_norm * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(base_rgb, 1 - alpha, heat_color, alpha, 0)
    return overlay


# --- Localization policy -------------------------------------------------
# Which merged classes get a BOUNDING BOX, and how many.
#
# A merged class only gets a box if it represents ONE visually coherent
# finding. CAM produces a heatmap per class; if a class is a merge of several
# different findings that can appear in different places, a single box drawn
# from that heatmap is misleading -- it would imply "the finding is here" when
# the class really means "one of three different things is somewhere".
#
#   fracture            -> pure (fracture only). BOX.
#                          max 3 boxes: ~4,100 images in this dataset have 2
#                          fracture boxes and ~200 have 3+, so a single box
#                          under-reports multi-fracture cases.
#   foreign_material    -> metal (707) + foreignbody (8). 99% metal, so
#                          effectively a single coherent finding. BOX, max 1.
#   softtissue_indirect -> softtissue (439) + pronatorsign (566) +
#                          periostealreaction (2,235): three genuinely
#                          different findings. NO BOX -- present/absent only.
#   bone_lesion         -> boneanomaly (192) + bonelesion (42): two findings,
#                          and the class is unreliable anyway (test AUROC
#                          0.664). NO BOX -- present/absent only.
BOX_POLICY: dict[str, int] = {
    "fracture": 3,
    "foreign_material": 1,
    # classes absent from this dict are reported present/absent with no box
}


def analyze_image(model, img_tensor: torch.Tensor, device, class_names: list[str],
                   thresholds: dict[str, float], box_threshold: float = 0.6,
                   box_policy: dict[str, int] | None = None) -> dict:
    """
    High-level entry point used by both the report-generation script and the
    FastAPI backend (Phase 3).

    Returns a dict:
        {
          "fracture": {"probability": 0.87, "present": True,
                       "localizable": True, "boxes": [(x1,y1,x2,y2), ...]},
          "softtissue_indirect": {"probability": 0.71, "present": True,
                       "localizable": False, "boxes": []},
          ...
        }
    plus "_cams_normalized": dict[class_name -> HxW float32 array in [0,1]]
    for building heatmap overlays.

    "localizable" tells the frontend whether this class is one we are willing
    to draw a box for at all (see BOX_POLICY). A class with localizable=False
    should be shown as a present/absent finding only -- never with a box --
    because it merges several distinct findings.
    """
    if box_policy is None:
        box_policy = BOX_POLICY

    probs, cams = get_probs_and_cams(model, img_tensor, device)

    result = {}
    cams_norm = {}
    for i, name in enumerate(class_names):
        cam_norm = normalize_cam(cams[i])
        cams_norm[name] = cam_norm

        prob = float(probs[i])
        thresh = thresholds.get(name, 0.5)
        present = prob >= thresh

        max_boxes = box_policy.get(name, 0)
        localizable = max_boxes > 0

        boxes = []
        if present and localizable:
            boxes = cam_to_boxes(cam_norm, box_threshold=box_threshold, max_boxes=max_boxes)

        result[name] = {
            "probability": prob,
            "threshold_used": thresh,
            "present": present,
            "localizable": localizable,
            # boxes are in the SAME pixel space as img_tensor (e.g. 256x256)
            "boxes": boxes,
        }

    result["_cams_normalized"] = cams_norm
    return result


if __name__ == "__main__":
    # Self-test: run `python src/cam_utils.py` after training WristNet, to
    # sanity-check CAM extraction + box drawing on a handful of real val
    # images and save annotated PNGs you can eyeball before wiring this into
    # the evaluation script or the backend.
    import sys
    import json

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dataset import CLASS_NAMES, WristMultiLabelDataset  # noqa: E402
    from model import WristNet  # noqa: E402

    project_root = Path(__file__).resolve().parent.parent
    csv_path = project_root / "data" / "classification" / "labels_multilabel.csv"
    images_root = project_root / "data" / "processed" / "images"
    ckpt_path = project_root / "models" / "wristnet" / "best.pt"
    thresholds_path = project_root / "models" / "wristnet" / "tuned_thresholds.json"
    out_dir = project_root / "results" / "cam_selftest"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ckpt_path.exists():
        sys.exit(f"[ERROR] {ckpt_path} not found -- run scripts/02_train_wristnet.py first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[test] loading model from {ckpt_path} on {device}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ck_args = ckpt.get("args", {})
    model = WristNet(
        n_classes=len(CLASS_NAMES),
        dropout=ck_args.get("dropout", 0.4),
        fine_cam=not ck_args.get("coarse_cam", False),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with open(thresholds_path) as f:
        thresholds = json.load(f)
    print(f"[info] using tuned thresholds: {thresholds}")

    img_size = ckpt["args"]["img_size"]
    val_ds = WristMultiLabelDataset(csv_path, images_root / "val", "val",
                                     img_size=img_size, augment=False)

    # pick a few val images that actually have at least one finding, so the
    # self-test is informative rather than showing 5 empty CAMs
    has_finding = val_ds.df[val_ds.df[val_ds.label_cols].sum(axis=1) >= 1]
    sample_positions = [val_ds.df.index.get_loc(i) for i in has_finding.index[:5]]

    for pos in sample_positions:
        img_t, label_t, stem = val_ds[pos]
        result = analyze_image(model, img_t, device, CLASS_NAMES, thresholds)

        base_np = (img_t[0].numpy() * 255).astype(np.uint8)  # single channel back out

        print(f"\n[{stem}]  true_label={label_t.tolist()}")
        for name in CLASS_NAMES:
            r = result[name]
            if r["localizable"]:
                print(f"    {name:22s}: p={r['probability']:.3f} (t={r['threshold_used']:.2f}) "
                      f"present={r['present']}  boxes={r['boxes']}")
            else:
                print(f"    {name:22s}: p={r['probability']:.3f} (t={r['threshold_used']:.2f}) "
                      f"present={r['present']}  [present/absent only -- merged class, not boxed]")

        # save one combined visualization: original | best heatmap overlay | boxes drawn
        boxed = cv2.cvtColor(base_np, cv2.COLOR_GRAY2BGR).copy()
        colors = {"fracture": (0, 0, 255), "foreign_material": (0, 255, 255),
                  "softtissue_indirect": (255, 0, 0), "bone_lesion": (0, 255, 0)}
        for name in CLASS_NAMES:
            for (x1, y1, x2, y2) in result[name]["boxes"]:
                cv2.rectangle(boxed, (x1, y1), (x2, y2), colors[name], 2)
                cv2.putText(boxed, name, (x1, max(0, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[name], 1)

        fracture_cam = result["_cams_normalized"]["fracture"]
        heatmap_vis = make_heatmap_overlay(base_np, fracture_cam)

        combined = np.hstack([cv2.cvtColor(base_np, cv2.COLOR_GRAY2BGR), heatmap_vis, boxed])
        out_path = out_dir / f"{stem}_cam.png"
        cv2.imwrite(str(out_path), combined)
        print(f"    [saved] {out_path}")

    print(f"\n[done] cam_utils.py self-test passed. Check {out_dir} for visualizations.")
