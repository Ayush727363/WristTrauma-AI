"""
main.py -- WristTrauma-AI backend (FastAPI)
--------------------------------------------
Serves the trained WristNet model behind a single /analyze endpoint.

What it does, per uploaded X-ray image:
  1. Loads the image (any common format), converts to the same preprocessing
     pipeline used in training (16-bit-safe grayscale read, per-image min/max
     normalize, letterbox to img_size).
  2. Runs WristNet -> per-class probabilities.
  3. Runs CAM -> heatmap + boxes, respecting the BOX_POLICY from cam_utils.py
     (only fracture and foreign_material ever get boxes; softtissue_indirect
     and bone_lesion are present/absent only, since they're merges of
     several different findings and a single box would misrepresent them).
  4. Applies the FROZEN val-tuned thresholds from
     models/wristnet/tuned_thresholds.json (same ones used in the official
     test evaluation -- nothing is re-tuned here).
  5. Tries to parse patient metadata (age/sex/side/projection) from the
     uploaded filename, GRAZPEDWRI-DX convention:
         <patient>_<study>_<timehash>_WRI-<L|R><1|2>_<Sex><Age>
     e.g. 0009_1112587669_01_WRI-R1_F013 -> Right, Female, 13
     This is real metadata that would normally travel with the DICOM/record,
     displayed as "Patient record" -- it is NEVER presented as something the
     model inferred from the pixels. If the filename doesn't match the
     pattern (e.g. a phone photo), metadata fields are simply omitted, not
     guessed.
  6. Returns everything the frontend needs: original image, heatmap overlay,
     boxed overlay (all as base64 PNG), per-class results, and the parsed
     patient record -- as one JSON response.

Run with:
    uvicorn main:app --reload --port 8000
(from the backend/ folder, with the project's venv active so it can import
 the sibling src/ modules and load models/wristnet/best.pt)
"""

from __future__ import annotations

import base64
import io
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Path setup: this file lives in <project_root>/backend/main.py and needs to
# import from <project_root>/src/. Adjust PROJECT_ROOT if you move this file.
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

try:
    from dataset import CLASS_NAMES, N_CLASSES, letterbox  # noqa: E402
    from model import WristNet  # noqa: E402
    from cam_utils import analyze_image, make_heatmap_overlay, BOX_POLICY  # noqa: E402
except ImportError as e:
    sys.exit(
        f"[FATAL] Could not import from {SRC_DIR}. Make sure this backend/ folder sits "
        f"next to your project's src/ folder (with dataset.py, model.py, cam_utils.py), "
        f"and that you're running with the project's venv active.\nOriginal error: {e}"
    )

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WEIGHTS_PATH = PROJECT_ROOT / "models" / "wristnet" / "best.pt"
THRESHOLDS_PATH = PROJECT_ROOT / "models" / "wristnet" / "tuned_thresholds.json"
METADATA_CSV = PROJECT_ROOT / "data" / "dataset_metadata_cleaned.csv"  # optional, used as a cross-check

# YOLO26 weights: point this at whichever trained run you want to serve.
# Looks for models/<name>/best.pt under the project root; edit YOLO_RUN_NAME
# to match your actual training run folder if it differs.
YOLO_RUN_NAME = "baseline_yolo26s_640"
YOLO_WEIGHTS_PATH = PROJECT_ROOT / "models" / YOLO_RUN_NAME / "best.pt"
YOLO_IMG_SIZE = 640

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_DISPLAY_NAMES = {
    "fracture": "Fracture",
    "foreign_material": "Foreign Material / Metal",
    "softtissue_indirect": "Soft-Tissue / Indirect Signs",
    "bone_lesion": "Bone Anomaly / Lesion",
}

CLASS_COLORS_HEX = {
    "fracture": "#e11d48",           # red
    "foreign_material": "#eab308",   # amber
    "softtissue_indirect": "#3b82f6",  # blue
    "bone_lesion": "#22c55e",        # green
}
CLASS_COLORS_BGR = {
    "fracture": (0, 0, 255),
    "foreign_material": (0, 255, 255),
    "softtissue_indirect": (255, 0, 0),
    "bone_lesion": (0, 255, 0),
}

# YOLO26's original 9-class GRAZPEDWRI-DX taxonomy (unmerged), in the same
# order the model was trained with -- must match dataset.yaml's class list.
YOLO_CLASS_NAMES = [
    "boneanomaly", "bonelesion", "foreignbody", "fracture", "metal",
    "periostealreaction", "pronatorsign", "softtissue", "text",
]
YOLO_CLASS_DISPLAY_NAMES = {
    "fracture": "Fracture",
    "periostealreaction": "Periosteal Reaction",
    "pronatorsign": "Pronator Sign",
    "softtissue": "Soft Tissue Swelling",
    "metal": "Metal",
    "boneanomaly": "Bone Anomaly",
    "bonelesion": "Bone Lesion",
    "foreignbody": "Foreign Body",
    "text": "Text / Marker",
}
YOLO_CLASS_COLORS_BGR = {
    "fracture": (74, 71, 224),
    "periostealreaction": (217, 144, 74),
    "pronatorsign": (214, 111, 124),
    "softtissue": (196, 184, 63),
    "metal": (74, 169, 212),
    "boneanomaly": (172, 216, 79),
    "bonelesion": (60, 138, 224),
    "foreignbody": (214, 122, 196),
    "text": (173, 163, 143),
}
YOLO_CLASS_COLORS_HEX = {
    "fracture": "#e0475a",
    "periostealreaction": "#4a90d9",
    "pronatorsign": "#7c6fd6",
    "softtissue": "#3fb8c4",
    "metal": "#d4a94a",
    "boneanomaly": "#4fd8ac",
    "bonelesion": "#e08a3c",
    "foreignbody": "#c47ad6",
    "text": "#8fa3ad",
}

# ---------------------------------------------------------------------------
# Model loading (once, at startup)
# ---------------------------------------------------------------------------
_model = None
_thresholds = None
_img_size = 256
_yolo_model = None


def load_model():
    global _model, _thresholds, _img_size
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"WristNet weights not found at {WEIGHTS_PATH}. "
            f"Train the model first (scripts/02_train_wristnet.py)."
        )
    if not THRESHOLDS_PATH.exists():
        raise FileNotFoundError(f"Tuned thresholds not found at {THRESHOLDS_PATH}.")

    ckpt = torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=False)
    ck_args = ckpt.get("args", {})
    _img_size = ck_args.get("img_size", 256)

    model = WristNet(
        n_classes=N_CLASSES,
        dropout=ck_args.get("dropout", 0.4),
        fine_cam=not ck_args.get("coarse_cam", False),
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with open(THRESHOLDS_PATH) as f:
        thresholds = json.load(f)

    print(f"[startup] loaded WristNet from {WEIGHTS_PATH} (epoch {ckpt.get('epoch')}), "
          f"img_size={_img_size}, device={DEVICE}")
    print(f"[startup] thresholds: {thresholds}")
    return model, thresholds


def load_yolo_model():
    """Loads the trained YOLO26 checkpoint, if present. Returns None (rather
    than raising) when the weights file is missing, so the CNN endpoint
    keeps working even if YOLO hasn't been trained/placed yet -- the
    /analyze-yolo endpoint reports a clear 503 in that case instead of the
    whole server failing to start."""
    if not YOLO_WEIGHTS_PATH.exists():
        print(f"[startup] YOLO weights not found at {YOLO_WEIGHTS_PATH} -- "
              f"/analyze-yolo will return 503 until they're added.")
        return None
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[startup] ultralytics not installed -- run: pip install ultralytics")
        return None

    model = YOLO(str(YOLO_WEIGHTS_PATH))
    print(f"[startup] loaded YOLO26 from {YOLO_WEIGHTS_PATH}")
    return model


# ---------------------------------------------------------------------------
# Filename metadata parsing (GRAZPEDWRI-DX convention)
#   <patient>_<study>_<timehash>_WRI-<side><proj>_<sex><age>
#   e.g. 0009_1112587669_01_WRI-R1_F013
# ---------------------------------------------------------------------------
FILENAME_PATTERN = re.compile(
    r"WRI-(?P<side>[LR])(?P<proj>\d)_(?P<sex>[MF])(?P<age>\d{3})",
    re.IGNORECASE,
)


def parse_patient_record(filename: str) -> dict | None:
    """Best-effort parse of GRAZPEDWRI-DX filename metadata. Returns None if
    the filename doesn't match the expected convention (e.g. a phone photo
    or a renamed file) -- we never guess or fabricate patient info."""
    match = FILENAME_PATTERN.search(filename)
    if not match:
        return None

    side = "Left" if match.group("side").upper() == "L" else "Right"
    sex = "Female" if match.group("sex").upper() == "F" else "Male"
    age_raw = match.group("age")
    try:
        age = int(age_raw)
    except ValueError:
        age = None
    proj_num = match.group("proj")
    projection = "AP (frontal)" if proj_num == "1" else ("Lateral" if proj_num == "2" else f"View {proj_num}")

    return {
        "side": side,
        "sex": sex,
        "age": age,
        "projection": projection,
        "source": "filename",  # this came from the file's naming convention, not the model
    }


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def read_upload_to_array(file_bytes: bytes) -> np.ndarray:
    """Reads uploaded image bytes into a single-channel float32 array in
    [0,1], same preprocessing as training: cv2 IMREAD_UNCHANGED (handles
    16-bit PNGs correctly), per-image min/max normalize."""
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Could not decode image. Supported formats: PNG, JPG, BMP, TIFF.")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = img.astype(np.float32)
    lo, hi = img.min(), img.max()
    if hi > lo:
        img = (img - lo) / (hi - lo)
    else:
        img = np.zeros_like(img)
    return img


def np_to_base64_png(img_bgr_or_gray: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img_bgr_or_gray)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="WristTrauma-AI API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your deployed frontend's origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup():
    global _model, _thresholds, _yolo_model
    _model, _thresholds = load_model()
    _yolo_model = load_yolo_model()


class ClassResult(BaseModel):
    key: str
    display_name: str
    probability: float
    threshold: float
    present: bool
    localizable: bool
    boxes: list[list[int]]  # each box is [x1, y1, x2, y2] in the analysis image's pixel space
    color: str


class AnalyzeResponse(BaseModel):
    filename: str
    img_size: int
    classes: list[ClassResult]
    patient_record: dict | None
    original_image_b64: str
    heatmap_image_b64: str
    boxed_image_b64: str
    top_class: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "model_loaded": _model is not None,
        "yolo_loaded": _yolo_model is not None,
    }


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...)):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    file_bytes = await file.read()
    try:
        img_norm = read_upload_to_array(file_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    img_letterboxed = letterbox(img_norm, _img_size)
    img_tensor = torch.from_numpy(img_letterboxed).unsqueeze(0).repeat(3, 1, 1).float()

    result = analyze_image(_model, img_tensor, DEVICE, CLASS_NAMES, _thresholds)

    base_np = (img_letterboxed * 255).astype(np.uint8)
    base_bgr = cv2.cvtColor(base_np, cv2.COLOR_GRAY2BGR)

    # heatmap: show the class with the highest predicted probability
    top_class = max(CLASS_NAMES, key=lambda n: result[n]["probability"])
    heatmap_vis = make_heatmap_overlay(base_np, result["_cams_normalized"][top_class])

    # boxed overlay: draw all boxes from all boxable classes (fracture, foreign_material)
    boxed_vis = base_bgr.copy()
    for name in CLASS_NAMES:
        for (x1, y1, x2, y2) in result[name]["boxes"]:
            color = CLASS_COLORS_BGR[name]
            cv2.rectangle(boxed_vis, (x1, y1), (x2, y2), color, 2)
            label = f"{CLASS_DISPLAY_NAMES[name]} {result[name]['probability']:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            ty = max(th + 4, y1 - 4)
            cv2.rectangle(boxed_vis, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), color, -1)
            cv2.putText(boxed_vis, label, (x1 + 2, ty - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    classes_out = []
    for name in CLASS_NAMES:
        r = result[name]
        classes_out.append(ClassResult(
            key=name,
            display_name=CLASS_DISPLAY_NAMES[name],
            probability=r["probability"],
            threshold=r["threshold_used"],
            present=r["present"],
            localizable=r["localizable"],
            boxes=[list(b) for b in r["boxes"]],
            color=CLASS_COLORS_HEX[name],
        ))

    patient_record = parse_patient_record(file.filename or "")

    return AnalyzeResponse(
        filename=file.filename or "upload.png",
        img_size=_img_size,
        classes=classes_out,
        patient_record=patient_record,
        original_image_b64=np_to_base64_png(base_bgr),
        heatmap_image_b64=np_to_base64_png(heatmap_vis),
        boxed_image_b64=np_to_base64_png(boxed_vis),
        top_class=top_class,
    )


class YoloClassResult(BaseModel):
    key: str
    display_name: str
    probability: float          # max confidence across all boxes of this class, 0 if none detected
    present: bool
    boxes: list[list[int]]      # [x1, y1, x2, y2] in the analysis image's pixel space
    color: str


class YoloAnalyzeResponse(BaseModel):
    filename: str
    img_size: int
    classes: list[YoloClassResult]
    patient_record: dict | None
    original_image_b64: str
    boxed_image_b64: str


@app.post("/analyze-yolo", response_model=YoloAnalyzeResponse)
async def analyze_yolo(file: UploadFile = File(...)):
    """
    Runs the trained YOLO26 detector over the full, unmerged 9-class
    GRAZPEDWRI-DX taxonomy. Unlike /analyze (WristNet), this endpoint does
    not apply any CAM/thresholding logic -- YOLO's own confidence scores and
    boxes are used directly, at YOLO's own default confidence threshold.
    """
    if _yolo_model is None:
        raise HTTPException(
            status_code=503,
            detail=f"YOLO26 weights not found at {YOLO_WEIGHTS_PATH}. "
                    "Train it first or point YOLO_RUN_NAME at your run folder.",
        )

    file_bytes = await file.read()
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img_raw = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img_raw is None:
        raise HTTPException(status_code=400, detail="Could not decode image. Supported formats: PNG, JPG, BMP, TIFF.")

    # Normalize the same way as the CNN path so 16-bit radiographs display
    # correctly, then convert to a plain 8-bit 3-channel image for YOLO.
    if img_raw.ndim == 3:
        img_gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img_raw
    img_gray = img_gray.astype(np.float32)
    lo, hi = img_gray.min(), img_gray.max()
    img_norm = (img_gray - lo) / (hi - lo) if hi > lo else np.zeros_like(img_gray)
    img_u8 = (img_norm * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)

    # Ultralytics handles its own internal resize/letterbox to YOLO_IMG_SIZE;
    # we pass the normalized image and let it manage that.
    results = _yolo_model.predict(source=img_bgr, imgsz=YOLO_IMG_SIZE, device=DEVICE, verbose=False)
    result = results[0]

    # result.orig_img is the image YOLO actually drew boxes against (same
    # pixel space as result.boxes.xyxy), so we display that, not img_bgr.
    display_img = result.orig_img.copy()
    boxed_vis = display_img.copy()

    per_class_boxes: dict[str, list[list[int]]] = {name: [] for name in YOLO_CLASS_NAMES}
    per_class_best_conf: dict[str, float] = {name: 0.0 for name in YOLO_CLASS_NAMES}

    if result.boxes is not None:
        for box in result.boxes:
            cls_idx = int(box.cls.item())
            conf = float(box.conf.item())
            name = result.names.get(cls_idx, YOLO_CLASS_NAMES[cls_idx] if cls_idx < len(YOLO_CLASS_NAMES) else None)
            if name not in per_class_boxes:
                continue  # unknown class index, skip defensively
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            per_class_boxes[name].append([x1, y1, x2, y2])
            per_class_best_conf[name] = max(per_class_best_conf[name], conf)

            color = YOLO_CLASS_COLORS_BGR.get(name, (255, 255, 255))
            cv2.rectangle(boxed_vis, (x1, y1), (x2, y2), color, 2)
            label = f"{YOLO_CLASS_DISPLAY_NAMES.get(name, name)} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            ty = max(th + 4, y1 - 4)
            cv2.rectangle(boxed_vis, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), color, -1)
            cv2.putText(boxed_vis, label, (x1 + 2, ty - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    classes_out = [
        YoloClassResult(
            key=name,
            display_name=YOLO_CLASS_DISPLAY_NAMES[name],
            probability=per_class_best_conf[name],
            present=len(per_class_boxes[name]) > 0,
            boxes=per_class_boxes[name],
            color=YOLO_CLASS_COLORS_HEX[name],
        )
        for name in YOLO_CLASS_NAMES
    ]

    patient_record = parse_patient_record(file.filename or "")

    return YoloAnalyzeResponse(
        filename=file.filename or "upload.png",
        img_size=YOLO_IMG_SIZE,
        classes=classes_out,
        patient_record=patient_record,
        original_image_b64=np_to_base64_png(display_img),
        boxed_image_b64=np_to_base64_png(boxed_vis),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
