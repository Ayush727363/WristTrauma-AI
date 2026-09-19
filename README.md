# WristTrauma-AI v2 — Meridian Orthopaedic Institute

A full React (Vite + Tailwind + Framer Motion) multi-page frontend, paired
with a FastAPI backend serving both trained models: **WristNet** (custom CNN
with CAM explainability) and **YOLO26s** (9-class object detector).

```
wristtrauma_v2/
├── backend/
│   ├── main.py              FastAPI app — /analyze (CNN) + /analyze-yolo (YOLO26)
│   ├── requirements.txt
│   └── render.yaml
├── frontend/                 React app (Vite)
│   ├── src/
│   │   ├── pages/            Home.jsx, CnnPage.jsx, YoloPage.jsx, AboutPage.jsx
│   │   ├── components/       NavBar, Footer, Uploader, ScanAnimation, ReportModal, ...
│   │   └── lib/               config.js (branding/API URL), api.js
│   ├── index.html / vite.config.js / package.json
│   └── ...
└── src/
    ├── dataset.py, model.py, cam_utils.py   (your existing WristNet code)
```

## 1. Where this goes in your project

Unzip into your `WristTrauma-AI/` project root, alongside your existing
`models/`, `data/`, `scripts/` folders. The backend imports `src/dataset.py`,
`src/model.py`, `src/cam_utils.py` and loads:
- `models/wristnet/best.pt` + `tuned_thresholds.json` (WristNet)
- `models/<YOLO_RUN_NAME>/best.pt` (YOLO26 — see step 3)

## 2. Run the backend

```powershell
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

You should see two startup lines:
```
[startup] loaded WristNet from ...models/wristnet/best.pt (epoch 65), img_size=256, device=cuda
[startup] loaded YOLO26 from ...models/baseline_yolo26s_640/best.pt
```

If the second line instead says `YOLO weights not found`, open
`backend/main.py` and check `YOLO_RUN_NAME` (near the top) matches your
actual training run folder name under `models/` — e.g. if your weights are
at `models/yolo26s_512_exp1/best.pt`, set:
```python
YOLO_RUN_NAME = "yolo26s_512_exp1"
```
The CNN endpoint (`/analyze`) works independently either way — YOLO being
unavailable only disables `/analyze-yolo` (with a clear 503 response),
not the whole server.

Visit `http://localhost:8000/health` — should show
`{"model_loaded":true,"yolo_loaded":true,...}`.

## 3. Run the frontend

```powershell
cd frontend
npm install
npm run dev
```

Open the URL Vite prints (typically `http://localhost:5173`). The status
pill in the top-right should turn green once it reaches your backend.

To point the frontend at a different backend URL (e.g. after deploying),
create `frontend/.env`:
```
VITE_API_BASE_URL=https://your-backend.onrender.com
```

## 4. Editing branding / letterhead

Everything hospital-name/doctor/color related lives in one file:
**`frontend/src/lib/config.js`** — `HOSPITAL`, `DOCTORS`, and the class
label/color maps for both models. No other file needs touching to rebrand.

## 5. The report / PDF export

Reports are generated as real HTML (matching what you see in the "View
Report" modal) and exported via the browser's native print pipeline
(`window.print()` → "Save as PDF"), not a canvas-rasterization library.
This was deliberately chosen after extensive testing: it produces exact,
undistorted, selectable-text A4 pages with proper multi-page pagination.
See the print CSS block at the bottom of `frontend/src/index.css` if you
ever need to adjust the report's print layout — it's commented in detail,
including a note on a subtle browser print/portal quirk that a naive
`visibility:hidden` approach runs into (the actual fix — `display:none`
plus rendering the modal via a React portal to `document.body` — is
already implemented in `ReportModal.jsx`).

## 6. Building for production

```powershell
cd frontend
npm run build
```
Outputs to `frontend/dist/` — deploy as a static site (Vercel, Netlify,
Render static site, etc.). See `backend/render.yaml` for deploying the
backend to Render.

## 7. What's different from the plain-HTML v1

This is a ground-up rebuild: React Router multi-page app (Home / WristNet
CNN / YOLO26 / About) instead of a single page, Framer Motion throughout
(page transitions, scroll-triggered reveals, the cinematic scan sequence),
a premium editorial hospital visual identity (serif display type, deep
emerald/gold palette) rather than a government-portal look, and a second
live model (YOLO26, full 9-class taxonomy) with its own dedicated page and
backend endpoint.
