// config.js
// ---------
// Single place to point the frontend at your backend, and to edit the
// report letterhead.
//
// LOCAL DEVELOPMENT:
//   Leave API_BASE_URL as-is if running the backend locally with:
//     uvicorn main:app --reload --port 8000
//
// DEPLOYMENT (e.g. Render):
//   1. Deploy backend/ as a web service on Render (or similar). It will
//      give you a URL like https://wristtrauma-api.onrender.com
//   2. Change API_BASE_URL below to that URL (no trailing slash).
//   3. Deploy frontend/ as a static site (Vercel, Render static site, etc.)
//
// The backend already sends permissive CORS headers (see main.py), so no
// other changes are needed for cross-origin requests to work.

const API_BASE_URL = "http://localhost:8000";

// Single character/short glyph shown inside the circular emblem on the
// report (page 1 header and each figure-page header). Plain text/emoji only
// -- this renders through html2canvas into the PDF, so keep it simple.
const BRAND_EMBLEM_TEXT = "WT";

// Hospital / report letterhead details — edit these for your report.
const REPORT_CONFIG = {
  hospitalName: "WristTrauma-AI Diagnostic Center",
  hospitalSub: "Department of Musculoskeletal Radiology &middot; AI-Assisted Reporting Unit",
  hospitalAddress: "Academic Research Deployment &middot; GRAZPEDWRI-DX Pediatric Wrist Cohort",
  doctors: [
    { name: "Dr. Ayush Dwivedi", role: "Project Lead, Model Development", reg: "Reg. No. 23BCE1539" },
    { name: "Dr. Kartik Ahlawat", role: "Co-Investigator, Clinical Validation", reg: "Reg. No. 23BCE1966" },
  ],
};
