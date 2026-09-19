// app.js
// ------
// Core frontend logic: file upload, calling the backend /analyze endpoint,
// rendering results, the zoom lightbox, and wiring up the report modal
// (report.js owns the actual report HTML/PDF generation).

(() => {
  "use strict";

  // ---------------- state ----------------
  let selectedFile = null;
  let lastResult = null; // full API response, kept for the report generator

  // ---------------- element refs ----------------
  const $ = (id) => document.getElementById(id);

  const dropzone = $("dropzone");
  const fileInput = $("fileInput");
  const browseBtn = $("browseBtn");
  const dzPreview = $("dzPreview");
  const dzPreviewImg = $("dzPreviewImg");
  const dzFileName = $("dzFileName");
  const dzClearBtn = $("dzClearBtn");
  const analyzeBtn = $("analyzeBtn");
  const analyzeBtnText = $("analyzeBtnText");

  const uploadSection = $("uploadSection");
  const loadingPanel = $("loadingPanel");
  const scanningImg = $("scanningImg");
  const resultsSection = $("resultsSection");
  const resultsFilename = $("resultsFilename");
  const newAnalysisBtn = $("newAnalysisBtn");
  const viewReportBtn = $("viewReportBtn");

  const imgOriginal = $("imgOriginal");
  const imgHeatmap = $("imgHeatmap");
  const imgBoxed = $("imgBoxed");
  const heatmapClassLabel = $("heatmapClassLabel");

  const findingsList = $("findingsList");
  const patientRecordBody = $("patientRecordBody");

  const apiStatusPill = $("apiStatusPill");
  const apiStatusText = $("apiStatusText");

  const lightbox = $("lightbox");
  const lightboxImg = $("lightboxImg");
  const lightboxTabs = $("lightboxTabs");
  const lightboxClose = $("lightboxClose");
  const lightboxBackdrop = $("lightboxBackdrop");
  const lightboxStage = $("lightboxStage");
  const zoomSlider = $("zoomSlider");
  const zoomPct = $("zoomPct");
  const zoomInBtn = $("zoomInBtn");
  const zoomOutBtn = $("zoomOutBtn");
  const zoomResetBtn = $("zoomResetBtn");

  const CLASS_DISPLAY_ORDER = ["fracture", "softtissue_indirect", "foreign_material", "bone_lesion"];

  // ---------------- health check ----------------
  async function checkApiHealth() {
    try {
      const res = await fetch(`${API_BASE_URL}/health`, { method: "GET" });
      if (!res.ok) throw new Error("bad status");
      const data = await res.json();
      apiStatusPill.querySelector(".dot").className = "dot dot-ok";
      apiStatusText.textContent = `Model ready · ${data.device.toUpperCase()}`;
    } catch (e) {
      apiStatusPill.querySelector(".dot").className = "dot dot-err";
      apiStatusText.textContent = "Backend unreachable";
    }
  }
  checkApiHealth();

  // ---------------- file selection ----------------
  function isValidImageFile(file) {
    return file && /^image\/(png|jpe?g|bmp|tiff?)$/i.test(file.type) || /\.(png|jpe?g|bmp|tiff?)$/i.test(file.name);
  }

  function setSelectedFile(file) {
    if (!isValidImageFile(file)) {
      alert("Please select a PNG, JPG, BMP, or TIFF image.");
      return;
    }
    selectedFile = file;
    const url = URL.createObjectURL(file);
    dzPreviewImg.src = url;
    dzFileName.textContent = file.name;
    dropzone.querySelector(".dz-inner").hidden = true;
    dzPreview.hidden = false;
    analyzeBtn.disabled = false;
    scanningImg.src = url;
  }

  function clearSelectedFile() {
    selectedFile = null;
    fileInput.value = "";
    dropzone.querySelector(".dz-inner").hidden = false;
    dzPreview.hidden = true;
    analyzeBtn.disabled = true;
  }

  browseBtn.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("click", (e) => {
    if (dzPreview.hidden) fileInput.click();
  });
  dropzone.addEventListener("keydown", (e) => {
    if ((e.key === "Enter" || e.key === " ") && dzPreview.hidden) fileInput.click();
  });
  fileInput.addEventListener("change", (e) => {
    if (e.target.files[0]) setSelectedFile(e.target.files[0]);
  });
  dzClearBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    clearSelectedFile();
  });

  ["dragenter", "dragover"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add("dragover");
    })
  );
  ["dragleave", "drop"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove("dragover");
    })
  );
  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (file) setSelectedFile(file);
  });

  // ---------------- analyze flow ----------------
  analyzeBtn.addEventListener("click", runAnalysis);

  async function runAnalysis() {
    if (!selectedFile) return;

    uploadSection.hidden = true;
    loadingPanel.hidden = false;
    resultsSection.hidden = true;
    analyzeBtn.disabled = true;

    animateLoadingSteps();

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const res = await fetch(`${API_BASE_URL}/analyze`, { method: "POST", body: formData });
      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}));
        throw new Error(errBody.detail || `Server error (${res.status})`);
      }
      const data = await res.json();
      lastResult = data;
      renderResults(data);
    } catch (err) {
      alert(`Analysis failed: ${err.message}\n\nMake sure the backend is running and reachable at ${API_BASE_URL}.`);
      loadingPanel.hidden = true;
      uploadSection.hidden = false;
      analyzeBtn.disabled = false;
      return;
    }

    loadingPanel.hidden = true;
    resultsSection.hidden = false;
    window.scrollTo({ top: resultsSection.offsetTop - 20, behavior: "smooth" });
  }

  function animateLoadingSteps() {
    const steps = document.querySelectorAll(".lstep");
    let i = 0;
    steps.forEach((s) => s.classList.remove("active", "done"));
    steps[0].classList.add("active");
    const interval = setInterval(() => {
      if (loadingPanel.hidden) {
        clearInterval(interval);
        return;
      }
      steps[i].classList.remove("active");
      steps[i].classList.add("done");
      i = (i + 1) % steps.length;
      if (i === 0) {
        steps.forEach((s) => s.classList.remove("done"));
      }
      steps[i].classList.add("active");
    }, 900);
  }

  newAnalysisBtn.addEventListener("click", () => {
    resultsSection.hidden = true;
    uploadSection.hidden = false;
    clearSelectedFile();
    window.scrollTo({ top: 0, behavior: "smooth" });
  });

  // ---------------- rendering ----------------
  function renderResults(data) {
    resultsFilename.textContent = data.filename;

    imgOriginal.src = `data:image/png;base64,${data.original_image_b64}`;
    imgHeatmap.src = `data:image/png;base64,${data.heatmap_image_b64}`;
    imgBoxed.src = `data:image/png;base64,${data.boxed_image_b64}`;

    const topClass = data.classes.find((c) => c.key === data.top_class);
    heatmapClassLabel.textContent = topClass ? topClass.display_name : "—";

    renderFindings(data.classes);
    renderPatientRecord(data.patient_record);
  }

  function renderFindings(classes) {
    findingsList.innerHTML = "";
    const ordered = CLASS_DISPLAY_ORDER
      .map((key) => classes.find((c) => c.key === key))
      .filter(Boolean);

    ordered.forEach((c) => {
      const row = document.createElement("div");
      row.className = "finding-row" + (c.present ? " is-present" : "");

      const pct = Math.round(c.probability * 100);
      row.innerHTML = `
        <span class="finding-dot" style="background:${c.color}"></span>
        <div class="finding-main">
          <div class="finding-name">
            ${c.display_name}
            <span class="finding-badge ${c.present ? "badge-present" : "badge-absent"}">
              ${c.present ? "Present" : "Absent"}
            </span>
          </div>
          <div class="finding-meta">
            ${c.localizable ? "Localizable · box drawn when present" : "Composite finding · no box (see note)"}
            &nbsp;·&nbsp; threshold ${(c.threshold * 100).toFixed(0)}%
          </div>
          <div class="finding-bar-wrap"><div class="finding-bar" style="width:${pct}%; background:${c.color}"></div></div>
        </div>
        <span class="finding-prob">${pct}%</span>
      `;
      findingsList.appendChild(row);
    });
  }

  function renderPatientRecord(record) {
    if (!record) {
      patientRecordBody.innerHTML = `
        <div class="record-empty">
          No structured patient record could be parsed from this filename.
          Rename the file using the GRAZPEDWRI-DX convention
          (e.g. <code>0009_..._WRI-R1_F013</code>) to auto-populate age, sex, side and projection.
        </div>`;
      return;
    }
    const rows = [
      ["Age", record.age != null ? `${record.age} yrs` : "—"],
      ["Sex", record.sex || "—"],
      ["Side", record.side || "—"],
      ["Projection", record.projection || "—"],
    ];
    patientRecordBody.innerHTML = rows
      .map(
        ([label, value]) => `
      <div class="record-item">
        <span class="record-label">${label}</span>
        <span class="record-value">${value}</span>
      </div>`
      )
      .join("");
  }

  // ---------------- lightbox + zoom ----------------
  const zoomTargets = {
    original: () => imgOriginal.src,
    heatmap: () => imgHeatmap.src,
    boxed: () => imgBoxed.src,
  };

  const ZOOM_MIN = 100;
  const ZOOM_MAX = 400;
  let currentZoom = 100;

  function applyZoom(pct) {
    currentZoom = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, pct));
    lightboxImg.style.transform = `scale(${currentZoom / 100})`;
    zoomSlider.value = currentZoom;
    zoomPct.textContent = `${Math.round(currentZoom)}%`;
    // once zoomed past fit, let the stage scroll so every part of the
    // image at high zoom is reachable, not just clipped to the frame
    lightboxStage.style.overflow = currentZoom > 100 ? "auto" : "hidden";
  }

  function resetZoom() {
    applyZoom(100);
    lightboxStage.scrollTo({ top: 0, left: 0 });
  }

  function openLightbox(target) {
    lightbox.hidden = false;
    setLightboxTab(target);
    resetZoom();
    document.body.style.overflow = "hidden";
  }
  function closeLightbox() {
    lightbox.hidden = true;
    document.body.style.overflow = "";
  }
  function setLightboxTab(target) {
    lightboxImg.src = zoomTargets[target]();
    [...lightboxTabs.children].forEach((btn) =>
      btn.classList.toggle("active", btn.dataset.target === target)
    );
    resetZoom();
  }

  document.querySelectorAll(".img-frame").forEach((frame) => {
    frame.addEventListener("click", () => {
      const target = frame.closest("[data-zoom-target]").dataset.zoomTarget;
      openLightbox(target);
    });
  });
  lightboxTabs.addEventListener("click", (e) => {
    const btn = e.target.closest(".lb-tab");
    if (btn) setLightboxTab(btn.dataset.target);
  });
  lightboxClose.addEventListener("click", closeLightbox);
  lightboxBackdrop.addEventListener("click", closeLightbox);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !lightbox.hidden) closeLightbox();
  });

  zoomSlider.addEventListener("input", (e) => applyZoom(Number(e.target.value)));
  zoomInBtn.addEventListener("click", () => applyZoom(currentZoom + 25));
  zoomOutBtn.addEventListener("click", () => applyZoom(currentZoom - 25));
  zoomResetBtn.addEventListener("click", resetZoom);

  // scroll-wheel zoom over the image, for convenience (holding no modifier
  // needed since the stage isn't otherwise scrollable at 100%)
  lightboxStage.addEventListener(
    "wheel",
    (e) => {
      if (lightbox.hidden) return;
      e.preventDefault();
      applyZoom(currentZoom + (e.deltaY < 0 ? 15 : -15));
    },
    { passive: false }
  );

  // ---------------- report modal trigger ----------------
  viewReportBtn.addEventListener("click", () => {
    if (!lastResult) return;
    window.WristReport.openReportModal(lastResult);
  });
})();
