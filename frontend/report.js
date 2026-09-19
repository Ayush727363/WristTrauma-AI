// report.js
// ---------
// Builds the 4-page diagnostic report and exports it to PDF.
//
//   Page 1 -- text report, strictly black-on-white, Indian diagnostic-lab
//             format (boxed header, ruled patient-info grid, ruled findings
//             table, plain-text status with no color, signature block).
//   Page 2 -- Figure 1: Original radiograph, full page, centered, titled.
//   Page 3 -- Figure 2: Class Activation Map heatmap, full page.
//   Page 4 -- Figure 3: Localization (boxed) image, full page.
//
// Each report page is rasterized with html2canvas and placed onto its own
// A4 jsPDF page, so what you see in the preview modal is exactly what gets
// downloaded -- no separate PDF-only layout to keep in sync.
//
// Exposed globally as window.WristReport = { openReportModal }

(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const reportModal = $("reportModal");
  const reportModalStage = $("reportModalStage");
  const reportModalClose = $("reportModalClose");
  const reportModalBackdrop = $("reportModalBackdrop");
  const downloadPdfBtn = $("downloadPdfBtn");

  const CLASS_DISPLAY_ORDER = ["fracture", "softtissue_indirect", "foreign_material", "bone_lesion"];

  const CLASS_IMPRESSIONS = {
    fracture: "A cortical discontinuity consistent with an acute fracture is identified, localized on the accompanying activation map and bounding-box overlay (see Figures 2-3).",
    softtissue_indirect: "Soft-tissue swelling and/or indirect radiographic signs (periosteal reaction / pronator fat-pad sign) are noted, suggestive of underlying trauma.",
    foreign_material: "A radio-opaque foreign material / metallic density is identified within the imaging field, localized on the accompanying overlay (see Figure 3).",
    bone_lesion: "A focal bone density abnormality is flagged with low model confidence. Given the limited training support for this finding, correlation with clinical history and dedicated radiologist review is strongly advised.",
  };

  function fmtDate(d) {
    return d.toLocaleDateString("en-IN", { day: "2-digit", month: "short", year: "numeric" });
  }
  function fmtTime(d) {
    return d.toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit" });
  }
  function genAccessionId() {
    const now = new Date();
    const y = now.getFullYear();
    const rand = Math.floor(1000 + Math.random() * 9000);
    return `WTA/${y}/${rand}`;
  }

  function buildImpressionText(classes) {
    const present = classes.filter((c) => c.present);
    if (present.length === 0) {
      return "No AI-flagged abnormal findings were identified on this radiograph at the applied confidence thresholds. Clinical correlation is nonetheless advised, particularly in the presence of point tenderness or a reliable mechanism of injury, as subtle non-displaced fractures can be radiographically occult.";
    }
    const sentences = present
      .sort((a, b) => b.probability - a.probability)
      .map((c) => CLASS_IMPRESSIONS[c.key] || `${c.display_name} flagged as present.`);
    return sentences.join(" ") + " Findings are AI-generated and require confirmation by a qualified radiologist prior to clinical action.";
  }

  // ------------------------------------------------------------------
  // Page 1 — text report
  // ------------------------------------------------------------------
  function buildReportPage1(data) {
    const now = new Date();
    const record = data.patient_record;
    const accessionId = genAccessionId();

    const rows = CLASS_DISPLAY_ORDER
      .map((key) => data.classes.find((c) => c.key === key))
      .filter(Boolean);

    const findingsRows = rows
      .map((c) => {
        const statusClass = c.present ? "rp-status-pos" : "rp-status-neg";
        const statusText = c.present ? "POSITIVE" : "NEGATIVE";
        const locText = c.localizable
          ? (c.present ? `${c.boxes.length} region(s), see Fig. 3` : "N/A")
          : "Not localized (composite finding)";
        return `
        <tr>
          <td>${c.display_name}</td>
          <td class="${statusClass}">${statusText}</td>
          <td>${(c.probability * 100).toFixed(1)}%</td>
          <td>${(c.threshold * 100).toFixed(0)}%</td>
          <td>${locText}</td>
        </tr>`;
      })
      .join("");

    const patientFields = record
      ? `
        <div><div class="rp-field-label">Age</div><div class="rp-field-value">${record.age != null ? record.age + " Years" : "Not on record"}</div></div>
        <div><div class="rp-field-label">Sex</div><div class="rp-field-value">${record.sex || "Not on record"}</div></div>
        <div><div class="rp-field-label">Laterality</div><div class="rp-field-value">${record.side || "Not on record"}</div></div>
        <div><div class="rp-field-label">Projection</div><div class="rp-field-value">${record.projection || "Not on record"}</div></div>
      `
      : `
        <div><div class="rp-field-label">Age</div><div class="rp-field-value">Not on record</div></div>
        <div><div class="rp-field-label">Sex</div><div class="rp-field-value">Not on record</div></div>
        <div><div class="rp-field-label">Laterality</div><div class="rp-field-value">Not on record</div></div>
        <div><div class="rp-field-label">Projection</div><div class="rp-field-value">Not on record</div></div>
      `;

    const impression = buildImpressionText(data.classes);

    return `
    <div class="report-page" id="reportPage1">
      <div class="rp-outer-border">

        <div class="rp-header">
          <div class="rp-hospital">
            <div class="rp-hospital-emblem">${BRAND_EMBLEM_TEXT}</div>
            <div>
              <div class="rp-hospital-name">${REPORT_CONFIG.hospitalName}</div>
              <div class="rp-hospital-sub">${REPORT_CONFIG.hospitalSub}</div>
              <div class="rp-hospital-addr">${REPORT_CONFIG.hospitalAddress}</div>
            </div>
          </div>
          <div class="rp-doc-id">
            <div>Accession No: ${accessionId}</div>
            <div>Report Date: ${fmtDate(now)}</div>
            <div>Report Time: ${fmtTime(now)}</div>
          </div>
        </div>

        <div class="rp-title-row">
          <div class="rp-title">Radiological Analysis Report &mdash; Wrist (AP / Lateral)</div>
          <div class="rp-title-sub">AI-Assisted Preliminary Screening &middot; Pediatric Wrist Trauma Protocol</div>
        </div>

        <div class="rp-section">
          <span class="rp-section-title">Patient Information</span>
          <div class="rp-grid">
            ${patientFields}
          </div>
        </div>

        <div class="rp-section">
          <span class="rp-section-title">Study Details</span>
          <div class="rp-grid">
            <div><div class="rp-field-label">Study Type</div><div class="rp-field-value">Wrist Radiograph (X-Ray)</div></div>
            <div><div class="rp-field-label">Source File</div><div class="rp-field-value" style="font-family:'JetBrains Mono',monospace; font-size:10px; word-break:break-all;">${data.filename}</div></div>
            <div><div class="rp-field-label">Analysis Model</div><div class="rp-field-value">WristNet (Custom CNN)</div></div>
            <div><div class="rp-field-label">Method</div><div class="rp-field-value">Class Activation Mapping</div></div>
          </div>
        </div>

        <div class="rp-section">
          <span class="rp-section-title">AI-Detected Findings</span>
          <table class="rp-findings-table">
            <thead>
              <tr><th>Finding</th><th>Status</th><th>Confidence</th><th>Threshold</th><th>Localization</th></tr>
            </thead>
            <tbody>${findingsRows}</tbody>
          </table>
          <div class="rp-legend">
            <span>Fig. 1 &mdash; Original Radiograph</span>
            <span>Fig. 2 &mdash; Class Activation Heatmap</span>
            <span>Fig. 3 &mdash; Localization Overlay</span>
          </div>
        </div>

        <div class="rp-section">
          <span class="rp-section-title">Impression</span>
          <div class="rp-impression-box">${impression}</div>
        </div>

        <div class="rp-footer">
          <div class="rp-sig-row">
            <div class="rp-sig">
              <div class="rp-sig-line"></div>
              <div class="rp-sig-name">${REPORT_CONFIG.doctors[0].name}</div>
              <div class="rp-sig-role">${REPORT_CONFIG.doctors[0].role}</div>
              <div class="rp-sig-reg">${REPORT_CONFIG.doctors[0].reg}</div>
            </div>
            <div class="rp-sig">
              <div class="rp-sig-line"></div>
              <div class="rp-sig-name">${REPORT_CONFIG.doctors[1].name}</div>
              <div class="rp-sig-role">${REPORT_CONFIG.doctors[1].role}</div>
              <div class="rp-sig-reg">${REPORT_CONFIG.doctors[1].reg}</div>
            </div>
          </div>
          <div class="rp-disclaimer">
            This report is generated by an AI-based research prototype (WristNet, a custom convolutional neural network with class-activation-map
            localization, trained on the GRAZPEDWRI-DX dataset) developed for academic demonstration purposes. It is not a certified diagnostic
            device and has not received regulatory approval for clinical use. All findings must be independently verified by a licensed
            radiologist before any clinical decision is made. This is a computer-generated report and does not require a physical signature.
            Generated at ${fmtTime(now)} on ${fmtDate(now)}. Page 1 of 4.
          </div>
        </div>

      </div>
    </div>`;
  }

  // ------------------------------------------------------------------
  // Pages 2-4 — one figure per page, full size
  // ------------------------------------------------------------------
  function buildFigurePage(pageId, pageNum, figNum, label, imgB64, caption) {
    return `
    <div class="report-page rp-fixed-page" id="${pageId}">
      <div class="rp-outer-border">
        <div class="rp-fig-page">
          <div class="rp-fig-header">
            <div class="rp-fig-header-title">${REPORT_CONFIG.hospitalName}</div>
            <div class="rp-fig-header-sub">Imaging Appendix &middot; Figure ${figNum} of 3</div>
          </div>
          <div class="rp-fig-label">${label}</div>
          <div class="rp-fig-img-wrap">
            <img src="data:image/png;base64,${imgB64}" />
          </div>
          <div class="rp-fig-caption">${caption}</div>
          <div class="rp-fig-footer">WristTrauma-AI &middot; Accession-linked imaging appendix &middot; Page ${pageNum} of 4</div>
        </div>
      </div>
    </div>`;
  }

  // (buildReportPage2/3/4 removed -- figure pages are now generated from
  // getFigureSpecs() for both the HTML preview and the PDF export.)

  // ------------------------------------------------------------------
  // Figure specs -- single source of truth for both the HTML preview and
  // the native-jsPDF export, so the two can't drift apart.
  // `captionPlain` is a plain-text (no HTML entities) version for jsPDF,
  // which renders literal strings and cannot parse &mdash; etc.
  // ------------------------------------------------------------------
  function getFigureSpecs(data) {
    const topClass = data.classes.find((c) => c.key === data.top_class);
    const topLabel = topClass ? topClass.display_name : "Leading Finding";
    return [
      {
        pageId: "reportPage2", pageNum: 2, figNum: 1,
        labelHtml: "Figure 1 &mdash; Original Radiograph",
        labelPlain: "FIGURE 1 - ORIGINAL RADIOGRAPH",
        imgB64: data.original_image_b64,
        caption: "Preprocessed input radiograph, normalized and letterboxed prior to model inference.",
      },
      {
        pageId: "reportPage3", pageNum: 3, figNum: 2,
        labelHtml: "Figure 2 &mdash; Class Activation Heatmap",
        labelPlain: "FIGURE 2 - CLASS ACTIVATION HEATMAP",
        imgB64: data.heatmap_image_b64,
        caption: `Model attention map for the leading finding (${topLabel}). Warmer regions indicate stronger model activation; this is a weak-supervision visualization, not a segmentation mask.`,
      },
      {
        pageId: "reportPage4", pageNum: 4, figNum: 3,
        labelHtml: "Figure 3 &mdash; Localization Overlay",
        labelPlain: "FIGURE 3 - LOCALIZATION OVERLAY",
        imgB64: data.boxed_image_b64,
        caption: "Bounding-box localization derived from the activation map, shown only for the Fracture and Foreign Material findings (single coherent findings). Soft-Tissue/Indirect and Bone Anomaly/Lesion are composite classes and are reported as present/absent only, without a box.",
      },
    ];
  }

  function openReportModal(data) {
    const figures = getFigureSpecs(data);
    reportModalStage.innerHTML =
      buildReportPage1(data) +
      figures
        .map((f) =>
          buildFigurePage(f.pageId, f.pageNum, f.figNum, f.labelHtml, f.imgB64, f.caption)
        )
        .join("");
    reportModal.hidden = false;
    document.body.style.overflow = "hidden";

    downloadPdfBtn.onclick = () => downloadPdf(data);
  }

  function closeReportModal() {
    reportModal.hidden = true;
    document.body.style.overflow = "";
  }

  reportModalClose.addEventListener("click", closeReportModal);
  reportModalBackdrop.addEventListener("click", closeReportModal);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !reportModal.hidden) closeReportModal();
  });

  // ------------------------------------------------------------------
  // Wait for every <img> inside a container to finish loading (or fail)
  // before rasterizing it. Without this, html2canvas can capture a page
  // before its base64 image has decoded, producing a blank image box in
  // the PDF even though the same page displays correctly on screen (the
  // browser had more time to decode it there).
  // ------------------------------------------------------------------
  function waitForImages(container) {
    const imgs = Array.from(container.querySelectorAll("img"));
    return Promise.all(
      imgs.map((img) => {
        if (img.complete && img.naturalWidth > 0) return Promise.resolve();
        return new Promise((resolve) => {
          img.addEventListener("load", resolve, { once: true });
          img.addEventListener("error", resolve, { once: true }); // don't hang the export on a bad image
        });
      })
    );
  }

  // ------------------------------------------------------------------
  // PDF export -- deliberately HYBRID, for reasons that took a while to
  // pin down:
  //
  // PAGE 1 (the complex ruled-table layout) is rasterized with html2canvas,
  // which handles it correctly.
  //
  //   Critically, `windowWidth` / `windowHeight` are NOT passed. Those
  //   options change the virtual viewport html2canvas lays the page out in.
  //   `.report-page` has `max-width:100%`, so a narrower virtual viewport
  //   made the page narrower, reflowed the text, and pushed the rendered
  //   content past 297mm -- which then (correctly) triggered the slicing
  //   branch and emitted a near-empty second page. Letting html2canvas use
  //   the real viewport keeps the capture identical to what's on screen.
  //   The result is also hard-clamped to a single A4 page, since page 1 is
  //   designed to fit one.
  //
  // PAGES 2-4 (the figures) are NOT rasterized at all -- they are drawn
  // natively with jsPDF primitives (rect / text / addImage).
  //
  //   html2canvas clones the document into a hidden iframe and re-resolves
  //   images inside that clone, so waiting for the ORIGINAL <img> elements
  //   to load does not guarantee the clone's copies are decoded at capture
  //   time -- and html2canvas additionally has known-broken support for
  //   `object-fit: contain`, which these images use. That combination is
  //   what produced blank image boxes in the PDF while the same pages
  //   looked correct on screen. Since the raw PNG base64 is already in
  //   hand, jsPDF's addImage draws it directly and deterministically --
  //   no clone, no decode race, no object-fit emulation.
  // ------------------------------------------------------------------

  const A4_W = 210;
  const A4_H = 297;

  function drawFigurePagePdf(pdf, fig) {
    const M = 12;                 // outer page margin
    const boxX = M, boxY = M;
    const boxW = A4_W - M * 2;    // 186
    const boxH = A4_H - M * 2;    // 273

    pdf.setDrawColor(0);
    pdf.setTextColor(0);

    // outer border
    pdf.setLineWidth(0.5);
    pdf.rect(boxX, boxY, boxW, boxH);

    // header
    pdf.setFont("helvetica", "bold");
    pdf.setFontSize(12);
    pdf.text(REPORT_CONFIG.hospitalName.toUpperCase(), A4_W / 2, boxY + 12, { align: "center" });

    pdf.setFont("helvetica", "normal");
    pdf.setFontSize(8);
    pdf.text(
      `Imaging Appendix  -  Figure ${fig.figNum} of 3`,
      A4_W / 2,
      boxY + 17,
      { align: "center" }
    );

    pdf.setLineWidth(0.6);
    pdf.line(boxX + 6, boxY + 21, boxX + boxW - 6, boxY + 21);

    // figure label, in its own bordered box
    pdf.setFont("helvetica", "bold");
    pdf.setFontSize(9.5);
    const labelW = pdf.getTextWidth(fig.labelPlain) + 10;
    const labelH = 7;
    const labelX = (A4_W - labelW) / 2;
    const labelY = boxY + 27;
    pdf.setLineWidth(0.3);
    pdf.rect(labelX, labelY, labelW, labelH);
    pdf.text(fig.labelPlain, A4_W / 2, labelY + 4.8, { align: "center" });

    // image (square), centered, with a border frame around it
    const imgSize = 150;
    const imgX = (A4_W - imgSize) / 2;
    const imgY = labelY + labelH + 8;
    const framePad = 3;
    pdf.setLineWidth(0.5);
    pdf.rect(imgX - framePad, imgY - framePad, imgSize + framePad * 2, imgSize + framePad * 2);
    pdf.addImage(
      `data:image/png;base64,${fig.imgB64}`,
      "PNG",
      imgX,
      imgY,
      imgSize,
      imgSize
    );

    // caption
    pdf.setFont("helvetica", "normal");
    pdf.setFontSize(8);
    const captionY = imgY + imgSize + 10;
    const captionLines = pdf.splitTextToSize(fig.caption, boxW - 20);
    pdf.text(captionLines, A4_W / 2, captionY, { align: "center" });

    // footer
    const footY = boxY + boxH - 8;
    pdf.setLineWidth(0.3);
    pdf.line(boxX + 6, footY - 4, boxX + boxW - 6, footY - 4);
    pdf.setFontSize(7);
    pdf.text(
      `WristTrauma-AI  -  Accession-linked imaging appendix  -  Page ${fig.pageNum} of 4`,
      A4_W / 2,
      footY,
      { align: "center" }
    );
  }

  async function downloadPdf(data) {
    downloadPdfBtn.disabled = true;
    const originalLabel = downloadPdfBtn.innerHTML;
    downloadPdfBtn.innerHTML = "Generating…";

    try {
      const { jsPDF } = window.jspdf;
      const pdf = new jsPDF({ unit: "mm", format: "a4", orientation: "portrait" });

      // ---- Page 1: rasterized ----
      const page1 = document.getElementById("reportPage1");
      await waitForImages(page1);
      const canvas = await html2canvas(page1, {
        scale: 2.5,
        useCORS: true,
        backgroundColor: "#ffffff",
        // no windowWidth/windowHeight -- see note above
      });
      const page1Data = canvas.toDataURL("image/jpeg", 0.96);
      pdf.addImage(page1Data, "JPEG", 0, 0, A4_W, A4_H);

      // ---- Pages 2-4: drawn natively ----
      const figures = getFigureSpecs(data);
      for (const fig of figures) {
        pdf.addPage();
        drawFigurePagePdf(pdf, fig);
      }

      const filename = `WristTrauma-AI-Report-${Date.now()}.pdf`;
      pdf.save(filename);
    } catch (err) {
      alert("PDF generation failed: " + err.message);
    } finally {
      downloadPdfBtn.disabled = false;
      downloadPdfBtn.innerHTML = originalLabel;
    }
  }

  window.WristReport = { openReportModal };
})();