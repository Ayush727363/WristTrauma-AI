import { useRef } from 'react';
import { createPortal } from 'react-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Download } from 'lucide-react';
import { HOSPITAL, DOCTORS } from '../lib/config';

function fmtDate(d) {
  return d.toLocaleDateString('en-IN', { day: '2-digit', month: 'short', year: 'numeric' });
}
function fmtTime(d) {
  return d.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' });
}

const CNN_IMPRESSIONS = {
  fracture: 'A cortical discontinuity consistent with an acute fracture is identified, localized on the accompanying activation map and bounding-box overlay.',
  softtissue_indirect: 'Soft-tissue swelling and/or indirect radiographic signs (periosteal reaction / pronator fat-pad sign) are noted, suggestive of underlying trauma.',
  foreign_material: 'A radio-opaque foreign material / metallic density is identified within the imaging field, localized on the accompanying overlay.',
  bone_lesion: 'A focal bone density abnormality is flagged with low model confidence. Given limited training support for this finding, correlation with clinical history and dedicated radiologist review is strongly advised.',
};

/**
 * Report modal used by both the CNN and YOLO pages. `data` shape differs
 * slightly between the two backends -- this component normalizes via the
 * classOrder/classMeta props rather than assuming a fixed 4-class shape.
 */
export default function ReportModal({ open, onClose, data, modelName, classOrder, classMeta }) {
  const accessionRef = useRef(`WTA/${new Date().getFullYear()}/${Math.floor(1000 + Math.random() * 9000)}`);

  if (!open || !data) return null;

  const now = new Date();
  const record = data.patient_record;

  const rows = classOrder
    .map((key) => data.classes.find((c) => c.key === key))
    .filter(Boolean);

  function buildImpression() {
    const present = rows.filter((c) => c.present);
    if (present.length === 0) {
      return 'No AI-flagged abnormal findings were identified on this radiograph at the applied confidence thresholds. Clinical correlation is nonetheless advised, as subtle non-displaced fractures can be radiographically occult.';
    }
    const sentences = present
      .sort((a, b) => b.probability - a.probability)
      .map((c) => CNN_IMPRESSIONS[c.key] || `${classMeta[c.key]?.label || c.key} flagged as present.`);
    return sentences.join(' ') + ' Findings are AI-generated and require confirmation by a qualified radiologist prior to clinical action.';
  }

  function handlePrint() {
    window.print();
  }

  return createPortal(
    <AnimatePresence>
      <motion.div id="reportModalRoot" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="fixed inset-0 z-[200] flex items-center justify-center p-4 print:static print:block print:p-0">
        <div className="absolute inset-0 bg-black/92 backdrop-blur-md print:hidden" onClick={onClose} />

        <motion.div
          initial={{ opacity: 0, scale: 0.97 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.97 }}
          className="relative z-10 w-full max-w-[880px] h-[90vh] rounded-2xl overflow-hidden bg-[#2b2b2b] flex flex-col print:static print:h-auto print:w-auto print:max-w-none print:rounded-none print:overflow-visible print:flex-none"
        >
          <div className="flex items-center justify-between px-5 py-3.5 bg-[#1a1a1a] border-b border-white/10 print:hidden">
            <span className="text-white/80 text-[13px] font-semibold">Report Preview</span>
            <div className="flex items-center gap-2">
              <button onClick={handlePrint} className="btn-primary !py-2 !px-4 !text-[12.5px]">
                <Download size={14} /> Download / Print PDF
              </button>
              <button onClick={onClose} className="text-white/50 hover:text-white p-1.5 rounded-lg hover:bg-white/10">
                <X size={18} />
              </button>
            </div>
          </div>

          <div id="report-print-area" className="flex-1 overflow-y-auto bg-[#2b2b2b] p-7 flex flex-col items-center gap-6 print:flex-none print:overflow-visible print:p-0 print:bg-white print:gap-0 print:block">
            {/* PAGE 1 */}
            <div className="report-page bg-white text-black" style={{ width: '210mm', minHeight: '297mm', padding: '12mm' }}>
              <div className="border-[1.5px] border-black p-[6mm]" style={{ fontFamily: 'Inter, Arial, sans-serif' }}>
                <div className="flex items-center justify-between border-b-2 border-black pb-2.5 mb-2.5">
                  <div className="flex items-center gap-2.5">
                    <div className="w-[42px] h-[42px] rounded-full border-[1.5px] border-black flex items-center justify-center text-[19px] font-bold">
                      {HOSPITAL.emblemText}
                    </div>
                    <div>
                      <div className="text-[17px] font-extrabold uppercase tracking-wide">{HOSPITAL.name}</div>
                      <div className="text-[9.5px] mt-0.5">{HOSPITAL.department}</div>
                      <div className="text-[8.5px] mt-0.5">{HOSPITAL.address}</div>
                    </div>
                  </div>
                  <div className="text-right font-mono text-[9px] leading-relaxed">
                    <div>Accession No: {accessionRef.current}</div>
                    <div>Report Date: {fmtDate(now)}</div>
                    <div>Report Time: {fmtTime(now)}</div>
                  </div>
                </div>

                <div className="text-center border-b border-black pb-2 mb-3">
                  <div className="text-[14px] font-extrabold uppercase tracking-wider">Radiological Analysis Report &mdash; Wrist (AP / Lateral)</div>
                  <div className="text-[9.5px] mt-0.5">AI-Assisted Preliminary Screening &middot; {modelName}</div>
                </div>

                <ReportSection title="Patient Information">
                  <div className="grid grid-cols-4 gap-3 border border-black p-2.5">
                    <Field label="Age" value={record?.age != null ? `${record.age} Years` : 'Not on record'} />
                    <Field label="Sex" value={record?.sex || 'Not on record'} />
                    <Field label="Laterality" value={record?.side || 'Not on record'} />
                    <Field label="Projection" value={record?.projection || 'Not on record'} />
                  </div>
                </ReportSection>

                <ReportSection title="Study Details">
                  <div className="grid grid-cols-4 gap-3 border border-black p-2.5">
                    <Field label="Study Type" value="Wrist Radiograph (X-Ray)" />
                    <Field label="Source File" value={data.filename} mono />
                    <Field label="Analysis Model" value={modelName} />
                    <Field label="Method" value={modelName.includes('YOLO') ? 'Object Detection' : 'Class Activation Mapping'} />
                  </div>
                </ReportSection>

                <ReportSection title="AI-Detected Findings">
                  <table className="w-full border-collapse border border-black text-[10.5px]">
                    <thead>
                      <tr>
                        {['Finding', 'Status', 'Confidence', 'Threshold', 'Localization'].map((h) => (
                          <th key={h} className="border border-black bg-gray-200 text-left px-2 py-1.5 text-[9px] uppercase font-extrabold">{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {rows.map((c) => (
                        <tr key={c.key}>
                          <td className="border border-black px-2 py-1.5">{classMeta[c.key]?.label || c.key}</td>
                          <td className={`border border-black px-2 py-1.5 ${c.present ? 'font-extrabold' : 'font-semibold'}`}>
                            {c.present ? 'POSITIVE' : 'NEGATIVE'}
                          </td>
                          <td className="border border-black px-2 py-1.5">{(c.probability * 100).toFixed(1)}%</td>
                          <td className="border border-black px-2 py-1.5">{Math.round(c.threshold * 100)}%</td>
                          <td className="border border-black px-2 py-1.5">
                            {c.localizable ? (c.present ? `${c.boxes.length} region(s)` : 'N/A') : 'Composite finding'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </ReportSection>

                <ReportSection title="Impression">
                  <div className="border border-black p-3 text-[11px] leading-relaxed">{buildImpression()}</div>
                </ReportSection>

                <div className="flex justify-between mt-9">
                  {DOCTORS.map((doc) => (
                    <div key={doc.name} className="text-center w-[210px]">
                      <div className="border-b border-black h-[30px] mb-1.5" />
                      <div className="text-[11px] font-extrabold">{doc.name}</div>
                      <div className="text-[8.5px] mt-0.5">{doc.role}</div>
                      <div className="text-[7.5px] mt-0.5">{doc.reg}</div>
                    </div>
                  ))}
                </div>

                <div className="mt-4 pt-1.5 border-t border-black text-[7.3px] leading-relaxed text-center">
                  This report is generated by an AI-based research prototype developed for academic demonstration purposes.
                  It is not a certified diagnostic device and has not received regulatory approval for clinical use. All
                  findings must be independently verified by a licensed radiologist before any clinical decision is made.
                  This is a computer-generated report and does not require a physical signature. Generated at {fmtTime(now)} on {fmtDate(now)}. Page 1 of 4.
                </div>
              </div>
            </div>

            {/* PAGES 2-4: figures */}
            {[
              { num: 2, fig: 1, label: 'FIGURE 1 - ORIGINAL RADIOGRAPH', src: data.original_image_b64, caption: 'Preprocessed input radiograph, normalized and letterboxed prior to model inference.' },
              { num: 3, fig: 2, label: 'FIGURE 2 - CLASS ACTIVATION HEATMAP', src: data.heatmap_image_b64, caption: 'Model attention map for the leading finding. Warmer regions indicate stronger activation.' },
              { num: 4, fig: 3, label: 'FIGURE 3 - LOCALIZATION OVERLAY', src: data.boxed_image_b64, caption: 'Bounding-box localization derived from the model output.' },
            ].map((page) => (
              <div key={page.num} className="report-page bg-white text-black" style={{ width: '210mm', height: '297mm', padding: '12mm' }}>
                <div className="border-[1.5px] border-black p-[6mm] h-full flex flex-col items-center" style={{ fontFamily: 'Inter, Arial, sans-serif' }}>
                  <div className="w-full text-center border-b-2 border-black pb-2 mb-3.5">
                    <div className="text-[13px] font-extrabold uppercase tracking-wide">{HOSPITAL.name}</div>
                    <div className="text-[9px] mt-0.5">Imaging Appendix &middot; Figure {page.fig} of 3</div>
                  </div>
                  <div className="border border-black px-3.5 py-1 text-[11px] font-extrabold uppercase tracking-wide mb-3.5">
                    {page.label}
                  </div>
                  <div className="border-[1.5px] border-black p-[6mm] bg-white">
                    <img src={`data:image/png;base64,${page.src}`} alt={page.label} style={{ width: '150mm', height: '150mm', objectFit: 'contain' }} />
                  </div>
                  <p className="text-[9.5px] mt-3.5 text-center max-w-[150mm] leading-relaxed">{page.caption}</p>
                  <div className="mt-auto w-full text-center text-[7.5px] border-t border-black pt-1.5">
                    WristTrauma-AI &middot; Accession-linked imaging appendix &middot; Page {page.num} of 4
                  </div>
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>,
    document.body
  );
}

function ReportSection({ title, children }) {
  return (
    <div className="mb-3">
      <span className="inline-block bg-black text-white text-[10px] font-extrabold uppercase tracking-wide px-2 py-0.5 mb-1.5">
        {title}
      </span>
      {children}
    </div>
  );
}

function Field({ label, value, mono }) {
  return (
    <div>
      <div className="text-[8px] uppercase tracking-wide font-bold text-gray-600">{label}</div>
      <div className={`text-[11.5px] font-semibold mt-0.5 ${mono ? 'font-mono text-[10px] break-all' : ''}`}>{value}</div>
    </div>
  );
}
