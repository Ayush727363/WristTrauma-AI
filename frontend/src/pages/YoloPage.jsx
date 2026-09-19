import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Bone, ArrowRight, FileCheck2, RotateCcw, AlertTriangle, Maximize2 } from 'lucide-react';
import PageTransition from '../components/PageTransition';
import Uploader from '../components/Uploader';
import ScanAnimation from '../components/ScanAnimation';
import ImageLightbox from '../components/ImageLightbox';
import FindingsList from '../components/FindingsList';
import PatientRecordPanel from '../components/PatientRecordPanel';
import ReportModal from '../components/ReportModal';
import { analyzeWithYOLO } from '../lib/api';
import { YOLO_CLASS_ORDER, YOLO_CLASS_META } from '../lib/config';

const STATE = { IDLE: 'idle', SCANNING: 'scanning', DONE: 'done', ERROR: 'error' };

export default function YoloPage() {
  const [file, setFile] = useState(null);
  const [state, setState] = useState(STATE.IDLE);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [lightboxOpen, setLightboxOpen] = useState(false);
  const [activeTab, setActiveTab] = useState('boxed');
  const [reportOpen, setReportOpen] = useState(false);

  const previewUrl = file ? URL.createObjectURL(file) : null;

  async function handleRun() {
    if (!file) return;
    setState(STATE.SCANNING);
    setError(null);
    try {
      const data = await analyzeWithYOLO(file);
      setResult(data);
    } catch (e) {
      setError(e.message);
    }
  }

  function handleScanComplete() {
    setState(error ? STATE.ERROR : STATE.DONE);
  }

  function reset() {
    setFile(null);
    setResult(null);
    setError(null);
    setState(STATE.IDLE);
  }

  const findings = result
    ? YOLO_CLASS_ORDER.map((key) => {
        const c = result.classes.find((x) => x.key === key);
        return {
          key,
          label: YOLO_CLASS_META[key].label,
          color: YOLO_CLASS_META[key].color,
          probability: c?.probability ?? 0,
          present: c?.present ?? false,
          meta: c?.present ? `${c.boxes?.length ?? 0} box(es) detected` : 'Not detected',
        };
      })
    : [];

  const tabs = result
    ? [
        { key: 'original', label: 'Original', src: `data:image/png;base64,${result.original_image_b64}` },
        { key: 'boxed', label: 'Detections', src: `data:image/png;base64,${result.boxed_image_b64}` },
      ]
    : [];

  return (
    <PageTransition>
      <section className="relative min-h-screen overflow-hidden pt-32 pb-24" style={{ background: 'linear-gradient(180deg, #04120f, #1a1006 60%, #04120f)' }}>
        <div className="max-w-[1100px] mx-auto px-6 lg:px-10">
          <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} className="mb-14">
            <div className="chip mb-5" style={{ background: 'rgba(212,169,74,0.1)', color: '#d4a94a', border: '1px solid rgba(212,169,74,0.25)' }}>
              <Bone size={13} /> YOLO26s &middot; Multi-Class Object Detector
            </div>
            <h1 className="font-display text-[36px] sm:text-[48px] font-medium text-white leading-[1.08] max-w-[640px]">
              Full 9-class detection &amp; localization
            </h1>
            <p className="mt-4 text-white/50 text-[16px] max-w-[620px] leading-relaxed">
              A modern single-stage detector trained on the complete GRAZPEDWRI-DX taxonomy &mdash;
              drawing precise bounding boxes for every annotated finding, from fractures to periosteal
              reactions to incidental foreign bodies.
            </p>
          </motion.div>

          <AnimatePresence mode="wait">
            {state === STATE.IDLE && (
              <motion.div key="idle" exit={{ opacity: 0, y: -10 }} className="flex flex-col gap-6">
                <Uploader
                  onFileSelected={setFile}
                  selectedFile={file}
                  onClear={() => setFile(null)}
                  accentColor="#d4a94a"
                  hint="This model outputs bounding boxes across all 9 GRAZPEDWRI-DX classes, unmerged."
                />
                <motion.button
                  disabled={!file}
                  onClick={handleRun}
                  whileHover={file ? { scale: 1.01 } : {}}
                  whileTap={file ? { scale: 0.98 } : {}}
                  className="self-start inline-flex items-center gap-2.5 text-white font-semibold text-[15.5px] py-4 px-9 rounded-full disabled:opacity-40 transition-all"
                  style={{ background: 'linear-gradient(135deg, #d4a94a, #b8862f)', boxShadow: '0 8px 40px -8px rgba(184,134,47,0.5)' }}
                >
                  Run YOLO26 Detection <ArrowRight size={18} />
                </motion.button>
              </motion.div>
            )}

            {state === STATE.SCANNING && (
              <motion.div key="scanning" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                <ScanAnimation imageUrl={previewUrl} accentColor="#d4a94a" durationMs={3200} onComplete={handleScanComplete} />
              </motion.div>
            )}

            {state === STATE.ERROR && (
              <motion.div key="error" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="rounded-2xl border border-red-400/25 bg-red-400/8 p-8 flex flex-col items-center text-center gap-4">
                <AlertTriangle size={28} className="text-red-300" />
                <div>
                  <p className="text-white font-medium">Analysis failed</p>
                  <p className="text-white/50 text-[13.5px] mt-1 max-w-md">{error}</p>
                </div>
                <button onClick={reset} className="btn-ghost-dark !text-[13px]">
                  <RotateCcw size={14} /> Try Again
                </button>
              </motion.div>
            )}

            {state === STATE.DONE && result && (
              <motion.div key="done" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} className="flex flex-col gap-8">
                <div className="flex items-center justify-between flex-wrap gap-4">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-full flex items-center justify-center" style={{ background: 'rgba(212,169,74,0.15)', border: '1px solid rgba(212,169,74,0.3)' }}>
                      <FileCheck2 size={17} style={{ color: '#d4a94a' }} />
                    </div>
                    <div>
                      <div className="text-white font-medium text-[15px]">Detection Complete</div>
                      <div className="text-white/35 text-[12px] font-mono">{result.filename}</div>
                    </div>
                  </div>
                  <div className="flex gap-3">
                    <button onClick={reset} className="btn-ghost-dark !text-[13.5px]">
                      <RotateCcw size={14} /> New Analysis
                    </button>
                    <button
                      onClick={() => setReportOpen(true)}
                      className="inline-flex items-center gap-2 text-white font-semibold text-[13.5px] py-3 px-6 rounded-full"
                      style={{ background: 'linear-gradient(135deg, #d4a94a, #b8862f)' }}
                    >
                      View Report
                    </button>
                  </div>
                </div>

                {/* Boxed detection image, large */}
                <div
                  onClick={() => { setActiveTab('boxed'); setLightboxOpen(true); }}
                  className="group cursor-zoom-in rounded-2xl overflow-hidden border border-white/8 bg-white/[0.02] max-w-[440px]"
                >
                  <div className="px-4 pt-3.5 pb-2 flex items-center gap-2">
                    <span className="text-[10.5px] font-bold uppercase tracking-wider px-2.5 py-1 rounded-md" style={{ background: 'rgba(212,169,74,0.18)', color: '#d4a94a' }}>
                      All Detections
                    </span>
                  </div>
                  <div className="relative aspect-square mx-4 mb-3 rounded-xl overflow-hidden bg-black">
                    <img src={`data:image/png;base64,${result.boxed_image_b64}`} alt="Detections" className="w-full h-full object-contain transition-transform duration-500 group-hover:scale-105" />
                    <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity bg-black/20 flex items-center justify-center">
                      <div className="w-10 h-10 rounded-full bg-white/15 backdrop-blur-sm flex items-center justify-center">
                        <Maximize2 size={16} className="text-white" />
                      </div>
                    </div>
                  </div>
                </div>

                <div className="grid md:grid-cols-[1.4fr_1fr] gap-6 items-start">
                  <div>
                    <h3 className="text-white font-semibold text-[14.5px] mb-4">Detected Classes (9-Class Taxonomy)</h3>
                    <FindingsList items={findings} />
                  </div>
                  <PatientRecordPanel record={result.patient_record} />
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </section>

      {result && (
        <ImageLightbox open={lightboxOpen} onClose={() => setLightboxOpen(false)} activeTab={activeTab} onTabChange={setActiveTab} tabs={tabs} />
      )}

      {result && (
        <ReportModal
          open={reportOpen}
          onClose={() => setReportOpen(false)}
          data={result}
          modelName="YOLO26s (9-Class Detector)"
          classOrder={YOLO_CLASS_ORDER}
          classMeta={YOLO_CLASS_META}
        />
      )}
    </PageTransition>
  );
}
