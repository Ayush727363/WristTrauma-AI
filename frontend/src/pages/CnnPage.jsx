import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Brain, ArrowRight, FileCheck2, RotateCcw, AlertTriangle } from 'lucide-react';
import PageTransition from '../components/PageTransition';
import Uploader from '../components/Uploader';
import ScanAnimation from '../components/ScanAnimation';
import ImageGallery from '../components/ImageGallery';
import ImageLightbox from '../components/ImageLightbox';
import FindingsList from '../components/FindingsList';
import PatientRecordPanel from '../components/PatientRecordPanel';
import ReportModal from '../components/ReportModal';
import { analyzeWithCNN } from '../lib/api';
import { CNN_CLASS_ORDER, CNN_CLASS_META } from '../lib/config';

const STATE = { IDLE: 'idle', SCANNING: 'scanning', DONE: 'done', ERROR: 'error' };

export default function CnnPage() {
  const [file, setFile] = useState(null);
  const [state, setState] = useState(STATE.IDLE);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [lightboxOpen, setLightboxOpen] = useState(false);
  const [activeTab, setActiveTab] = useState('original');
  const [reportOpen, setReportOpen] = useState(false);

  const previewUrl = file ? URL.createObjectURL(file) : null;

  async function handleRun() {
    if (!file) return;
    setState(STATE.SCANNING);
    setError(null);
    try {
      const data = await analyzeWithCNN(file);
      setResult(data);
    } catch (e) {
      setError(e.message);
    }
  }

  function handleScanComplete() {
    if (error) {
      setState(STATE.ERROR);
    } else {
      setState(STATE.DONE);
    }
  }

  function reset() {
    setFile(null);
    setResult(null);
    setError(null);
    setState(STATE.IDLE);
  }

  const findings = result
    ? CNN_CLASS_ORDER.map((key) => {
        const c = result.classes.find((x) => x.key === key);
        return {
          key,
          label: CNN_CLASS_META[key].label,
          color: CNN_CLASS_META[key].color,
          probability: c?.probability ?? 0,
          present: c?.present ?? false,
          meta: c?.localizable
            ? c.present
              ? `${c.boxes.length} region(s) localized`
              : 'Localizable, threshold ' + Math.round(c.threshold * 100) + '%'
            : 'Composite finding · no box',
        };
      })
    : [];

  const galleryImages = result
    ? [
        { key: 'original', tag: 'Original', tagColor: '#7fa398', src: `data:image/png;base64,${result.original_image_b64}`, caption: 'Preprocessed input, normalized and letterboxed.' },
        { key: 'heatmap', tag: 'Heatmap', tagColor: '#e0475a', subtitle: CNN_CLASS_META[result.top_class]?.label, src: `data:image/png;base64,${result.heatmap_image_b64}`, caption: 'CAM activation for the leading finding.' },
        { key: 'boxed', tag: 'Localization', tagColor: '#5eead4', src: `data:image/png;base64,${result.boxed_image_b64}`, caption: 'Bounding boxes for fracture & foreign material.' },
      ]
    : [];

  return (
    <PageTransition>
      <section className="relative min-h-screen grad-hero-mesh pt-32 pb-24">
        <div className="max-w-[1100px] mx-auto px-6 lg:px-10">
          {/* header */}
          <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} className="mb-14">
            <div className="chip bg-emerald-400/10 text-emerald-300 border border-emerald-300/25 mb-5">
              <Brain size={13} /> WristNet v1 &middot; Explainable Diagnostic CNN
            </div>
            <h1 className="font-display text-[36px] sm:text-[48px] font-medium text-white leading-[1.08] max-w-[640px]">
              Upload a radiograph to begin screening
            </h1>
            <p className="mt-4 text-white/50 text-[16px] max-w-[560px] leading-relaxed">
              A from-scratch residual CNN trained with focal loss for imbalance robustness.
              Every result includes a Class Activation Map showing exactly where the model looked.
            </p>
          </motion.div>

          <AnimatePresence mode="wait">
            {state === STATE.IDLE && (
              <motion.div key="idle" exit={{ opacity: 0, y: -10 }} className="flex flex-col gap-6">
                <Uploader
                  onFileSelected={setFile}
                  selectedFile={file}
                  onClear={() => setFile(null)}
                  accentColor="#16a37d"
                  hint="Filenames in GRAZPEDWRI-DX format (e.g. 0009_..._WRI-R1_F013) auto-populate the patient record panel."
                />
                <motion.button
                  disabled={!file}
                  onClick={handleRun}
                  whileHover={file ? { scale: 1.01 } : {}}
                  whileTap={file ? { scale: 0.98 } : {}}
                  className="btn-primary self-start !py-4 !px-9 !text-[15.5px]"
                >
                  Run WristNet Analysis <ArrowRight size={18} />
                </motion.button>
              </motion.div>
            )}

            {state === STATE.SCANNING && (
              <motion.div key="scanning" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                <ScanAnimation imageUrl={previewUrl} accentColor="#5eead4" durationMs={3200} onComplete={handleScanComplete} />
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
              <motion.div key="done" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }} className="flex flex-col gap-8">
                <div className="flex items-center justify-between flex-wrap gap-4">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-full bg-emerald-400/15 border border-emerald-300/30 flex items-center justify-center">
                      <FileCheck2 size={17} className="text-emerald-300" />
                    </div>
                    <div>
                      <div className="text-white font-medium text-[15px]">Analysis Complete</div>
                      <div className="text-white/35 text-[12px] font-mono">{result.filename}</div>
                    </div>
                  </div>
                  <div className="flex gap-3">
                    <button onClick={reset} className="btn-ghost-dark !text-[13.5px]">
                      <RotateCcw size={14} /> New Analysis
                    </button>
                    <button onClick={() => setReportOpen(true)} className="btn-primary !text-[13.5px] !py-3 !px-6">
                      View Report
                    </button>
                  </div>
                </div>

                <ImageGallery images={galleryImages} onOpen={(key) => { setActiveTab(key); setLightboxOpen(true); }} />

                <div className="grid md:grid-cols-[1.4fr_1fr] gap-6 items-start">
                  <div>
                    <h3 className="text-white font-semibold text-[14.5px] mb-4">Detected Findings</h3>
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
        <ImageLightbox
          open={lightboxOpen}
          onClose={() => setLightboxOpen(false)}
          activeTab={activeTab}
          onTabChange={setActiveTab}
          tabs={galleryImages.map((img) => ({ key: img.key, label: img.tag, src: img.src }))}
        />
      )}

      {result && (
        <ReportModal
          open={reportOpen}
          onClose={() => setReportOpen(false)}
          data={result}
          modelName="WristNet v1 (Custom CNN)"
          classOrder={CNN_CLASS_ORDER}
          classMeta={CNN_CLASS_META}
        />
      )}
    </PageTransition>
  );
}
