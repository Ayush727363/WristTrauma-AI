import { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, ZoomIn, ZoomOut, RotateCcw } from 'lucide-react';

export default function ImageLightbox({ open, onClose, tabs, activeTab, onTabChange }) {
  const [zoom, setZoom] = useState(100);
  const stageRef = useRef(null);

  useEffect(() => {
    if (open) setZoom(100);
  }, [open, activeTab]);

  useEffect(() => {
    function onKey(e) {
      if (e.key === 'Escape') onClose();
    }
    if (open) document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  if (!open) return null;
  const current = tabs.find((t) => t.key === activeTab) || tabs[0];

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
        className="fixed inset-0 z-[200] flex items-center justify-center p-4"
      >
        <div className="absolute inset-0 bg-black/92 backdrop-blur-md" onClick={onClose} />

        <motion.div
          initial={{ opacity: 0, scale: 0.96 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.96 }}
          transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
          className="relative z-10 w-full max-w-[1200px] h-[88vh] rounded-2xl overflow-hidden border border-white/10 bg-[#0a231e] flex flex-col"
        >
          <div className="flex items-center justify-between px-5 py-3.5 border-b border-white/8">
            <div className="flex gap-2">
              {tabs.map((t) => (
                <button
                  key={t.key}
                  onClick={() => onTabChange(t.key)}
                  className="px-4 py-2 rounded-lg text-[13px] font-medium transition-all"
                  style={{
                    background: activeTab === t.key ? 'rgba(94,234,212,0.15)' : 'transparent',
                    color: activeTab === t.key ? '#5eead4' : 'rgba(255,255,255,0.5)',
                    border: activeTab === t.key ? '1px solid rgba(94,234,212,0.3)' : '1px solid transparent',
                  }}
                >
                  {t.label}
                </button>
              ))}
            </div>
            <button onClick={onClose} className="text-white/50 hover:text-white p-1.5 rounded-lg hover:bg-white/5">
              <X size={20} />
            </button>
          </div>

          <div ref={stageRef} className="flex-1 overflow-auto flex items-center justify-center bg-black p-6">
            <img
              src={current?.src}
              alt={current?.label}
              className="max-w-full max-h-full object-contain transition-transform duration-150"
              style={{ transform: `scale(${zoom / 100})` }}
            />
          </div>

          <div className="flex items-center gap-4 px-5 py-3.5 border-t border-white/8 bg-[#071b17]">
            <button
              onClick={() => setZoom((z) => Math.max(100, z - 25))}
              className="w-8 h-8 rounded-lg bg-white/5 hover:bg-white/10 text-white flex items-center justify-center"
            >
              <ZoomOut size={15} />
            </button>
            <input
              type="range" min={100} max={400} value={zoom}
              onChange={(e) => setZoom(Number(e.target.value))}
              className="flex-1 accent-emerald-400"
            />
            <button
              onClick={() => setZoom((z) => Math.min(400, z + 25))}
              className="w-8 h-8 rounded-lg bg-white/5 hover:bg-white/10 text-white flex items-center justify-center"
            >
              <ZoomIn size={15} />
            </button>
            <span className="font-mono text-[12px] text-white/60 w-12 text-right">{zoom}%</span>
            <button
              onClick={() => setZoom(100)}
              className="text-[12px] font-medium text-white/50 hover:text-emerald-300 flex items-center gap-1.5 px-2"
            >
              <RotateCcw size={12} /> Reset
            </button>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}
