import { useCallback, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, X, FileImage, Sparkles } from 'lucide-react';

export default function Uploader({ onFileSelected, selectedFile, onClear, accentColor = '#16a37d', hint }) {
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef(null);
  const previewUrl = selectedFile ? URL.createObjectURL(selectedFile) : null;

  const handleFiles = useCallback(
    (files) => {
      const file = files?.[0];
      if (!file) return;
      const isValid = /^image\/(png|jpe?g|bmp|tiff?)$/i.test(file.type) || /\.(png|jpe?g|bmp|tiff?)$/i.test(file.name);
      if (!isValid) {
        alert('Please select a PNG, JPG, BMP, or TIFF image.');
        return;
      }
      onFileSelected(file);
    },
    [onFileSelected]
  );

  return (
    <div className="w-full">
      <AnimatePresence mode="wait">
        {!selectedFile ? (
          <motion.div
            key="empty"
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => { e.preventDefault(); setDragOver(false); handleFiles(e.dataTransfer.files); }}
            onClick={() => inputRef.current?.click()}
            className="relative cursor-pointer rounded-3xl border-2 border-dashed transition-all duration-300 overflow-hidden"
            style={{
              borderColor: dragOver ? accentColor : 'rgba(255,255,255,0.14)',
              background: dragOver ? `${accentColor}0d` : 'rgba(255,255,255,0.03)',
            }}
          >
            <input
              ref={inputRef}
              type="file"
              accept="image/png,image/jpeg,image/jpg,image/bmp,image/tiff"
              className="hidden"
              onChange={(e) => handleFiles(e.target.files)}
            />

            <div className="relative flex flex-col items-center justify-center gap-5 px-8 py-16 sm:py-20">
              <motion.div
                animate={{ y: dragOver ? -6 : 0, scale: dragOver ? 1.08 : 1 }}
                transition={{ type: 'spring', stiffness: 300, damping: 20 }}
                className="w-16 h-16 rounded-2xl flex items-center justify-center"
                style={{ background: `${accentColor}18`, border: `1px solid ${accentColor}35` }}
              >
                <Upload size={26} style={{ color: accentColor }} strokeWidth={1.8} />
              </motion.div>

              <div className="text-center">
                <p className="text-white font-medium text-[16px]">
                  Drag &amp; drop a radiograph, or{' '}
                  <span style={{ color: accentColor }} className="underline underline-offset-4">browse files</span>
                </p>
                <p className="text-white/40 text-[13px] mt-2">PNG, JPG, BMP, TIFF &middot; 16-bit imaging supported</p>
              </div>

              {hint && (
                <div className="mt-2 flex items-start gap-2 text-white/35 text-[12px] max-w-[420px] text-center leading-relaxed">
                  <Sparkles size={13} className="shrink-0 mt-0.5" style={{ color: accentColor }} />
                  <span>{hint}</span>
                </div>
              )}
            </div>

            {/* animated border glow on drag */}
            {dragOver && (
              <motion.div
                className="absolute inset-0 pointer-events-none rounded-3xl"
                style={{ boxShadow: `inset 0 0 40px ${accentColor}30` }}
                initial={{ opacity: 0 }} animate={{ opacity: 1 }}
              />
            )}
          </motion.div>
        ) : (
          <motion.div
            key="filled"
            initial={{ opacity: 0, scale: 0.97 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.97 }}
            className="relative rounded-3xl overflow-hidden border border-white/10 bg-white/[0.03] p-5 flex items-center gap-5"
          >
            <div className="w-20 h-20 rounded-xl overflow-hidden shrink-0 border border-white/10 bg-black">
              <img src={previewUrl} alt="Selected" className="w-full h-full object-cover" />
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 text-white/85">
                <FileImage size={15} style={{ color: accentColor }} />
                <span className="text-[14px] font-medium truncate font-mono">{selectedFile.name}</span>
              </div>
              <span className="text-white/35 text-[12px] mt-1 block">
                {(selectedFile.size / 1024).toFixed(0)} KB &middot; Ready for analysis
              </span>
            </div>
            <button
              onClick={onClear}
              className="shrink-0 w-9 h-9 rounded-full flex items-center justify-center bg-white/5 hover:bg-white/10 text-white/50 hover:text-white transition-colors"
              aria-label="Remove file"
            >
              <X size={16} />
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
