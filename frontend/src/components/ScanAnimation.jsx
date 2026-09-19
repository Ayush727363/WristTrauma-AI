import { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const STAGES = [
  { label: 'Calibrating imaging pipeline', sub: 'Normalizing radiographic density' },
  { label: 'Neural inference in progress', sub: 'Forward pass through convolutional layers' },
  { label: 'Extracting activation topology', sub: 'Computing class-conditional attention maps' },
  { label: 'Cross-referencing localization', sub: 'Resolving finding boundaries' },
  { label: 'Finalizing diagnostic synthesis', sub: 'Compiling structured output' },
];

/**
 * Full cinematic scan sequence, ~3s by default. Renders the uploaded image
 * inside a scanning rig with a moving beam, orbiting corner brackets,
 * a live particle field, and a rotating status readout -- deliberately
 * over-the-top ("2050 sci-fi") per the brief, not a generic spinner.
 */
export default function ScanAnimation({ imageUrl, accentColor = '#5eead4', durationMs = 3200, onComplete }) {
  const [stageIdx, setStageIdx] = useState(0);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const stageTime = durationMs / STAGES.length;
    const stageTimer = setInterval(() => {
      setStageIdx((i) => Math.min(i + 1, STAGES.length - 1));
    }, stageTime);

    const start = Date.now();
    const progressTimer = setInterval(() => {
      const pct = Math.min(100, ((Date.now() - start) / durationMs) * 100);
      setProgress(pct);
      if (pct >= 100) {
        clearInterval(progressTimer);
        clearInterval(stageTimer);
        setTimeout(onComplete, 250);
      }
    }, 30);

    return () => {
      clearInterval(stageTimer);
      clearInterval(progressTimer);
    };
  }, [durationMs, onComplete]);

  return (
    <div className="relative w-full flex flex-col items-center justify-center py-10">
      {/* ambient glow behind rig */}
      <div
        className="absolute w-[600px] h-[600px] rounded-full blur-[100px] opacity-25 pointer-events-none"
        style={{ background: `radial-gradient(circle, ${accentColor}, transparent 70%)` }}
      />

      <div className="relative w-[300px] h-[300px] sm:w-[380px] sm:h-[380px]">
        {/* rotating outer ring */}
        <motion.svg
          viewBox="0 0 200 200"
          className="absolute inset-0 w-full h-full"
          animate={{ rotate: 360 }}
          transition={{ duration: 20, repeat: Infinity, ease: 'linear' }}
        >
          <circle cx="100" cy="100" r="96" fill="none" stroke={accentColor} strokeOpacity="0.15" strokeWidth="0.5" strokeDasharray="2 6" />
        </motion.svg>
        <motion.svg
          viewBox="0 0 200 200"
          className="absolute inset-0 w-full h-full"
          animate={{ rotate: -360 }}
          transition={{ duration: 14, repeat: Infinity, ease: 'linear' }}
        >
          <circle cx="100" cy="100" r="88" fill="none" stroke={accentColor} strokeOpacity="0.25" strokeWidth="0.5" strokeDasharray="0.5 4" />
        </motion.svg>

        {/* corner brackets, pulsing */}
        {[
          { top: 8, left: 8, rotate: 0 },
          { top: 8, right: 8, rotate: 90 },
          { bottom: 8, right: 8, rotate: 180 },
          { bottom: 8, left: 8, rotate: 270 },
        ].map((pos, i) => (
          <motion.svg
            key={i}
            viewBox="0 0 24 24"
            className="absolute w-6 h-6"
            style={{ ...pos, transform: `rotate(${pos.rotate}deg)` }}
            animate={{ opacity: [0.4, 1, 0.4] }}
            transition={{ duration: 2, repeat: Infinity, delay: i * 0.2 }}
          >
            <path d="M2 8V2H8" stroke={accentColor} strokeWidth="2" fill="none" strokeLinecap="round" />
          </motion.svg>
        ))}

        {/* image frame */}
        <div
          className="absolute inset-[14%] rounded-lg overflow-hidden border"
          style={{ borderColor: `${accentColor}33`, boxShadow: `0 0 40px -8px ${accentColor}55` }}
        >
          {imageUrl && (
            <img
              src={imageUrl}
              alt="Analyzing"
              className="w-full h-full object-cover"
              style={{ filter: 'grayscale(1) contrast(1.15) brightness(0.9)' }}
            />
          )}
          <div className="absolute inset-0" style={{ background: `linear-gradient(180deg, transparent 60%, ${accentColor}22)` }} />

          {/* horizontal scan beam */}
          <motion.div
            className="absolute left-0 right-0 h-[2px]"
            style={{
              background: `linear-gradient(90deg, transparent, ${accentColor}, transparent)`,
              boxShadow: `0 0 20px 3px ${accentColor}`,
            }}
            animate={{ top: ['0%', '100%', '0%'] }}
            transition={{ duration: 2.2, repeat: Infinity, ease: 'easeInOut' }}
          />

          {/* scanline grid overlay */}
          <div
            className="absolute inset-0 opacity-20 pointer-events-none"
            style={{
              backgroundImage: `repeating-linear-gradient(0deg, ${accentColor}22 0px, transparent 1px, transparent 3px)`,
            }}
          />

          {/* random flicker "detection" pips */}
          <AnimatePresence>
            {progress > 30 && (
              <motion.div
                key="pip1"
                className="absolute w-2.5 h-2.5 rounded-full border-2"
                style={{ borderColor: accentColor, top: '35%', left: '42%' }}
                initial={{ scale: 0, opacity: 0 }}
                animate={{ scale: [0, 1.4, 1], opacity: [0, 1, 0.7] }}
                transition={{ duration: 0.6 }}
              />
            )}
            {progress > 55 && (
              <motion.div
                key="pip2"
                className="absolute w-2 h-2 rounded-full border-2"
                style={{ borderColor: accentColor, top: '58%', left: '60%' }}
                initial={{ scale: 0, opacity: 0 }}
                animate={{ scale: [0, 1.4, 1], opacity: [0, 1, 0.7] }}
                transition={{ duration: 0.6 }}
              />
            )}
          </AnimatePresence>
        </div>

        {/* progress ring */}
        <svg viewBox="0 0 200 200" className="absolute inset-0 w-full h-full -rotate-90">
          <circle cx="100" cy="100" r="94" fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="1.5" />
          <motion.circle
            cx="100" cy="100" r="94" fill="none" stroke={accentColor} strokeWidth="1.5" strokeLinecap="round"
            strokeDasharray={2 * Math.PI * 94}
            strokeDashoffset={2 * Math.PI * 94 * (1 - progress / 100)}
            style={{ filter: `drop-shadow(0 0 6px ${accentColor})` }}
          />
        </svg>
      </div>

      {/* status readout */}
      <div className="mt-10 text-center min-h-[70px]">
        <AnimatePresence mode="wait">
          <motion.div
            key={stageIdx}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.3 }}
          >
            <div className="font-mono text-[11px] uppercase tracking-[0.18em] mb-1.5" style={{ color: accentColor }}>
              {String(Math.round(progress)).padStart(3, '0')}% &middot; Stage {stageIdx + 1}/{STAGES.length}
            </div>
            <div className="text-white font-medium text-[16px]">{STAGES[stageIdx].label}</div>
            <div className="text-white/40 text-[13px] mt-1">{STAGES[stageIdx].sub}</div>
          </motion.div>
        </AnimatePresence>
      </div>

      {/* progress bar */}
      <div className="mt-6 w-full max-w-[320px] h-[3px] rounded-full bg-white/10 overflow-hidden">
        <motion.div
          className="h-full rounded-full"
          style={{ background: `linear-gradient(90deg, ${accentColor}, transparent 200%)`, width: `${progress}%` }}
        />
      </div>
    </div>
  );
}
