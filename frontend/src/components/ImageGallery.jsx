import { motion } from 'framer-motion';
import { Maximize2 } from 'lucide-react';

export default function ImageGallery({ images, onOpen }) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
      {images.map((img, i) => (
        <motion.div
          key={img.key}
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: i * 0.1, duration: 0.5 }}
          onClick={() => onOpen(img.key)}
          className="group cursor-zoom-in rounded-2xl overflow-hidden border border-white/8 bg-white/[0.02]"
        >
          <div className="flex items-center justify-between px-4 pt-3.5 pb-2">
            <span
              className="text-[10.5px] font-bold uppercase tracking-wider px-2.5 py-1 rounded-md"
              style={{ background: `${img.tagColor}22`, color: img.tagColor }}
            >
              {img.tag}
            </span>
            {img.subtitle && <span className="text-white/35 text-[11px] font-mono">{img.subtitle}</span>}
          </div>
          <div className="relative aspect-square mx-4 mb-3 rounded-xl overflow-hidden bg-black">
            <img src={img.src} alt={img.tag} className="w-full h-full object-contain transition-transform duration-500 group-hover:scale-105" />
            <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity bg-black/20 flex items-center justify-center">
              <div className="w-10 h-10 rounded-full bg-white/15 backdrop-blur-sm flex items-center justify-center">
                <Maximize2 size={16} className="text-white" />
              </div>
            </div>
          </div>
          <p className="px-4 pb-4 text-white/35 text-[11.5px] leading-relaxed">{img.caption}</p>
        </motion.div>
      ))}
    </div>
  );
}
