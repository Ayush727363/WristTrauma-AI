import { motion } from 'framer-motion';

export default function FindingsList({ items }) {
  return (
    <div className="flex flex-col gap-3">
      {items.map((item, i) => (
        <motion.div
          key={item.key}
          initial={{ opacity: 0, x: -12 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: i * 0.08, duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
          className="relative rounded-2xl p-4 border overflow-hidden"
          style={{
            borderColor: item.present ? `${item.color}40` : 'rgba(255,255,255,0.08)',
            background: item.present ? `${item.color}0d` : 'rgba(255,255,255,0.02)',
          }}
        >
          <div className="flex items-center gap-3">
            <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ background: item.color }} />
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-white font-medium text-[14px]">{item.label}</span>
                <span
                  className="text-[9.5px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-md"
                  style={{
                    background: item.present ? `${item.color}25` : 'rgba(255,255,255,0.06)',
                    color: item.present ? item.color : 'rgba(255,255,255,0.4)',
                  }}
                >
                  {item.present ? 'Present' : 'Absent'}
                </span>
              </div>
              {item.meta && <div className="text-white/35 text-[11px] mt-1 font-mono">{item.meta}</div>}
              <div className="mt-2.5 h-[5px] rounded-full bg-white/8 overflow-hidden">
                <motion.div
                  className="h-full rounded-full"
                  style={{ background: item.color }}
                  initial={{ width: 0 }}
                  animate={{ width: `${item.probability * 100}%` }}
                  transition={{ delay: i * 0.08 + 0.2, duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
                />
              </div>
            </div>
            <span className="font-mono font-semibold text-[15px] text-white shrink-0 w-14 text-right">
              {Math.round(item.probability * 100)}%
            </span>
          </div>
        </motion.div>
      ))}
    </div>
  );
}
