import { motion } from 'framer-motion';
import { UserRound, Info } from 'lucide-react';

export default function PatientRecordPanel({ record }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3, duration: 0.5 }}
      className="rounded-2xl border border-white/8 bg-white/[0.02] p-6"
    >
      <div className="flex items-center gap-2.5 mb-5">
        <UserRound size={17} className="text-emerald-300/70" />
        <h3 className="text-white font-semibold text-[14.5px]">Patient Record</h3>
      </div>

      {record ? (
        <div className="grid grid-cols-2 gap-4">
          <Field label="Age" value={record.age != null ? `${record.age} yrs` : '—'} />
          <Field label="Sex" value={record.sex || '—'} />
          <Field label="Side" value={record.side || '—'} />
          <Field label="Projection" value={record.projection || '—'} />
        </div>
      ) : (
        <p className="text-white/35 text-[13px] leading-relaxed">
          No structured patient record could be parsed from this filename. Rename using the GRAZPEDWRI-DX
          convention (e.g. <code className="font-mono text-emerald-300/70">0009_..._WRI-R1_F013</code>) to
          auto-populate this panel.
        </p>
      )}

      <div className="mt-5 pt-4 border-t border-white/8 flex items-start gap-2 text-white/30 text-[11px] leading-relaxed">
        <Info size={13} className="shrink-0 mt-0.5" />
        Parsed from file metadata &mdash; not inferred by the model.
      </div>
    </motion.div>
  );
}

function Field({ label, value }) {
  return (
    <div>
      <div className="text-white/35 text-[10px] uppercase tracking-wider font-semibold mb-1">{label}</div>
      <div className="text-white font-medium text-[15px]">{value}</div>
    </div>
  );
}
