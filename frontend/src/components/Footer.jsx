import { Link } from 'react-router-dom';
import { Activity, ShieldCheck, Mail } from 'lucide-react';
import { HOSPITAL } from '../lib/config';

export default function Footer() {
  return (
    <footer className="relative bg-[#04120f] border-t border-emerald-300/10 pt-16 pb-10">
      <div className="max-w-[1320px] mx-auto px-6 lg:px-10">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-12 pb-12 border-b border-white/5">
          <div className="md:col-span-2">
            <div className="flex items-center gap-3 mb-4">
              <div className="relative w-9 h-9 flex items-center justify-center rounded-full bg-gradient-to-br from-emerald-400 to-emerald-600">
                <Activity className="w-4.5 h-4.5 text-white" strokeWidth={2.4} size={18} />
              </div>
              <span className="font-display text-lg font-semibold text-white">{HOSPITAL.shortName}</span>
            </div>
            <p className="text-white/45 text-[14px] leading-relaxed max-w-sm">
              {HOSPITAL.department}. An academic research deployment demonstrating weakly-supervised
              deep learning for pediatric wrist trauma screening.
            </p>
          </div>

          <div>
            <div className="text-white/85 text-[13px] font-semibold uppercase tracking-wider mb-4">Platform</div>
            <ul className="flex flex-col gap-3">
              <li><Link to="/cnn" className="text-white/45 hover:text-emerald-300 text-[14px] transition-colors">WristNet CNN</Link></li>
              <li><Link to="/yolo" className="text-white/45 hover:text-emerald-300 text-[14px] transition-colors">YOLO26 Detector</Link></li>
              <li><Link to="/about" className="text-white/45 hover:text-emerald-300 text-[14px] transition-colors">Methodology</Link></li>
            </ul>
          </div>

          <div>
            <div className="text-white/85 text-[13px] font-semibold uppercase tracking-wider mb-4">Compliance</div>
            <div className="flex items-start gap-2.5 text-white/45 text-[13px] leading-relaxed">
              <ShieldCheck size={16} className="mt-0.5 shrink-0 text-emerald-400/70" />
              <span>Research prototype. Not a certified diagnostic device. Not approved for independent clinical use.</span>
            </div>
          </div>
        </div>

        <div className="pt-8 flex flex-col md:flex-row items-center justify-between gap-4">
          <span className="text-white/30 text-[12.5px]">
            &copy; {new Date().getFullYear()} {HOSPITAL.name} &middot; Academic Demonstration Deployment
          </span>
          <span className="text-white/30 text-[12.5px] font-mono">GRAZPEDWRI-DX &middot; WristNet v1 &middot; YOLO26s</span>
        </div>
      </div>
    </footer>
  );
}
