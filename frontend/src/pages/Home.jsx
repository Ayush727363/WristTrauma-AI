import { Link } from 'react-router-dom';
import { motion, useScroll, useTransform } from 'framer-motion';
import { useRef } from 'react';
import {
  ArrowRight, Bone, ScanLine, Brain, ShieldCheck, Sparkles,
  Microscope, Target, Layers, ChevronRight, Award, Users, TrendingUp,
} from 'lucide-react';
import PageTransition from '../components/PageTransition';
import { HOSPITAL } from '../lib/config';

const fadeUp = {
  hidden: { opacity: 0, y: 28 },
  show: (i = 0) => ({
    opacity: 1, y: 0,
    transition: { duration: 0.7, delay: i * 0.08, ease: [0.16, 1, 0.3, 1] },
  }),
};

function StatCard({ value, label, icon: Icon, delay }) {
  return (
    <motion.div
      variants={fadeUp} custom={delay} initial="hidden" whileInView="show" viewport={{ once: true, margin: '-60px' }}
      className="glass-panel-dark rounded-2xl p-6 flex flex-col gap-3"
    >
      <Icon size={20} className="text-emerald-300/80" strokeWidth={1.8} />
      <div className="font-display text-3xl md:text-4xl font-semibold text-white tabular-nums">{value}</div>
      <div className="text-white/50 text-[13px] leading-snug">{label}</div>
    </motion.div>
  );
}

function ModelCard({ to, eyebrow, title, description, stats, gradient, icon: Icon, delay }) {
  return (
    <motion.div
      variants={fadeUp} custom={delay} initial="hidden" whileInView="show" viewport={{ once: true, margin: '-80px' }}
      className="group relative"
    >
      <Link to={to} className="block relative rounded-[28px] overflow-hidden h-full">
        <div className={`absolute inset-0 ${gradient} opacity-95`} />
        <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500"
             style={{ background: 'radial-gradient(circle at 30% 20%, rgba(255,255,255,0.12), transparent 60%)' }} />
        <div className="relative p-9 md:p-10 h-full flex flex-col min-h-[420px]">
          <div className="flex items-center justify-between mb-8">
            <div className="w-14 h-14 rounded-2xl bg-white/10 backdrop-blur-sm flex items-center justify-center border border-white/15">
              <Icon size={26} className="text-white" strokeWidth={1.7} />
            </div>
            <span className="chip bg-white/10 text-white/80 border border-white/15 font-mono !text-[11px]">{eyebrow}</span>
          </div>

          <h3 className="font-display text-[28px] md:text-[32px] font-semibold text-white leading-tight mb-3">
            {title}
          </h3>
          <p className="text-white/65 text-[15px] leading-relaxed mb-8 flex-1">
            {description}
          </p>

          <div className="grid grid-cols-3 gap-4 mb-8 pb-8 border-b border-white/10">
            {stats.map((s) => (
              <div key={s.label}>
                <div className="text-white font-display text-xl font-semibold">{s.value}</div>
                <div className="text-white/45 text-[11px] mt-0.5 leading-tight">{s.label}</div>
              </div>
            ))}
          </div>

          <div className="flex items-center gap-2 text-white font-medium text-[14.5px] group-hover:gap-3.5 transition-all">
            Run Screening
            <ArrowRight size={17} strokeWidth={2.2} />
          </div>
        </div>
      </Link>
    </motion.div>
  );
}

export default function Home() {
  const heroRef = useRef(null);
  const { scrollYProgress } = useScroll({ target: heroRef, offset: ['start start', 'end start'] });
  const heroY = useTransform(scrollYProgress, [0, 1], [0, 120]);
  const heroOpacity = useTransform(scrollYProgress, [0, 0.8], [1, 0]);
  const heroScale = useTransform(scrollYProgress, [0, 1], [1, 1.08]);

  return (
    <PageTransition>
      {/* ============ HERO ============ */}
      <section ref={heroRef} className="relative min-h-screen grad-hero-mesh overflow-hidden flex items-center pt-24">
        {/* animated aurora blobs */}
        <motion.div
          className="absolute top-[-10%] left-[10%] w-[500px] h-[500px] rounded-full blur-[120px] opacity-40"
          style={{ background: 'radial-gradient(circle, #5eead4, transparent 70%)' }}
          animate={{ x: [0, 40, 0], y: [0, 30, 0] }}
          transition={{ duration: 14, repeat: Infinity, ease: 'easeInOut' }}
        />
        <motion.div
          className="absolute bottom-[-15%] right-[5%] w-[600px] h-[600px] rounded-full blur-[140px] opacity-30"
          style={{ background: 'radial-gradient(circle, #d4a94a, transparent 70%)' }}
          animate={{ x: [0, -30, 0], y: [0, -40, 0] }}
          transition={{ duration: 18, repeat: Infinity, ease: 'easeInOut' }}
        />

        {/* subtle radiograph silhouette, decorative */}
        <motion.div
          style={{ y: heroY, opacity: heroOpacity, scale: heroScale }}
          className="absolute right-[-5%] top-[18%] w-[520px] h-[520px] hidden lg:block pointer-events-none"
        >
          <WristGlyph />
        </motion.div>

        <div className="relative z-10 max-w-[1320px] mx-auto px-6 lg:px-10 w-full">
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}>
            <div className="chip bg-emerald-400/10 text-emerald-300 border border-emerald-300/25 mb-7">
              <Sparkles size={13} />
              Weakly-Supervised Deep Learning &middot; GRAZPEDWRI-DX
            </div>
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.9, delay: 0.1, ease: [0.16, 1, 0.3, 1] }}
            className="font-display text-[44px] sm:text-[58px] lg:text-[74px] font-medium text-white leading-[1.04] max-w-[820px] tracking-tight"
          >
            Pediatric wrist fractures,{' '}
            <span className="text-gradient-emerald italic">seen clearly</span>{' '}
            in seconds.
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8, delay: 0.25 }}
            className="mt-7 text-white/60 text-[17px] md:text-[19px] leading-relaxed max-w-[600px]"
          >
            {HOSPITAL.name} runs two purpose-built AI models side-by-side on every radiograph &mdash;
            a from-scratch diagnostic CNN with explainable heatmaps, and a modern object detector &mdash;
            so every finding is cross-verified before it reaches a radiologist's desk.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8, delay: 0.4 }}
            className="mt-11 flex flex-wrap items-center gap-4"
          >
            <Link to="/cnn" className="btn-primary !text-[15.5px] !py-4 !px-8">
              Begin Screening <ArrowRight size={18} />
            </Link>
            <Link to="/about" className="btn-ghost-dark">
              How it works
            </Link>
          </motion.div>

          <motion.div
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 1, delay: 0.7 }}
            className="mt-20 grid grid-cols-2 md:grid-cols-4 gap-5 max-w-[820px]"
          >
            <StatCard value="20,327" label="Pediatric radiographs in training cohort" icon={Layers} delay={0} />
            <StatCard value="0.97" label="AUROC on primary fracture detection" icon={Target} delay={1} />
            <StatCard value="2" label="Independent model architectures" icon={Microscope} delay={2} />
            <StatCard value="<2s" label="Median inference time per study" icon={ScanLine} delay={3} />
          </motion.div>
        </div>

        <motion.div
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.2, duration: 1 }}
          className="absolute bottom-10 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 text-white/30"
        >
          <span className="text-[11px] uppercase tracking-[0.2em]">Scroll</span>
          <motion.div animate={{ y: [0, 6, 0] }} transition={{ duration: 1.6, repeat: Infinity }}>
            <ChevronRight size={16} className="rotate-90" />
          </motion.div>
        </motion.div>
      </section>

      {/* ============ TRUST STRIP ============ */}
      <section className="bg-[#f4f1ea] py-14 border-b border-black/5">
        <div className="max-w-[1320px] mx-auto px-6 lg:px-10">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8 items-center">
            {[
              { icon: Award, text: 'Academic Research Grade' },
              { icon: ShieldCheck, text: 'Patient-Level Data Split' },
              { icon: Users, text: 'Radiologist-Reviewed Protocol' },
              { icon: TrendingUp, text: 'Continuously Benchmarked' },
            ].map(({ icon: Icon, text }, i) => (
              <motion.div
                key={text}
                variants={fadeUp} custom={i} initial="hidden" whileInView="show" viewport={{ once: true }}
                className="flex items-center gap-3 justify-center md:justify-start"
              >
                <Icon size={19} className="text-emerald-600 shrink-0" strokeWidth={1.8} />
                <span className="text-[13.5px] font-medium text-[#3c5952]">{text}</span>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ============ TWO MODELS ============ */}
      <section className="bg-[#fbfaf7] py-28">
        <div className="max-w-[1320px] mx-auto px-6 lg:px-10">
          <motion.div variants={fadeUp} initial="hidden" whileInView="show" viewport={{ once: true }} className="max-w-[640px] mb-16">
            <span className="chip bg-emerald-500/8 text-emerald-700 border border-emerald-600/15 mb-5">
              Dual-Model Architecture
            </span>
            <h2 className="font-display text-[36px] md:text-[46px] font-medium text-[#0c1f1a] leading-[1.1] tracking-tight">
              Two independent readings. One confident answer.
            </h2>
            <p className="mt-5 text-[#3c5952] text-[16px] leading-relaxed">
              Rather than relying on a single black box, every radiograph is screened by two
              architecturally distinct systems &mdash; giving radiologists cross-validated evidence,
              not just a probability score.
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 gap-7">
            <ModelCard
              to="/cnn"
              eyebrow="WristNet v1"
              icon={Brain}
              gradient="bg-gradient-to-br from-[#123830] via-[#0d2b24] to-[#04120f]"
              title="Explainable Diagnostic CNN"
              description="A residual convolutional network built from first principles, trained with focal loss for class-imbalance robustness. Every prediction ships with a Class Activation Map, so radiologists see exactly where the model is looking."
              stats={[
                { value: '97.3%', label: 'Fracture AUROC' },
                { value: '4', label: 'Finding classes' },
                { value: 'CAM', label: 'Explainability' },
              ]}
              delay={0}
            />
            <ModelCard
              to="/yolo"
              eyebrow="YOLO26s"
              icon={Bone}
              gradient="bg-gradient-to-br from-[#5c4413] via-[#3d2f16] to-[#04120f]"
              title="Multi-Class Object Detector"
              description="A modern single-stage detector trained on the full 9-class GRAZPEDWRI-DX taxonomy, drawing precise bounding boxes around every annotated finding type &mdash; from fractures to periosteal reactions."
              stats={[
                { value: '9', label: 'Detection classes' },
                { value: 'mAP50', label: 'Localization metric' },
                { value: 'Boxes', label: 'Direct output' },
              ]}
              delay={1}
            />
          </div>
        </div>
      </section>

      {/* ============ PROCESS ============ */}
      <section className="bg-[#0a231e] py-28 relative overflow-hidden">
        <div className="absolute inset-0 opacity-40" style={{ background: 'radial-gradient(ellipse 800px 400px at 50% 0%, rgba(94,234,212,0.08), transparent 70%)' }} />
        <div className="relative max-w-[1320px] mx-auto px-6 lg:px-10">
          <motion.div variants={fadeUp} initial="hidden" whileInView="show" viewport={{ once: true }} className="text-center max-w-[600px] mx-auto mb-20">
            <span className="chip bg-white/8 text-emerald-300 border border-emerald-300/20 mb-5">The Process</span>
            <h2 className="font-display text-[34px] md:text-[42px] font-medium text-white leading-tight">
              From radiograph to report in four steps
            </h2>
          </motion.div>

          <div className="grid md:grid-cols-4 gap-6">
            {[
              { n: '01', title: 'Upload', desc: 'Drop a wrist radiograph — 16-bit DICOM-derived PNGs supported natively.' },
              { n: '02', title: 'Inference', desc: 'The selected model runs preprocessing, forward pass, and CAM extraction in real time.' },
              { n: '03', title: 'Localization', desc: 'Findings are cross-referenced against activation maps and rendered as bounding boxes.' },
              { n: '04', title: 'Report', desc: 'A structured, signable diagnostic report is generated — ready for radiologist review.' },
            ].map((step, i) => (
              <motion.div
                key={step.n}
                variants={fadeUp} custom={i} initial="hidden" whileInView="show" viewport={{ once: true, margin: '-40px' }}
                className="relative pl-0"
              >
                <div className="font-display text-5xl font-light text-emerald-400/25 mb-4">{step.n}</div>
                <h4 className="text-white font-semibold text-[17px] mb-2">{step.title}</h4>
                <p className="text-white/45 text-[14px] leading-relaxed">{step.desc}</p>
                {i < 3 && <div className="hidden md:block absolute top-6 left-[calc(100%-8px)] w-full h-px bg-gradient-to-r from-emerald-400/20 to-transparent" />}
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ============ CTA ============ */}
      <section className="bg-[#fbfaf7] py-28">
        <div className="max-w-[1320px] mx-auto px-6 lg:px-10">
          <motion.div
            variants={fadeUp} initial="hidden" whileInView="show" viewport={{ once: true }}
            className="relative rounded-[32px] overflow-hidden grad-emerald px-10 py-16 md:px-20 md:py-20 text-center"
          >
            <div className="absolute inset-0 opacity-50" style={{ background: 'radial-gradient(circle at 20% 20%, rgba(94,234,212,0.25), transparent 60%)' }} />
            <div className="relative">
              <h2 className="font-display text-[32px] md:text-[44px] font-medium text-white leading-tight max-w-[700px] mx-auto">
                Ready to see a radiograph analyzed in real time?
              </h2>
              <p className="mt-5 text-white/65 text-[16px] max-w-[520px] mx-auto">
                No account required. Upload a sample image and watch both models work.
              </p>
              <div className="mt-9 flex flex-wrap items-center justify-center gap-4">
                <Link to="/cnn" className="btn-primary !bg-white !text-[#0c1f1a] !shadow-none hover:!shadow-xl">
                  Try WristNet CNN <ArrowRight size={18} />
                </Link>
                <Link to="/yolo" className="btn-ghost-dark">
                  Try YOLO26 <ArrowRight size={16} />
                </Link>
              </div>
            </div>
          </motion.div>
        </div>
      </section>
    </PageTransition>
  );
}

function WristGlyph() {
  return (
    <svg viewBox="0 0 400 400" fill="none" className="w-full h-full opacity-60">
      <defs>
        <linearGradient id="wg1" x1="0" y1="0" x2="400" y2="400">
          <stop offset="0%" stopColor="#5eead4" stopOpacity="0.5" />
          <stop offset="100%" stopColor="#d4a94a" stopOpacity="0.2" />
        </linearGradient>
      </defs>
      <g stroke="url(#wg1)" strokeWidth="1.4" fill="none" strokeLinecap="round">
        <path d="M150 60 L155 180 Q160 210 175 220 L175 340" />
        <path d="M180 55 L183 175 Q186 205 195 218 L195 340" />
        <path d="M210 58 L212 178 Q214 208 205 220 L205 340" />
        <path d="M240 62 L235 180 Q230 210 220 222 L215 340" />
        <path d="M130 190 Q200 175 270 195" />
        <path d="M125 220 Q200 250 275 225" />
        <circle cx="200" cy="260" r="55" opacity="0.5" />
        <circle cx="200" cy="260" r="80" opacity="0.25" />
      </g>
    </svg>
  );
}
