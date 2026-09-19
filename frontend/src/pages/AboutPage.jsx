import { motion } from 'framer-motion';
import { Database, GitBranch, Layers, Target, Users, BookOpen, Scale } from 'lucide-react';
import PageTransition from '../components/PageTransition';
import { HOSPITAL, DOCTORS } from '../lib/config';

const fadeUp = {
  hidden: { opacity: 0, y: 24 },
  show: (i = 0) => ({ opacity: 1, y: 0, transition: { duration: 0.6, delay: i * 0.07, ease: [0.16, 1, 0.3, 1] } }),
};

export default function AboutPage() {
  return (
    <PageTransition>
      <section className="relative grad-hero-mesh pt-36 pb-24">
        <div className="max-w-[1000px] mx-auto px-6 lg:px-10">
          <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}>
            <div className="chip bg-emerald-400/10 text-emerald-300 border border-emerald-300/25 mb-6">
              <BookOpen size={13} /> Methodology &amp; Provenance
            </div>
            <h1 className="font-display text-[38px] sm:text-[52px] font-medium text-white leading-[1.08] max-w-[720px]">
              Built for rigor, not just demonstration.
            </h1>
            <p className="mt-5 text-white/55 text-[17px] max-w-[640px] leading-relaxed">
              Every design decision in this project &mdash; from the patient-level data split to the
              dual-model architecture &mdash; was made to produce results that would hold up to
              scrutiny, not just look good in a demo.
            </p>
          </motion.div>
        </div>
      </section>

      <section className="bg-[#fbfaf7] py-24">
        <div className="max-w-[1000px] mx-auto px-6 lg:px-10 grid md:grid-cols-2 gap-8">
          {[
            {
              icon: Database, title: 'GRAZPEDWRI-DX Dataset',
              body: '20,327 pediatric wrist radiographs from 6,091 patients, collected at University Hospital Graz (2008\u20132018), with expert-annotated bounding boxes across 9 finding categories.',
            },
            {
              icon: GitBranch, title: 'Patient-Level Splitting',
              body: 'Train/validation/test splits are enforced at the patient level, not the image level \u2014 no patient\u2019s radiographs appear in more than one split, preventing leakage across multiple views of the same injury.',
            },
            {
              icon: Layers, title: 'Class Imbalance Handling',
              body: 'The 9 raw finding categories were consolidated into 4 clinically coherent classes for WristNet, and focal loss was used during training to prevent common findings from drowning out rare ones.',
            },
            {
              icon: Target, title: 'Dual-Model Cross-Validation',
              body: 'WristNet (a custom CNN with CAM explainability) and YOLO26s (a modern object detector) are trained and evaluated independently, giving two architecturally distinct perspectives on every study.',
            },
          ].map((item, i) => (
            <motion.div
              key={item.title}
              variants={fadeUp} custom={i} initial="hidden" whileInView="show" viewport={{ once: true }}
              className="rounded-2xl border border-black/6 bg-white p-7 shadow-sm"
            >
              <div className="w-11 h-11 rounded-xl bg-emerald-500/8 border border-emerald-600/15 flex items-center justify-center mb-5">
                <item.icon size={20} className="text-emerald-700" strokeWidth={1.7} />
              </div>
              <h3 className="font-display text-[19px] font-semibold text-[#0c1f1a] mb-2.5">{item.title}</h3>
              <p className="text-[#3c5952] text-[14px] leading-relaxed">{item.body}</p>
            </motion.div>
          ))}
        </div>
      </section>

      <section className="bg-[#0a231e] py-24">
        <div className="max-w-[1000px] mx-auto px-6 lg:px-10">
          <motion.div variants={fadeUp} initial="hidden" whileInView="show" viewport={{ once: true }} className="text-center mb-16">
            <span className="chip bg-white/8 text-emerald-300 border border-emerald-300/20 mb-5">
              <Users size={13} /> Research Team
            </span>
            <h2 className="font-display text-[32px] md:text-[40px] font-medium text-white">
              The people behind the project
            </h2>
          </motion.div>

          <div className="grid sm:grid-cols-2 gap-6 max-w-[680px] mx-auto">
            {DOCTORS.map((doc, i) => (
              <motion.div
                key={doc.name}
                variants={fadeUp} custom={i} initial="hidden" whileInView="show" viewport={{ once: true }}
                className="rounded-2xl border border-white/8 bg-white/[0.03] p-7 text-center"
              >
                <div className="w-16 h-16 rounded-full mx-auto mb-4 flex items-center justify-center font-display text-2xl font-semibold text-white"
                     style={{ background: 'linear-gradient(135deg, #16a37d, #0f7a5c)' }}>
                  {doc.name.split(' ').slice(-1)[0][0]}
                </div>
                <div className="text-white font-semibold text-[16px]">{doc.name}</div>
                <div className="text-emerald-300/70 text-[13px] mt-1">{doc.role}</div>
                <div className="text-white/30 text-[11px] mt-2 font-mono">{doc.reg}</div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      <section className="bg-[#fbfaf7] py-24">
        <div className="max-w-[800px] mx-auto px-6 lg:px-10 text-center">
          <motion.div variants={fadeUp} initial="hidden" whileInView="show" viewport={{ once: true }}>
            <Scale size={26} className="mx-auto text-emerald-700 mb-5" strokeWidth={1.6} />
            <h2 className="font-display text-[26px] font-medium text-[#0c1f1a] mb-4">A note on scope</h2>
            <p className="text-[#3c5952] text-[15px] leading-relaxed">
              {HOSPITAL.name} and this platform are an academic research deployment built to demonstrate
              weakly-supervised localization and multi-model comparison methodology on a public pediatric
              radiograph dataset. Nothing on this platform is a certified diagnostic device, and no output
              should inform a real clinical decision without independent radiologist review.
            </p>
          </motion.div>
        </div>
      </section>
    </PageTransition>
  );
}
