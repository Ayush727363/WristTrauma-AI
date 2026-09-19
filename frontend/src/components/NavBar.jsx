import { useEffect, useState } from 'react';
import { Link, NavLink } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Activity, Menu, X } from 'lucide-react';
import { HOSPITAL } from '../lib/config';

const NAV_ITEMS = [
  { to: '/', label: 'Home' },
  { to: '/cnn', label: 'WristNet CNN' },
  { to: '/yolo', label: 'YOLO26' },
  { to: '/about', label: 'About' },
];

export default function NavBar() {
  const [scrolled, setScrolled] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 12);
    window.addEventListener('scroll', onScroll);
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  return (
    <header
      className="fixed top-0 left-0 right-0 z-50 transition-all duration-500"
      style={{
        background: scrolled ? 'rgba(4,18,15,0.78)' : 'transparent',
        backdropFilter: scrolled ? 'blur(18px) saturate(160%)' : 'none',
        borderBottom: scrolled ? '1px solid rgba(94,234,212,0.10)' : '1px solid transparent',
      }}
    >
      <div className="max-w-[1320px] mx-auto px-6 lg:px-10 flex items-center justify-between h-[76px]">
        <Link to="/" className="flex items-center gap-3 group">
          <div className="relative w-10 h-10 flex items-center justify-center">
            <div className="absolute inset-0 rounded-full bg-gradient-to-br from-emerald-400 to-emerald-600 opacity-90 group-hover:opacity-100 transition-opacity" />
            <Activity className="relative w-5 h-5 text-white" strokeWidth={2.4} />
          </div>
          <div className="flex flex-col leading-tight">
            <span className="font-display text-[17px] font-semibold text-white tracking-tight">
              {HOSPITAL.shortName}
            </span>
            <span className="text-[10px] uppercase tracking-[0.14em] text-emerald-300/70 font-medium">
              Diagnostics
            </span>
          </div>
        </Link>

        <nav className="hidden md:flex items-center gap-1">
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === '/'}
              className={({ isActive }) =>
                `relative px-4 py-2 text-[14px] font-medium rounded-full transition-colors duration-300 ${
                  isActive ? 'text-white' : 'text-white/55 hover:text-white/90'
                }`
              }
            >
              {({ isActive }) => (
                <>
                  {isActive && (
                    <motion.span
                      layoutId="nav-pill"
                      className="absolute inset-0 rounded-full bg-white/10 border border-emerald-300/20"
                      transition={{ type: 'spring', stiffness: 380, damping: 32 }}
                    />
                  )}
                  <span className="relative">{item.label}</span>
                </>
              )}
            </NavLink>
          ))}
        </nav>

        <div className="hidden md:flex items-center gap-3">
          <Link to="/cnn" className="btn-ghost-dark !py-2.5 !px-5 !text-[13.5px]">
            Launch Screening
          </Link>
        </div>

        <button
          className="md:hidden text-white p-2"
          onClick={() => setMobileOpen((v) => !v)}
          aria-label="Toggle menu"
        >
          {mobileOpen ? <X size={22} /> : <Menu size={22} />}
        </button>
      </div>

      {mobileOpen && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          className="md:hidden bg-[#04120f]/98 backdrop-blur-xl border-t border-emerald-300/10 px-6 py-4 flex flex-col gap-1"
        >
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === '/'}
              onClick={() => setMobileOpen(false)}
              className={({ isActive }) =>
                `px-4 py-3 rounded-xl text-[15px] font-medium ${
                  isActive ? 'bg-white/10 text-white' : 'text-white/60'
                }`
              }
            >
              {item.label}
            </NavLink>
          ))}
        </motion.div>
      )}
    </header>
  );
}
