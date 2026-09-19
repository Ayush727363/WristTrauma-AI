import { Routes, Route, useLocation } from 'react-router-dom';
import { AnimatePresence } from 'framer-motion';
import NavBar from './components/NavBar';
import Footer from './components/Footer';
import Home from './pages/Home';
import CnnPage from './pages/CnnPage';
import YoloPage from './pages/YoloPage';
import AboutPage from './pages/AboutPage';

export default function App() {
  const location = useLocation();

  return (
    <div className="min-h-screen flex flex-col">
      <div className="grain-overlay" />
      <NavBar />
      <main className="flex-1">
        <AnimatePresence mode="wait">
          <Routes location={location} key={location.pathname}>
            <Route path="/" element={<Home />} />
            <Route path="/cnn" element={<CnnPage />} />
            <Route path="/yolo" element={<YoloPage />} />
            <Route path="/about" element={<AboutPage />} />
          </Routes>
        </AnimatePresence>
      </main>
      <Footer />
    </div>
  );
}
