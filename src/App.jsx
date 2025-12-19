import { useCallback, useEffect } from 'react';
import { Routes, Route } from 'react-router-dom';
import { useDispatch, useSelector } from 'react-redux';
import { setViewport } from './features/common/viewportSlice';

import './common/css/App.css';
import { setVh } from './common/js/ui';
import { QueryClient, QueryClientProvider } from "react-query";

import Header from './components/common/Layout/Header';
import Footer from './components/common/Layout/Footer';
import ScrollToTop from './components/common/ScrollToTop';

import Main from './pages/Main/Main';
import About from './pages/About';
import Portfolio from './pages/Portfolio';
import Contact from './pages/Contact';
import History from './pages/History';
import Technology from './pages/Technology';
import Privacy from './pages/Privacy';
import NotFound from './pages/NotFound/NotFound';
import FloatingContactButton from "./components/FloatingContactButton"; // 경로 맞게 조정
import AdminLogin from './components/pages/admin/AdminLogin';
import AdminUpload from './components/pages/admin/AdminUpload';


const queryClient = new QueryClient();

const App = () => {
  const dispatch = useDispatch();
  const { windowHeight } = useSelector((state) => state.viewport);

  const handleResize = useCallback(() => {
    dispatch(setViewport({
      windowWidth: window.innerWidth,
      windowHeight: window.innerHeight
    }));
  }, [dispatch]);

  setVh(windowHeight);

  useEffect(() => {
    window.addEventListener('resize', handleResize);
    return () => window.addEventListener('resize', handleResize);
  }, [handleResize]);

  return (
    <div id="app">
      <QueryClientProvider client={queryClient}>
        <Header />
        <ScrollToTop />
        <main>
          <Routes>
            <Route index element={<Main />} />
            <Route path="/about" element={<About />} />
            <Route path="/history" element={<History />} />
            <Route path="/portfolio" element={<Portfolio />} />
            <Route path="/contact" element={<Contact />} />
            <Route path="/technology" element={<Technology />} />
            <Route path="/privacy" element={<Privacy />} />
            <Route path="/admin/login" element={<AdminLogin />} />
            <Route path="/admin/upload" element={<AdminUpload />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </main>
        <Footer />
        {/* ✅ 모든 페이지에서 표시 */}
        <FloatingContactButton />
      </QueryClientProvider>
    </div>
  );
}

export default App;
