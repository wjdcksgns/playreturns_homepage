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
import RemapPrivacyEn from './pages/RemapPrivacyEn';
import RemapTermsEn from './pages/RemapTermsEn';
import SajuYeonPrivacy from './pages/SajuYeonPrivacy';
import SudokuPrivacy from './pages/SudokuPrivacy';
import SudokuDeleteAccount from './pages/SudokuDeleteAccount';
import PetsangPrivacy from './pages/PetsangPrivacy';
import PetsangTerms from './pages/PetsangTerms';
import PetsangDeleteAccount from './pages/PetsangDeleteAccount';
import SaveFlyPrivacy from './pages/SaveFlyPrivacy';
import SaveFlyTerms from './pages/SaveFlyTerms';
import SaveFlyDeleteAccount from './pages/SaveFlyDeleteAccount';
import Terms from './pages/Terms';
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
    return () => window.removeEventListener('resize', handleResize);
  }, [handleResize]);

  // ===== 전역 스크롤 페이드인 옵저버 =====
  // .reveal 클래스가 붙은 모든 요소를 자동으로 감지해 뷰포트 진입 시 .is-visible 추가
  useEffect(() => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      document.querySelectorAll('.reveal').forEach((el) =>
        el.classList.add('is-visible')
      );
      return undefined;
    }

    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add('is-visible');
            io.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.12, rootMargin: '0px 0px -60px 0px' }
    );

    const scan = () => {
      document
        .querySelectorAll('.reveal:not(.is-visible)')
        .forEach((el) => io.observe(el));
    };

    scan();

    // 라우트 변경 등으로 새로 추가된 .reveal 요소도 감지
    const mo = new MutationObserver(scan);
    mo.observe(document.body, { childList: true, subtree: true });

    return () => {
      io.disconnect();
      mo.disconnect();
    };
  }, []);

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
            <Route path="/remapprivacy" element={<Privacy />} />
            <Route path="/remapprivacy-en" element={<RemapPrivacyEn />} />
            <Route path="/terms-en" element={<RemapTermsEn />} />
            <Route path="/sajuprivacy" element={<SajuYeonPrivacy />} />
            <Route path="/sudoku-privacy" element={<SudokuPrivacy />} />
            <Route path="/sudoku-delete-account" element={<SudokuDeleteAccount />} />
            <Route path="/petsang-privacy" element={<PetsangPrivacy />} />
            <Route path="/petsang-terms" element={<PetsangTerms />} />
            <Route path="/petsang-delete-account" element={<PetsangDeleteAccount />} />
            <Route path="/savefly-privacy" element={<SaveFlyPrivacy />} />
            <Route path="/savefly-terms" element={<SaveFlyTerms />} />
            <Route path="/savefly-delete-account" element={<SaveFlyDeleteAccount />} />
            <Route path="/terms" element={<Terms />} />
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
