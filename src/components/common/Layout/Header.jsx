import { useEffect, useState } from 'react';
import { Link, NavLink, useLocation } from 'react-router-dom';
import { useSelector } from 'react-redux';

import styles from './Header.module.css';
import logoPlayReturns from '../../../assets/images/logo_wh (2).png';

import Container from './Container';
import MobileNav from './MobileNav';

const navLinks = [
  { to: '/about', label: '회사 소개' },
  { to: '/history', label: '회사 연혁' },
  { to: '/portfolio', label: '포트폴리오' },
  { to: '/technology', label: '보유 기술' },
  { to: '/contact', label: 'Contact' },
];

const Header = () => {
  const { windowWidth } = useSelector((state) => state.viewport);
  const location = useLocation();
  const [scrolled, setScrolled] = useState(false);

  const isHome = location.pathname === '/';
  // 모바일/태블릿(1024px 이하)에선 transparent 모드 비활성화 — 항상 흰 배경 헤더로
  // CSS specificity 싸움 대신 JS로 클래스 자체를 분기해서 100% 확실하게 적용
  const isMobile = windowWidth <= 1024;
  const isTransparent = isHome && !scrolled && !isMobile;

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 16);
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  const headerClass = `${styles.header} ${
    isTransparent ? styles.transparent : styles.solid
  }`;

  // 모바일에선 인라인 스타일로 흰 배경을 강제 (CSS specificity / 캐시 이슈 우회)
  const headerInlineStyle = isMobile
    ? {
        background: '#ffffff',
        backgroundColor: '#ffffff',
        backdropFilter: 'none',
        WebkitBackdropFilter: 'none',
        borderBottom: '1px solid rgba(15, 23, 42, 0.08)',
        boxShadow: '0 2px 8px rgba(15, 23, 42, 0.06)',
        color: '#0f172a',
      }
    : undefined;

  const renderNav = (closeOnClick = false) => (
    <nav className={styles.nav}>
      {navLinks.map((link) => (
        <NavLink
          key={link.to}
          to={link.to}
          className={({ isActive }) =>
            `${styles.navLink} ${isActive ? styles.navLinkActive : ''}`
          }
        >
          {link.label}
        </NavLink>
      ))}
      <Link to="/admin/login" className={styles.adminBtn}>
        <span className={styles.adminDot} />
        서울대학교 멘토-멘티 매칭
      </Link>
    </nav>
  );

  return (
    <header id="header" className={headerClass} style={headerInlineStyle}>
      <Container isWide={true}>
        <div className={styles.contents}>
          <h1 className={styles.logo}>
            <Link to="/" className={styles.logoBox}>
              <img
                src={logoPlayReturns}
                alt="PlayReturns"
                className={styles.logoImg}
              />
              <span className={styles.launchBadge}>
                <span className={styles.launchDot} />
                NEW · 3개 앱 출시
              </span>
            </Link>
          </h1>

          {windowWidth > 1024 ? (
            renderNav()
          ) : (
            <MobileNav>{renderNav(true)}</MobileNav>
          )}
        </div>
      </Container>
    </header>
  );
};

export default Header;
