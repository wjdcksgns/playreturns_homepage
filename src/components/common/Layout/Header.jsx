import { Link } from 'react-router-dom';
import { useSelector } from 'react-redux';

import styles from './Header.module.css';
import imgLogo from '../../../assets/images/logo_wh (2).png';

import Container from './Container';
import Gnb from './Gnb';
import Users from './Users';
import MobileNav from './MobileNav';
import { useEffect, useRef } from 'react';

const Header = () => {
  const { windowWidth } = useSelector((state) => state.viewport);
  const headerRef = useRef(null);

  const handleScroll = () => {
    const scrollPosition = window.scrollY || document.documentElement.scrollTop;
    if (!headerRef.current) return;

    if (scrollPosition === 0) {
      // 최상단: 완전 흰색
      headerRef.current.style.background = '#ffffff';
      headerRef.current.style.color = '#000000';
    } else {
      // 스크롤 내림: 약간 투명 회색
      headerRef.current.style.background = 'rgba(255, 255, 255, 0.9)';
      headerRef.current.style.color = '#000000';
    }
  };

  useEffect(() => {
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <header ref={headerRef} id="header" className={styles.header}>
      <div className="wrap">
        <Container isWide={true}>
          <div className={styles.contents}>
            <h1 className={styles.logo}>
              <Link to="/" className={styles.logoBox}>
                <img src={imgLogo} alt="logo" />
                <span className={styles.launchText}>리맵 RE:MAP 정식 배포!</span>
              </Link>
            </h1>

            {/* 데스크톱 메뉴 */}
            {windowWidth > 1024 ? (
              <nav className={styles.nav}>
                <Link to="/about">회사 소개</Link>
                <Link to="/history">회사 연혁</Link>
                <Link to="/portfolio">포트폴리오</Link>
                <Link to="/technology">보유 기술</Link>
                <Link to="/contact">Contact</Link>
                <Link to="/admin/login" className={styles.adminBtn}>
                  서울대학교 멘토-멘티 매칭
                </Link>
                <Users />
              </nav>
            ) : (
              /* 모바일 메뉴 */
              <MobileNav>
                <Users />
                <nav className={styles.nav}>
                  <Link to="/about">회사 소개</Link>
                  <Link to="/history">회사 연혁</Link>
                  <Link to="/portfolio">포트폴리오</Link>
                  <Link to="/technology">보유 기술</Link>
                  <Link to="/contact">Contact</Link>
                  <Link to="/admin/login" className={styles.adminBtn}>
                    서울대학교 멘토-멘티 매칭
                  </Link>
                </nav>
              </MobileNav>
            )}
          </div>
        </Container>
      </div>
    </header>
  )
}

export default Header;