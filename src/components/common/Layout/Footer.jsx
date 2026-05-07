import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { FaMapMarkerAlt, FaPhoneAlt, FaEnvelope } from 'react-icons/fa';

import styles from './Footer.module.css';
import imgLogo from '../../../assets/images/logo_wh (2).png';

const navLinks = [
  { to: '/about', label: '회사 소개' },
  { to: '/history', label: '회사 연혁' },
  { to: '/portfolio', label: '포트폴리오' },
  { to: '/technology', label: '보유 기술' },
  { to: '/contact', label: 'Contact' },
];

const legalLinks = [
  { to: '/remapprivacy', label: '리맵 개인정보처리방침' },
  { to: '/terms', label: '리맵 이용약관' },
  { to: '/sajuprivacy', label: '사주명 개인정보처리방침' },
  { to: '/sudoku-privacy', label: '스도쿠 개인정보처리방침' },
];

const Footer = () => {
  const [thisYear, setThisYear] = useState(0);

  useEffect(() => {
    setThisYear(new Date().getFullYear());
  }, []);

  return (
    <footer id="footer" className={styles.footer}>
      <div className="wrap">
        <div className="container">
          <div className={styles.top}>
            <div className={styles.brand}>
              <img src={imgLogo} alt="PlayReturns" className={styles.logo} />
              <p className={styles.tagline}>See something, know something.</p>
              <ul className={styles.contactList}>
                <li>
                  <FaMapMarkerAlt className={styles.icon} />
                  <span>경기도 수원시 영통구 광교산로 154-42, 경기대학교 창업보육센터 408호</span>
                </li>
                <li>
                  <FaPhoneAlt className={styles.icon} />
                  <a href="tel:01028680655">010-2868-0655</a>
                </li>
                <li>
                  <FaEnvelope className={styles.icon} />
                  <a href="mailto:playreturns2025@gmail.com">playreturns2025@gmail.com</a>
                </li>
              </ul>
            </div>

            <div className={styles.linksGroup}>
              <div className={styles.linkColumn}>
                <h4>회사</h4>
                <ul>
                  {navLinks.map((link) => (
                    <li key={link.to}>
                      <Link to={link.to}>{link.label}</Link>
                    </li>
                  ))}
                </ul>
              </div>

              <div className={styles.linkColumn}>
                <h4>약관 / 정책</h4>
                <ul>
                  {legalLinks.map((link) => (
                    <li key={link.to}>
                      <Link to={link.to}>{link.label}</Link>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>

          <div className={styles.bottom}>
            <p className={styles.company}>
              <span>플레이리턴즈</span>
              <span className={styles.divider} />
              <span>대표이사 정유진</span>
            </p>
            <p className={styles.copyright}>
              © {thisYear} PlayReturns. All rights reserved.
            </p>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
