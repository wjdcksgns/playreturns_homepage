import { useState } from 'react';
import { Link } from 'react-router-dom';
import ContactModal from '../../common/Modals/ContactModal';
import styles from './MainContents.module.css';

const apps = [
  { name: '리맵', en: 'RE:MAP' },
  { name: '사주명', en: 'SAJUMYEONG' },
  { name: '스도쿠', en: 'SUDOKU' },
  { name: '펫상', en: 'PETSANG' },
  { name: '파리 살려!', en: 'SAVE THE FLY' },
];

const MainContents = () => {
  const [showContact, setShowContact] = useState(false);

  const onScroll = () => {
    window.scrollTo({ top: window.innerHeight, behavior: 'smooth' });
  };

  return (
    <div className={styles.contents}>
      <div className={styles.eyebrow}>
        <span className={styles.dot} />
        <span>WELCOME TO PLAYRETURNS</span>
      </div>

      <h1 className={styles.brandTitle}>
        <span className={styles.brandKr}>플레이리턴즈</span>
        <span className={styles.brandEn}>PlayReturns</span>
      </h1>

      <p className={styles.tagline}>“See something, know something.”</p>

      <p className={styles.subtitle}>
        AR/VR · AI · 디지털트윈 융합 기술로
        <br />
        모바일·웹·콘텐츠를 만드는 기업입니다.
      </p>

      <div className={styles.appPills}>
        {apps.map((app) => (
          <span key={app.name} className={styles.pill}>
            <strong>{app.name}</strong>
            <small>{app.en}</small>
          </span>
        ))}
      </div>

      <div className={styles.actions}>
        <button
          type="button"
          className={styles.primaryBtn}
          onClick={onScroll}
        >
          앱 둘러보기 ↓
        </button>
        <Link to="/portfolio" className={styles.secondaryBtn}>
          포트폴리오 보기
        </Link>
        <button
          type="button"
          className={styles.tertiaryBtn}
          onClick={() => setShowContact(true)}
        >
          문의하기
        </button>
      </div>

      {showContact && (
        <ContactModal onClose={() => setShowContact(false)} />
      )}

      <button
        type="button"
        className={styles.scrollCue}
        onClick={onScroll}
        aria-label="아래로 스크롤"
      >
        <span className={styles.scrollLabel}>SCROLL</span>
        <span className={styles.scrollLine} />
      </button>
    </div>
  );
};

export default MainContents;
