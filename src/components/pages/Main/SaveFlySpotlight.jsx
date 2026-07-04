import styles from './SaveFlySpotlight.module.css';

const PUBLIC = process.env.PUBLIC_URL || '';

const PlayBadge = () => (
  <svg className={styles.playIcon} viewBox="0 0 24 24" aria-hidden="true">
    <path
      fill="currentColor"
      d="M3 2.5v19l8.5-9.5L3 2.5zm10.2 8.4l3.4-1.9L4.6 2.6l8.6 8.3zm0 2.2l-8.6 8.3L16.6 15l-3.4-1.9zm9-2.6l-4.4-2.5-3.7 4.1 3.7 4.1 4.4-2.5c1-.6 1-2 0-2.7v-.5z"
    />
  </svg>
);

const STORE_URL =
  'https://play.google.com/store/apps/details?id=com.savefly.save_the_fly';

const shots = [
  { src: `${PUBLIC}/images/savefly/04_ranking.webp`, cls: 'phoneLeft' },
  { src: `${PUBLIC}/images/savefly/01_main.webp`, cls: 'phoneCenter' },
  { src: `${PUBLIC}/images/savefly/05_hall_of_fame.webp`, cls: 'phoneRight' },
];

const features = ['방치형 키우기', '스킨 · 칭호 수집', '온라인 랭킹', '명예의 전당'];

const SaveFlySpotlight = () => {
  return (
    <section className={styles.section}>
      <div className={styles.bg}>
        <div className={styles.blob1} />
        <div className={styles.blob2} />
      </div>

      <div className={styles.inner}>
        <div className={`${styles.copy} reveal`}>
          <span className={styles.badge}>NEW RELEASE · 신작</span>
          <h2 className={styles.title}>파리 살려!</h2>
          <p className={styles.tagline}>파리 키우기 방치형 캐주얼 게임</p>
          <p className={styles.desc}>
            작은 파리 한 마리를 먹이고 돌보며 키우는 방치형 게임. 다양한 스킨과
            칭호를 모으고, 온라인 랭킹과 명예의 전당에서 다른 이용자와
            경쟁해보세요.
          </p>

          <ul className={styles.chips}>
            {features.map((f) => (
              <li key={f} className={styles.chip}>
                {f}
              </li>
            ))}
          </ul>

          <a
            href={STORE_URL}
            target="_blank"
            rel="noopener noreferrer"
            className={styles.storeBtn}
          >
            <PlayBadge />
            <span>
              <small>지금 다운로드</small>
              <strong>Google Play</strong>
            </span>
          </a>
        </div>

        <div className={`${styles.visual} reveal`}>
          <div className={styles.stage}>
            {shots.map((s) => (
              <div key={s.cls} className={`${styles.phone} ${styles[s.cls]}`}>
                <img src={s.src} alt="파리 살려! 게임 화면" loading="lazy" />
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};

export default SaveFlySpotlight;
