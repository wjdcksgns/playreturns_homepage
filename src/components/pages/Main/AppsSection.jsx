import { useState } from 'react';
import { Link } from 'react-router-dom';
import styles from './AppsSection.module.css';

const apps = [
  {
    id: 'remap',
    name: '리맵',
    nameEn: 'RE:MAP',
    tagline: '당신의 순간을 지도에 담다',
    description:
      '지도 위에 사진과 추억을 핀으로 기록하는 여행 다이어리. GPS 기반 자동 기록부터 AI 여행 일기 생성, 추억 슬라이드쇼, PDF 여행책까지 한 번에 만들어주는 올인원 여행 기록 앱입니다.',
    features: [
      '지도 위 위치 기반 핀맵 기록',
      '사진·영상·텍스트 업로드 및 저장',
      'AI 여행 일기 / 여행지 추천 (구독)',
      '추억 슬라이드쇼 영상 · PDF 여행책 생성',
      '공개 기록으로 다른 여행자와 공유',
    ],
    color: '#22c55e',
    bgGradient: 'linear-gradient(135deg, #34d399 0%, #10b981 100%)',
    cardBg: '#ecfdf5',
    storeUrl:
      'https://play.google.com/store/apps/details?id=kr.co.playreturns.remap',
    privacyPath: '/privacy',
    termsPath: '/terms',
    heroImage: '/images/remap/real/KakaoTalk_20260428_113408623.jpg',
    screenshots: [
      '/images/remap/real/KakaoTalk_20260428_113408623.jpg',
      '/images/remap/real/KakaoTalk_20260428_113408623_01.jpg',
      '/images/remap/real/KakaoTalk_20260428_113408623_02.jpg',
      '/images/remap/real/KakaoTalk_20260428_113408623_03.jpg',
      '/images/remap/real/KakaoTalk_20260428_113408623_04.jpg',
      '/images/remap/real/KakaoTalk_20260428_113408623_05.jpg',
      '/images/remap/real/KakaoTalk_20260428_113408623_06.jpg',
      '/images/remap/real/KakaoTalk_20260428_113408623_07.jpg',
      '/images/remap/real/KakaoTalk_20260428_113408623_08.jpg',
      '/images/remap/real/KakaoTalk_20260428_113408623_09.jpg',
      '/images/remap/real/KakaoTalk_20260428_113408623_10.jpg',
    ],
  },
  {
    id: 'saju',
    name: '사주명',
    nameEn: 'SAJUMYEONG',
    tagline: 'AI 사주풀이 · 궁합 분석',
    description:
      '정통 사주명리에 AI의 분석력을 더한 맞춤형 운세 서비스. 이름·생년월일·태어난 시각만 입력하면 AI가 당신의 사주와 궁합을 풀어드립니다.',
    features: [
      '연애·결혼·재물 등 분야별 사주풀이',
      '두 사람의 궁합 분석',
      '관심 월·궁금한 운에 맞춘 맞춤 해석',
      '최근 10건 히스토리 자동 저장',
      'Google · 카카오 간편 로그인',
    ],
    color: '#d4af37',
    bgGradient: 'linear-gradient(135deg, #1a1a2e 0%, #2d2640 100%)',
    cardBg: '#1a1a2e',
    cardText: '#f5f0d6',
    storeUrl:
      'https://play.google.com/store/apps/details?id=com.playreturns.sajuyeon',
    privacyPath: '/saju-privacy',
    heroImage: '/images/saju/real/스크린샷_1.jpg',
    screenshots: [
      '/images/saju/real/스크린샷_1.jpg',
      '/images/saju/real/스크린샷_2.jpg',
      '/images/saju/real/스크린샷_3.jpg',
      '/images/saju/real/스크린샷_4.jpg',
    ],
  },
  {
    id: 'sudoku',
    name: '스도쿠',
    nameEn: 'SUDOKU',
    tagline: '숫자 퍼즐 두뇌 게임',
    description:
      '클래식 스도쿠를 깔끔한 디자인으로. 다양한 난이도와 일일 챌린지로 매일 새로운 두뇌 운동을 즐기고, Google 계정으로 기기 간 진행 상황을 동기화할 수 있습니다.',
    features: [
      '쉬움부터 매우 어려움까지 4단계 난이도',
      '일일 챌린지 모드',
      '힌트 / 메모 / 자동 검사 기능',
      'Google 계정으로 기기 간 동기화',
      '깔끔한 다크모드 지원',
    ],
    color: '#3b82f6',
    bgGradient: 'linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%)',
    cardBg: '#1e3a8a',
    cardText: '#e0e7ff',
    storeUrl:
      'https://play.google.com/store/apps/details?id=com.sudokuapp.sudoku_game',
    privacyPath: '/sudoku-privacy',
    deletePath: '/sudoku-delete-account',
    heroImage: '/images/sudoku/real/KakaoTalk_20260428_113852741.jpg',
    screenshots: [
      '/images/sudoku/real/KakaoTalk_20260428_113852741.jpg',
      '/images/sudoku/real/KakaoTalk_20260428_113852741_01.jpg',
      '/images/sudoku/real/KakaoTalk_20260428_113852741_02.jpg',
      '/images/sudoku/real/KakaoTalk_20260428_113852741_03.jpg',
      '/images/sudoku/real/KakaoTalk_20260428_113852741_04.jpg',
      '/images/sudoku/real/KakaoTalk_20260428_113852741_05.jpg',
    ],
  },
];

const PlayBadge = () => (
  <svg
    className={styles.playBadgeIcon}
    viewBox="0 0 24 24"
    aria-hidden="true"
  >
    <path
      fill="currentColor"
      d="M3 2.5v19l8.5-9.5L3 2.5zm10.2 8.4l3.4-1.9L4.6 2.6l8.6 8.3zm0 2.2l-8.6 8.3L16.6 15l-3.4-1.9zm9-2.6l-4.4-2.5-3.7 4.1 3.7 4.1 4.4-2.5c1-.6 1-2 0-2.7v-.5z"
    />
  </svg>
);

const AppsSection = () => {
  const [selected, setSelected] = useState(null);
  const [slideIdx, setSlideIdx] = useState(0);

  const open = (app) => {
    setSelected(app);
    setSlideIdx(0);
  };
  const close = () => setSelected(null);

  const prev = () =>
    setSlideIdx((i) =>
      i === 0 ? selected.screenshots.length - 1 : i - 1
    );
  const next = () =>
    setSlideIdx((i) =>
      i === selected.screenshots.length - 1 ? 0 : i + 1
    );

  return (
    <section className={styles.section}>
      <div className={styles.heading}>
        <p className={styles.eyebrow}>OUR APPS</p>
        <h2>플레이리턴즈가 만든 앱</h2>
        <p className={styles.subtitle}>
          지금 Google Play 스토어에서 만나보실 수 있습니다.
        </p>
      </div>

      <div className={styles.grid}>
        {apps.map((app) => {
          const isDark = !!app.cardText;
          return (
            <article
              key={app.id}
              className={`${styles.card} ${isDark ? styles.cardDark : ''} reveal`}
              style={{
                background: app.cardBg,
                color: app.cardText || '#1f2937',
              }}
            >
              <div
                className={styles.preview}
                style={{ background: app.bgGradient }}
              >
                <img
                  src={process.env.PUBLIC_URL + app.heroImage}
                  alt={`${app.name} 스크린샷`}
                  className={styles.phoneShot}
                  loading="lazy"
                />
              </div>

              <div className={styles.body}>
                <div className={styles.titleRow}>
                  <h3>
                    {app.name}
                    <span className={styles.nameEn}>{app.nameEn}</span>
                  </h3>
                </div>
                <p
                  className={styles.tagline}
                  style={{ color: app.color }}
                >
                  {app.tagline}
                </p>
                <p className={styles.desc}>{app.description}</p>

                <div className={styles.actions}>
                  <button
                    type="button"
                    className={styles.detailBtn}
                    onClick={() => open(app)}
                    style={{
                      borderColor: app.color,
                      color: isDark ? app.color : app.color,
                    }}
                  >
                    자세히 보기
                  </button>
                  <a
                    href={app.storeUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className={styles.storeBtn}
                    style={{ background: app.color }}
                  >
                    <PlayBadge />
                    <span>
                      <small>GET IT ON</small>
                      <strong>Google Play</strong>
                    </span>
                  </a>
                </div>
              </div>
            </article>
          );
        })}
      </div>

      {selected && (
        <div className={styles.modalOverlay} onClick={close}>
          <div
            className={styles.modal}
            onClick={(e) => e.stopPropagation()}
          >
            <button
              type="button"
              className={styles.modalClose}
              onClick={close}
              aria-label="닫기"
            >
              ×
            </button>

            <div className={styles.modalHeader}>
              <h3>
                {selected.name}
                <span className={styles.nameEn}>{selected.nameEn}</span>
              </h3>
              <p style={{ color: selected.color }}>{selected.tagline}</p>
            </div>

            <div className={styles.modalSlider}>
              <button
                type="button"
                className={`${styles.navBtn} ${styles.navLeft}`}
                onClick={prev}
                aria-label="이전"
              >
                ‹
              </button>
              <img
                src={
                  process.env.PUBLIC_URL +
                  selected.screenshots[slideIdx]
                }
                alt={`${selected.name} 스크린샷 ${slideIdx + 1}`}
                className={styles.slideImage}
              />
              <button
                type="button"
                className={`${styles.navBtn} ${styles.navRight}`}
                onClick={next}
                aria-label="다음"
              >
                ›
              </button>
            </div>
            <p className={styles.counter}>
              {slideIdx + 1} / {selected.screenshots.length}
            </p>

            <div className={styles.modalDesc}>
              <p>{selected.description}</p>
              <ul className={styles.featureList}>
                {selected.features.map((f, i) => (
                  <li key={i}>{f}</li>
                ))}
              </ul>
            </div>

            <div className={styles.modalLinks}>
              {selected.privacyPath && (
                <Link to={selected.privacyPath} onClick={close}>
                  개인정보처리방침
                </Link>
              )}
              {selected.termsPath && (
                <Link to={selected.termsPath} onClick={close}>
                  이용약관
                </Link>
              )}
              {selected.deletePath && (
                <Link to={selected.deletePath} onClick={close}>
                  계정 삭제
                </Link>
              )}
            </div>

            <a
              href={selected.storeUrl}
              target="_blank"
              rel="noopener noreferrer"
              className={styles.modalStoreBtn}
              style={{ background: selected.color }}
            >
              <PlayBadge />
              <span>
                <small>GET IT ON</small>
                <strong>Google Play</strong>
              </span>
            </a>
          </div>
        </div>
      )}
    </section>
  );
};

export default AppsSection;
