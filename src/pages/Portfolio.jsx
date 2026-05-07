import { useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import styles from './Portfolio.module.css';
import ContactModal from '../components/common/Modals/ContactModal';
import { usePageTitle } from '../common/hooks/usePageTitle';

const PUBLIC = process.env.PUBLIC_URL || '';
const portfolioBase = `${PUBLIC}/assets/images/portfolio/`;

const CATEGORIES = [
  { key: 'all', label: '전체' },
  { key: 'app', label: '자사 앱' },
  { key: 'b2b', label: 'B2B 프로젝트' },
];

const projects = [
  // ===== 자사 앱 =====
  {
    id: 'remap',
    category: 'app',
    title: '리맵 (RE:MAP)',
    tagline: '당신의 순간을 지도에 담다',
    year: '2026',
    platform: 'Android',
    accent: '#22c55e',
    summary:
      '지도 위에 사진과 추억을 핀으로 기록하는 여행 다이어리. AI 여행 일기, 추억 슬라이드쇼, PDF 여행책 생성까지.',
    description: `지도 위에 사진과 추억을 핀으로 기록하는 여행 다이어리 앱입니다. GPS 기반 자동 기록, AI 여행 일기 생성, 추억 슬라이드쇼와 PDF 여행책 제작 등 한 번의 여행을 다양한 형태로 남길 수 있도록 지원합니다.`,
    features: [
      '지도 위 위치 기반 핀맵 기록',
      '사진·영상·텍스트 업로드 및 보관',
      'AI 여행 일기 / 여행지 추천 (구독)',
      '추억 슬라이드쇼 영상 · PDF 여행책 생성',
      '공개 기록으로 다른 여행자와 공유',
    ],
    tech: ['Android', 'Google Maps API', 'Firebase'],
    storeUrl:
      'https://play.google.com/store/apps/details?id=kr.co.playreturns.remap',
    privacyPath: '/remapprivacy',
    termsPath: '/terms',
    cover: `${PUBLIC}/images/remap/real/KakaoTalk_20260428_113408623.jpg`,
    images: [
      `${PUBLIC}/images/remap/real/KakaoTalk_20260428_113408623.jpg`,
      `${PUBLIC}/images/remap/real/KakaoTalk_20260428_113408623_01.jpg`,
      `${PUBLIC}/images/remap/real/KakaoTalk_20260428_113408623_02.jpg`,
      `${PUBLIC}/images/remap/real/KakaoTalk_20260428_113408623_03.jpg`,
      `${PUBLIC}/images/remap/real/KakaoTalk_20260428_113408623_04.jpg`,
      `${PUBLIC}/images/remap/real/KakaoTalk_20260428_113408623_05.jpg`,
      `${PUBLIC}/images/remap/real/KakaoTalk_20260428_113408623_06.jpg`,
      `${PUBLIC}/images/remap/real/KakaoTalk_20260428_113408623_07.jpg`,
      `${PUBLIC}/images/remap/real/KakaoTalk_20260428_113408623_08.jpg`,
      `${PUBLIC}/images/remap/real/KakaoTalk_20260428_113408623_09.jpg`,
      `${PUBLIC}/images/remap/real/KakaoTalk_20260428_113408623_10.jpg`,
    ],
  },
  {
    id: 'saju',
    category: 'app',
    title: '사주명',
    tagline: 'AI 사주풀이 · 궁합 분석',
    year: '2026',
    platform: 'Android',
    accent: '#d4af37',
    summary:
      '정통 사주명리에 AI의 분석력을 더한 맞춤형 운세 서비스. 이름·생년월일만으로 AI가 사주와 궁합을 풀어드립니다.',
    description: `정통 사주명리에 AI의 분석력을 더한 맞춤형 운세 서비스입니다. 이름·생년월일·태어난 시각만 입력하면 사주풀이와 궁합 분석 결과를 받아볼 수 있습니다.`,
    features: [
      '연애·결혼·재물 등 분야별 사주풀이',
      '두 사람의 궁합 분석',
      '관심 월·궁금한 운에 맞춘 맞춤 해석',
      '최근 10건 히스토리 자동 저장',
      'Google · 카카오 간편 로그인',
    ],
    tech: ['Android', 'Google OAuth', 'Kakao Login', 'AdMob'],
    storeUrl:
      'https://play.google.com/store/apps/details?id=com.playreturns.sajuyeon',
    privacyPath: '/sajuprivacy',
    cover: `${PUBLIC}/images/saju/store/sajuMyeong_feature_graphic.jpg`,
    images: [
      `${PUBLIC}/images/saju/real/스크린샷_1.jpg`,
      `${PUBLIC}/images/saju/real/스크린샷_2.jpg`,
      `${PUBLIC}/images/saju/real/스크린샷_3.jpg`,
      `${PUBLIC}/images/saju/real/스크린샷_4.jpg`,
    ],
  },
  {
    id: 'sudoku',
    category: 'app',
    title: '스도쿠',
    tagline: '숫자 퍼즐 두뇌 게임',
    year: '2026',
    platform: 'Android',
    accent: '#3b82f6',
    summary:
      '클래식 스도쿠를 깔끔한 디자인으로. 다양한 난이도와 일일 챌린지, Google 계정 동기화까지 지원합니다.',
    description: `클래식 스도쿠를 깔끔한 디자인으로 즐길 수 있는 두뇌 퍼즐 게임입니다. 다양한 난이도와 일일 챌린지로 매일 새로운 두뇌 운동을 즐길 수 있고, Google 계정으로 기기 간 진행 상황을 동기화할 수 있습니다.`,
    features: [
      '쉬움부터 매우 어려움까지 4단계 난이도',
      '일일 챌린지 모드',
      '힌트 / 메모 / 자동 검사 기능',
      'Google 계정으로 기기 간 동기화',
      '깔끔한 다크모드 지원',
    ],
    tech: ['Android', 'Firebase', 'Google OAuth', 'AdMob'],
    storeUrl:
      'https://play.google.com/store/apps/details?id=com.sudokuapp.sudoku_game',
    privacyPath: '/sudoku-privacy',
    deletePath: '/sudoku-delete-account',
    cover: `${PUBLIC}/images/sudoku/store/graphic_ko.png`,
    images: [
      `${PUBLIC}/images/sudoku/real/KakaoTalk_20260428_113852741.jpg`,
      `${PUBLIC}/images/sudoku/real/KakaoTalk_20260428_113852741_01.jpg`,
      `${PUBLIC}/images/sudoku/real/KakaoTalk_20260428_113852741_02.jpg`,
      `${PUBLIC}/images/sudoku/real/KakaoTalk_20260428_113852741_03.jpg`,
      `${PUBLIC}/images/sudoku/real/KakaoTalk_20260428_113852741_04.jpg`,
      `${PUBLIC}/images/sudoku/real/KakaoTalk_20260428_113852741_05.jpg`,
    ],
  },

  // ===== B2B 프로젝트 =====
  {
    id: 'face',
    category: 'b2b',
    title: 'AI 안면인식 솔루션',
    tagline: '실시간 얼굴 인식 기반 헬스케어/보안 통합 관리',
    year: '2023~',
    platform: 'PC · Android',
    accent: '#6366f1',
    summary:
      '실시간 안면인식으로 방문 이력과 증상·처방 정보를 통합 관리하는 스마트 헬스케어/보안 솔루션입니다.',
    description: `실시간 안면인식으로 방문 이력과 증상·처방 정보를 통합 관리하는 스마트 헬스케어/보안 솔루션입니다. 폐쇄망과 암호화로 개인정보를 보호하면서, 데이터 기반의 맞춤 상담·추천 서비스 확장도 지원합니다.`,
    features: [
      '비마커 기반 실시간 얼굴 인식',
      '방문·증상·처방 이력 통합 조회·시각화',
      '폐쇄망(Local Network)·암호화 등 개인정보 보호',
      '데이터 기반 맞춤 상담/추천 서비스 확장',
    ],
    tech: ['Deep Learning', 'OpenCV', 'PyTorch', 'On-premise'],
    cover: `${portfolioBase}face_1.png`,
    images: [
      `${portfolioBase}face_1.png`,
      `${portfolioBase}face_2.png`,
      `${portfolioBase}face_3.png`,
    ],
  },
  {
    id: 'rhythm',
    category: 'b2b',
    title: '슈퍼리듬스타',
    tagline: '도트 캐릭터 × 스토리 진행 리듬 액션 게임',
    year: '2023',
    platform: 'Android',
    accent: '#ec4899',
    summary:
      "귀여운 도트 캐릭터와 스토리 진행이 결합된 리듬 액션 게임. 다양한 음악 장르와 손맛 나는 타격감으로 색다른 재미를 제공합니다.",
    description: `귀여운 도트 캐릭터와 스토리 진행이 결합된 리듬 액션 게임. 'ONE! TWO! THREE! GO!' 큐에 맞춘 타이밍 플레이와 다채로운 음악 장르, 손맛 나는 타격감으로 색다른 재미를 제공합니다.`,
    features: [
      "'ONE! TWO! THREE! GO!' 큐에 맞춘 타이밍 플레이",
      '스토리 × 리듬의 색다른 진행',
      '다채로운 음악 장르와 그래픽/사운드 이펙트',
      '리더보드로 점수 경쟁 및 친구와 순위 비교',
    ],
    tech: ['Unity', 'C#', 'Android'],
    cover: `${portfolioBase}rhythm_1.jpg`,
    images: [
      `${portfolioBase}rhythm_1.jpg`,
      `${portfolioBase}rhythm_2.jpg`,
      `${portfolioBase}rhythm_3.png`,
      `${portfolioBase}rhythm_4.jpg`,
      `${portfolioBase}rhythm_5.jpg`,
      `${portfolioBase}rhythm_6.jpg`,
      `${portfolioBase}rhythm_7.jpg`,
      `${portfolioBase}rhythm_8.png`,
      `${portfolioBase}rhythm_9.jpg`,
    ],
  },
  {
    id: 'ar',
    category: 'b2b',
    title: 'AR 원격협업 플랫폼',
    tagline: 'AI · IoT 기반 다자간 비대면 원격 협업',
    year: '2021~2024',
    platform: 'AR · Android · UWP',
    accent: '#0ea5e9',
    summary:
      'AI/IoT 기반 AR 원격협업 플랫폼. 실시간 영상·음성 스트리밍과 2D/3D 모델링으로 현장 인력과 원격 전문가의 협업을 지원합니다.',
    description: `AI와 IoT 기반의 AR 원격협업 플랫폼입니다. 실시간 영상·음성 스트리밍과 2D/3D 모델링 시뮬레이션을 통해 현장 인력과 원격 전문가가 같은 공간을 공유하며 협력할 수 있도록 지원합니다. HoloLens 2, Magic Leap 2, Nreal 등 다양한 AR 기기를 지원했습니다.`,
    features: [
      '다자간 비대면 원격 협업 (WebRTC 기반 영상/음성)',
      '2D/3D 모델링 및 시뮬레이션 협업',
      'IoT 데이터 연동으로 실시간 현장 정보 제공',
      '음성·영상·AR 모델·파일 공유',
    ],
    tech: ['Unity', 'WebRTC', 'HoloLens 2', 'Magic Leap 2', 'Nreal'],
    cover: `${portfolioBase}ar_1.png`,
    images: [
      `${portfolioBase}ar_1.png`,
      `${portfolioBase}ar_2.png`,
      `${portfolioBase}ar_3.png`,
      `${portfolioBase}ar_4.png`,
    ],
  },
];

const PlayBadge = () => (
  <svg className={styles.playIcon} viewBox="0 0 24 24" aria-hidden="true">
    <path
      fill="currentColor"
      d="M3 2.5v19l8.5-9.5L3 2.5zm10.2 8.4l3.4-1.9L4.6 2.6l8.6 8.3zm0 2.2l-8.6 8.3L16.6 15l-3.4-1.9zm9-2.6l-4.4-2.5-3.7 4.1 3.7 4.1 4.4-2.5c1-.6 1-2 0-2.7v-.5z"
    />
  </svg>
);

const Portfolio = () => {
  usePageTitle('포트폴리오');
  const [filter, setFilter] = useState('all');
  const [selected, setSelected] = useState(null);
  const [slide, setSlide] = useState(0);
  const [showContact, setShowContact] = useState(false);

  const filtered = useMemo(
    () =>
      filter === 'all'
        ? projects
        : projects.filter((p) => p.category === filter),
    [filter]
  );

  const open = (project) => {
    setSelected(project);
    setSlide(0);
  };
  const close = () => setSelected(null);

  const prev = () =>
    setSlide((i) => (i === 0 ? selected.images.length - 1 : i - 1));
  const next = () =>
    setSlide((i) => (i === selected.images.length - 1 ? 0 : i + 1));

  return (
    <section className={styles.portfolio}>
      <div className={styles.hero}>
        <p className={styles.eyebrow}>OUR PORTFOLIO</p>
        <h2>플레이리턴즈가 만든 것들</h2>
        <p className={styles.subtitle}>
          자사 출시 앱부터 다양한 산업 분야의 B2B 프로젝트까지, 그동안의 작업을
          소개합니다.
        </p>
      </div>

      <div className={styles.filters}>
        {CATEGORIES.map((cat) => {
          const count =
            cat.key === 'all'
              ? projects.length
              : projects.filter((p) => p.category === cat.key).length;
          return (
            <button
              key={cat.key}
              type="button"
              className={`${styles.filterBtn} ${
                filter === cat.key ? styles.filterActive : ''
              }`}
              onClick={() => setFilter(cat.key)}
            >
              {cat.label}
              <span className={styles.count}>{count}</span>
            </button>
          );
        })}
      </div>

      <div className={styles.grid}>
        {filtered.map((p) => (
          <article
            key={p.id}
            className={`${styles.card} reveal`}
            style={{ '--accent': p.accent }}
          >
            <button
              type="button"
              className={styles.cardImage}
              onClick={() => open(p)}
              aria-label={`${p.title} 자세히 보기`}
            >
              <img src={p.cover} alt={p.title} loading="lazy" />
              <span
                className={`${styles.badge} ${
                  p.category === 'app' ? styles.badgeApp : styles.badgeB2B
                }`}
              >
                {p.category === 'app' ? '자사 앱' : 'B2B 프로젝트'}
              </span>
              <div className={styles.imageOverlay}>
                <span>자세히 보기 →</span>
              </div>
            </button>

            <div className={styles.cardBody}>
              <div className={styles.metaRow}>
                <span className={styles.year}>{p.year}</span>
                <span className={styles.dot}>·</span>
                <span className={styles.platform}>{p.platform}</span>
              </div>
              <h3 className={styles.title}>{p.title}</h3>
              <p className={styles.tagline}>{p.tagline}</p>
              <p className={styles.summary}>{p.summary}</p>

              <div className={styles.tags}>
                {p.tech.map((t) => (
                  <span key={t} className={styles.tag}>
                    {t}
                  </span>
                ))}
              </div>

              <div className={styles.cardActions}>
                <button
                  type="button"
                  className={styles.detailBtn}
                  onClick={() => open(p)}
                >
                  자세히 보기
                </button>
                {p.storeUrl && (
                  <a
                    href={p.storeUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className={styles.storeBtn}
                  >
                    <PlayBadge />
                    Google Play
                  </a>
                )}
              </div>
            </div>
          </article>
        ))}
      </div>

      {/* 모달 */}
      {selected && (
        <div className={styles.modalOverlay} onClick={close}>
          <div
            className={styles.modal}
            style={{ '--accent': selected.accent }}
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
              <span
                className={`${styles.badge} ${
                  selected.category === 'app'
                    ? styles.badgeApp
                    : styles.badgeB2B
                }`}
              >
                {selected.category === 'app' ? '자사 앱' : 'B2B 프로젝트'}
              </span>
              <h3>{selected.title}</h3>
              <p className={styles.modalTagline}>{selected.tagline}</p>
              <p className={styles.modalMeta}>
                {selected.year} · {selected.platform}
              </p>
            </div>

            <div className={styles.slider}>
              <button
                type="button"
                className={`${styles.navBtn} ${styles.navLeft}`}
                onClick={prev}
                aria-label="이전"
              >
                ‹
              </button>
              <img
                src={selected.images[slide]}
                alt={`${selected.title} ${slide + 1}`}
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
              {slide + 1} / {selected.images.length}
            </p>

            <div className={styles.modalDesc}>
              <p>{selected.description}</p>
              <ul className={styles.featureList}>
                {selected.features.map((f, i) => (
                  <li key={i}>{f}</li>
                ))}
              </ul>
            </div>

            <div className={styles.modalTags}>
              {selected.tech.map((t) => (
                <span key={t} className={styles.tag}>
                  {t}
                </span>
              ))}
            </div>

            {(selected.privacyPath ||
              selected.termsPath ||
              selected.deletePath) && (
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
            )}

            {selected.storeUrl ? (
              <a
                href={selected.storeUrl}
                target="_blank"
                rel="noopener noreferrer"
                className={styles.modalCta}
              >
                <PlayBadge />
                <span>
                  <small>GET IT ON</small>
                  <strong>Google Play</strong>
                </span>
              </a>
            ) : (
              <button
                type="button"
                className={styles.modalCta}
                onClick={() => {
                  close();
                  setShowContact(true);
                }}
              >
                <span>
                  <strong>이런 프로젝트가 필요하신가요? 문의하기</strong>
                </span>
              </button>
            )}
          </div>
        </div>
      )}

      <div className={`${styles.contactBox} reveal`}>
        <p>비슷한 프로젝트를 검토하고 계신가요?</p>
        <button
          type="button"
          onClick={() => setShowContact(true)}
          className={styles.contactBtn}
        >
          프로젝트 문의하기
        </button>
      </div>

      {showContact && <ContactModal onClose={() => setShowContact(false)} />}
    </section>
  );
};

export default Portfolio;
