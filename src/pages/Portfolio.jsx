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
  {
    id: 'petsang',
    category: 'app',
    title: '펫상',
    tagline: '반려동물 AI 관상·궁합 분석',
    year: '2026',
    platform: 'Android',
    accent: '#f97316',
    summary:
      '반려동물 사진 한 장으로 관상·MBTI·닮은꼴·운세·궁합까지 AI가 분석해주는 재미 앱입니다.',
    description: `반려동물 사진 한 장으로 AI가 관상, 성격 MBTI, 닮은꼴, 오늘의 운세, 궁합까지 분석해주는 재미 앱입니다. 분석 결과는 히스토리로 저장하고 친구와 공유할 수 있으며, 업로드한 사진은 분석 후 24시간 이내에 자동 삭제되어 개인정보를 보호합니다.`,
    features: [
      '반려동물 사진 기반 AI 관상 분석',
      '성격 MBTI · 닮은꼴 · 오늘의 운세',
      '두 반려동물의 궁합 분석',
      '분석 결과 히스토리 저장 · 공유',
      '사진 24시간 자동 삭제로 개인정보 보호',
    ],
    tech: ['Android', 'GPT-4o', 'Firebase', 'AdMob'],
    storeUrl:
      'https://play.google.com/store/apps/details?id=com.playreturns.petsang',
    privacyPath: '/petsang-privacy',
    termsPath: '/petsang-terms',
    deletePath: '/petsang-delete-account',
    cover: `${PUBLIC}/images/petsang/feature_graphic.webp`,
    images: [
      `${PUBLIC}/images/petsang/01_home.webp`,
      `${PUBLIC}/images/petsang/02_register.webp`,
      `${PUBLIC}/images/petsang/03_upload.webp`,
      `${PUBLIC}/images/petsang/04_photo_ready.webp`,
      `${PUBLIC}/images/petsang/05_fortune.webp`,
      `${PUBLIC}/images/petsang/06_mbti.webp`,
      `${PUBLIC}/images/petsang/07_share.webp`,
    ],
  },
  {
    id: 'savefly',
    category: 'app',
    title: '파리 살려!',
    tagline: '파리 키우기 방치형 캐주얼 게임',
    year: '2026',
    platform: 'Android',
    accent: '#65a30d',
    summary:
      '작은 파리 한 마리를 먹이고 돌보며 키우는 방치형 게임. 스킨·칭호를 모으고 랭킹과 명예의 전당에 도전하세요.',
    description: `작은 파리 한 마리를 먹이고 돌보며 키우는 방치형(idle) 캐주얼 게임입니다. 다양한 스킨과 칭호를 수집하고 일일 퀘스트를 완료하며, 온라인 랭킹과 명예의 전당에서 다른 이용자와 경쟁할 수 있습니다. Google 계정으로 진행 상황을 클라우드에 동기화합니다.`,
    features: [
      '파리를 먹이고 돌보는 방치형 키우기',
      '다양한 스킨 · 칭호 수집',
      '일일 퀘스트 · 인벤토리 시스템',
      '온라인 랭킹 · 명예의 전당 경쟁',
      'Google 계정 클라우드 동기화',
    ],
    tech: ['Android', 'Flutter', 'Firebase', 'AdMob'],
    storeUrl:
      'https://play.google.com/store/apps/details?id=com.savefly.save_the_fly',
    privacyPath: '/savefly-privacy',
    termsPath: '/savefly-terms',
    deletePath: '/savefly-delete-account',
    cover: `${PUBLIC}/images/savefly/feature_graphic.webp`,
    images: [
      `${PUBLIC}/images/savefly/01_main.webp`,
      `${PUBLIC}/images/savefly/02_title.webp`,
      `${PUBLIC}/images/savefly/03_skin.webp`,
      `${PUBLIC}/images/savefly/04_ranking.webp`,
      `${PUBLIC}/images/savefly/05_hall_of_fame.webp`,
      `${PUBLIC}/images/savefly/06_daily_quest.webp`,
      `${PUBLIC}/images/savefly/07_inventory.webp`,
    ],
  },
  {
    id: 'rhythm',
    category: 'app',
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

  // ===== B2B 프로젝트 =====
  {
    id: 'snu-matching',
    category: 'b2b',
    title: '서울대학교 멘토-멘티 AI 매칭 시스템',
    client: '서울대학교 글로벌사회공헌단',
    tagline: 'AI 기반 멘토-멘티 매칭 자동화',
    year: '2026',
    platform: 'Web · AI',
    accent: '#1a4fa0',
    graphic: 'match',
    summary:
      '수작업 중심의 멘토-멘티 매칭을 AI 분석·자동화 시스템으로 전환. 서술형 응답을 정량화해 전 조합을 전수 평가하고, 공정하고 설명 가능한 매칭 기준을 수립했습니다.',
    description: `서울대학교 글로벌사회공헌단의 SNU 멘토링 프로그램을 위해, 기존 담당자 경험에 의존하던 수작업 매칭을 AI 기반 분석·자동화 시스템으로 전환한 프로젝트입니다. 멘토·멘티가 제출한 서술형 응답을 자연어 분석으로 정량화하고, 모든 조합의 적합도를 산출하는 전수 평가 구조를 설계했습니다. 성별·학교급 등 운영 원칙을 선제적 제약 조건으로 적용한 뒤, AI 점수를 기준으로 단계적 매칭을 수행합니다.`,
    features: [
      '서술형 응답(비고·경험·취미·진로) 자연어 분석·정량화',
      '전체 멘토-멘티 점수 행렬 기반 전수 평가',
      '성별·학교급 등 필수 조건 단계적 반영',
      'Hard / Soft / Interest 다단계 매칭 로직',
      '관리자 페이지 CSV 업로드 → 자동 검증 → 결과 다운로드',
    ],
    tech: ['Python', 'AI / NLP', '매칭 알고리즘', 'CSV 파이프라인'],
  },
  {
    id: 'snu-judging',
    category: 'b2b',
    title: '서울대학교 PLUS+ 경진대회 심사 시스템',
    client: '서울대학교 글로벌사회공헌단',
    tagline: '태블릿 기반 온라인 심사 · 표준화 채점',
    year: '2026',
    platform: 'Web · Tablet',
    accent: '#0e5aa8',
    graphic: 'judge',
    summary:
      '글로벌 사회공헌 PLUS+ 경진대회의 예선·사전평가·본선 3단계 심사를 웹으로 처리. Z-score 표준화로 심사위원 편차를 보정하고, 순위와 상훈까지 자동 산출합니다.',
    description: `서울대학교 글로벌사회공헌단이 운영하는 PLUS+ 경진대회의 심사 전 과정을 웹으로 처리하는 온라인 심사 시스템입니다. 심사위원은 태블릿(갤럭시탭)으로 접속해 발표자료 뷰어와 채점표를 한 화면에서 확인하며 채점하고, 관리자는 채점 현황과 결과를 실시간으로 관리합니다. 심사위원마다 다른 점수 기준을 Z-score로 표준화해 편차를 보정하고, 예선·사전평가·본선 비중을 반영한 최종 순위와 상훈을 자동으로 산출합니다.`,
    features: [
      '예선(온라인)·사전평가·본선(현장 태블릿) 3단계 심사',
      '제안서·발표자료 뷰어 + 채점표 좌우 분할 레이아웃',
      'Z-score 표준화 점수 엔진으로 심사위원 편차 보정',
      '승인제 심사위원 계정 · 실시간 채점 현황 모니터링',
      '순위-상훈 자동 매핑 · 결과 엑셀 다운로드',
    ],
    tech: ['Next.js', 'PostgreSQL', 'Prisma', 'Z-score 표준화'],
  },
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

const GRAPHIC_ICONS = {
  match: (
    <svg viewBox="0 0 48 48" fill="none" aria-hidden="true">
      <circle cx="12" cy="14" r="5" stroke="currentColor" strokeWidth="2.4" />
      <circle cx="36" cy="14" r="5" stroke="currentColor" strokeWidth="2.4" />
      <circle cx="12" cy="34" r="5" stroke="currentColor" strokeWidth="2.4" />
      <circle cx="36" cy="34" r="5" stroke="currentColor" strokeWidth="2.4" />
      <path
        d="M17 14h14M17 34h14M12 19v10M36 19v10M16 18l16 12"
        stroke="currentColor"
        strokeWidth="2.4"
        strokeLinecap="round"
      />
    </svg>
  ),
  judge: (
    <svg viewBox="0 0 48 48" fill="none" aria-hidden="true">
      <rect
        x="9"
        y="7"
        width="30"
        height="34"
        rx="4"
        stroke="currentColor"
        strokeWidth="2.4"
      />
      <path
        d="M17 18h14M17 25h14M17 32h9"
        stroke="currentColor"
        strokeWidth="2.4"
        strokeLinecap="round"
      />
      <path
        d="M31.5 34.5l2.5 2.5 5-5.5"
        stroke="currentColor"
        strokeWidth="2.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  ),
};

const GraphicCover = ({ project }) => (
  <div className={styles.graphicCover}>
    <div className={styles.graphicIcon}>{GRAPHIC_ICONS[project.graphic]}</div>
    {project.client && (
      <span className={styles.graphicClient}>{project.client}</span>
    )}
  </div>
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
              {p.cover ? (
                <img src={p.cover} alt={p.title} loading="lazy" />
              ) : (
                <GraphicCover project={p} />
              )}
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
              {selected.client && (
                <p className={styles.modalClient}>{selected.client}</p>
              )}
              <p className={styles.modalMeta}>
                {selected.year} · {selected.platform}
              </p>
            </div>

            {selected.images ? (
              <>
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
              </>
            ) : (
              <div className={styles.modalGraphic}>
                <div className={styles.modalGraphicIcon}>
                  {GRAPHIC_ICONS[selected.graphic]}
                </div>
              </div>
            )}

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
            ) : selected.category === 'b2b' ? (
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
            ) : null}
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
