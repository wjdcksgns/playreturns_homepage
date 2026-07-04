import styles from './History.module.css';
import { usePageTitle } from '../common/hooks/usePageTitle';

const historyData = [
  {
    year: '2026',
    events: [
      { text: '서울대학교 멘토-멘티 AI 기반 매칭 솔루션 개발', tags: ['Web', 'B2B'] },
      { text: '나만의 추억을 지도에 기록하는 리맵(RE:MAP) 출시', tags: ['Android', '자사 앱'] },
      { text: 'AI 사주풀이·궁합 분석 사주명 출시', tags: ['Android', '자사 앱'] },
      { text: '클래식 스도쿠 게임 출시', tags: ['Android', '자사 앱'] },
      { text: '서울대학교 PLUS+ 경진대회 온라인 심사 시스템 개발', tags: ['Web', 'B2B'] },
      { text: '반려동물 AI 관상·궁합 분석 앱 펫상 출시', tags: ['Android', '자사 앱'] },
      { text: '파리 키우기 캐주얼 게임 파리 살려! 출시', tags: ['Android', '자사 앱'] },
    ],
  },
  {
    year: '2025',
    events: [
      { text: '공사현장 시공 품질관리 어플리케이션 개발', tags: ['Android'] },
      { text: '카메라 영상과 UI 합성 안드로이드 기능 개발', tags: ['Android'] },
    ],
  },
  {
    year: '2024',
    events: [
      { text: 'Magic Leap 2 기반 원격협업 플랫폼 개발', tags: ['AR', 'Android'] },
      { text: '낭산 김준연 선생 기념관 체험 콘텐츠 개발', tags: ['AR', 'WebGL'] },
      { text: '피코(VR)와 HW 연동 동영상 스트리밍 콘텐츠 개발', tags: ['VR', 'Android'] },
    ],
  },
  {
    year: '2023',
    events: [
      { text: '슈퍼리듬스타 모바일 게임 출시', tags: ['Android'] },
      { text: 'CCTV 연계 AI 기반 경계 시스템 개발', tags: ['AR', 'PC'] },
      { text: '태전그룹과 AI 기반 안면인식 기술 활용 고객 관리 PoC 수행', tags: ['B2B', 'PoC'] },
    ],
  },
  {
    year: '2022',
    events: [
      { text: 'Nreal Light 기반 원격협업 플랫폼 개발', tags: ['AR', 'Android'] },
      { text: '삼성SDC와 원격협업 플랫폼 PoC 수행', tags: ['B2B', 'PoC'] },
    ],
  },
  {
    year: '2021',
    events: [
      { text: '홀로렌즈2 기반 원격협업 플랫폼 개발', tags: ['AR', 'UWP'] },
      { text: 'CCTV 연계 모니터링 시스템 개발', tags: ['AR', 'PC'] },
    ],
  },
  {
    year: '2020',
    events: [{ text: '박물관/식물원 체험 콘텐츠 개발', tags: ['VR'] }],
  },
  {
    year: '2019',
    events: [{ text: '구청 CCTV 모니터링 디지털트윈 개발', tags: ['PC'] }],
  },
];

const History = () => {
  usePageTitle('회사 연혁');

  return (
    <section className={styles.history}>
      <div className={styles.hero}>
        <p className={styles.eyebrow}>OUR HISTORY</p>
        <h2>플레이리턴즈의 발자취</h2>
        <p className={styles.subtitle}>
          AR/VR 콘텐츠부터 AI 솔루션, 자사 앱 출시까지 — 그동안의 여정입니다.
        </p>
      </div>

      <div className={styles.timeline}>
        <div className={styles.timelineLine} />
        {historyData.map((item, idx) => (
          <div key={idx} className={`${styles.timelineItem} reveal`}>
            <div className={styles.yearWrap}>
              <span className={styles.dot} />
              <span className={styles.year}>{item.year}</span>
            </div>
            <ul className={styles.events}>
              {item.events.map((event, i) => (
                <li key={i} className={styles.eventCard}>
                  <p className={styles.eventText}>{event.text}</p>
                  <div className={styles.tagRow}>
                    {event.tags.map((t) => (
                      <span key={t} className={styles.tag}>
                        {t}
                      </span>
                    ))}
                  </div>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </section>
  );
};

export default History;
