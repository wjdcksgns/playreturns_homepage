import { useNavigate } from 'react-router-dom';
import styles from './SnuSection.module.css';

const MatchIcon = () => (
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
);

const JudgeIcon = () => (
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
);

const projects = [
  {
    id: 'snu-matching',
    icon: <MatchIcon />,
    title: '멘토-멘티 AI 매칭 시스템',
    desc:
      '수작업으로 진행하던 멘토-멘티 매칭을 AI 분석·자동화로 전환. 서술형 응답을 정량화해 전수 평가하고, 공정하고 설명 가능한 매칭 기준을 수립했습니다.',
    tags: ['AI / NLP', '매칭 알고리즘', '자동화'],
  },
  {
    id: 'snu-judging',
    icon: <JudgeIcon />,
    title: 'PLUS+ 경진대회 온라인 심사 시스템',
    desc:
      '예선·사전평가·본선 3단계 심사를 웹으로 처리. Z-score 표준화로 심사위원 간 점수 편차를 보정하고, 순위·상훈까지 자동 산출하는 태블릿 최적화 시스템입니다.',
    tags: ['Next.js', '표준화 채점', '태블릿 심사'],
  },
];

const SnuSection = () => {
  const navigate = useNavigate();

  return (
    <section className={styles.section}>
      <div className={styles.inner}>
        <div className={`${styles.heading} reveal`}>
          <p className={styles.eyebrow}>FEATURED B2B PROJECT</p>
          <h2>서울대학교와 함께한 프로젝트</h2>
          <p className={styles.subtitle}>
            <strong>서울대학교 글로벌사회공헌단</strong>의 멘토링·경진대회 운영을 위한
            AI·웹 시스템을 플레이리턴즈가 설계하고 구축했습니다.
          </p>
        </div>

        <div className={styles.grid}>
          {projects.map((p) => (
            <article
              key={p.id}
              className={`${styles.card} reveal`}
              onClick={() => navigate('/portfolio')}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  navigate('/portfolio');
                }
              }}
            >
              <span className={styles.badge}>서울대학교</span>
              <div className={styles.icon}>{p.icon}</div>
              <h3>{p.title}</h3>
              <p className={styles.desc}>{p.desc}</p>
              <div className={styles.tags}>
                {p.tags.map((t) => (
                  <span key={t} className={styles.tag}>
                    {t}
                  </span>
                ))}
              </div>
              <span className={styles.more}>포트폴리오에서 보기 →</span>
            </article>
          ))}
        </div>
      </div>
    </section>
  );
};

export default SnuSection;
