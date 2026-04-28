import { useReveal } from '../../../common/hooks/useReveal';
import styles from './StatsSection.module.css';

const stats = [
  { value: '7+', label: 'Years', sub: '2019년부터 쌓아온 경험' },
  { value: '20+', label: 'Projects', sub: 'B2B 수주 · 자체 개발 합산' },
  { value: '3', label: 'Apps', sub: '리맵 · 사주명 · 스도쿠 출시' },
  { value: '8', label: 'Tech Areas', sub: 'AR/VR · AI · 디지털트윈 외' },
];

const StatsSection = () => {
  const ref = useReveal();

  return (
    <section ref={ref} className={`${styles.section} reveal`}>
      <div className={styles.bg}>
        <div className={styles.blob1} />
        <div className={styles.blob2} />
      </div>

      <div className={styles.inner}>
        <div className={styles.heading}>
          <p className={styles.eyebrow}>BY THE NUMBERS</p>
          <h2>숫자로 보는 플레이리턴즈</h2>
        </div>

        <div className={styles.grid}>
          {stats.map((s, i) => (
            <div key={s.label} className={styles.statCard}>
              <span className={styles.statValue}>{s.value}</span>
              <span className={styles.statLabel}>{s.label}</span>
              <span className={styles.statSub}>{s.sub}</span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default StatsSection;
