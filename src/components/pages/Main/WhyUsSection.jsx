import { FaLayerGroup, FaRocket, FaIndustry } from 'react-icons/fa';
import { useReveal } from '../../../common/hooks/useReveal';
import styles from './WhyUsSection.module.css';

const items = [
  {
    icon: <FaLayerGroup />,
    title: '기술 융합 역량',
    text:
      'AR/VR, AI, 디지털트윈, 모바일·웹 — 한 회사가 다 다룹니다. 분야가 결합된 프로젝트도 외주 없이 자체 인력으로 완성합니다.',
    accent: '#6366f1',
  },
  {
    icon: <FaRocket />,
    title: 'B2B + 자체 서비스',
    text:
      '수주 프로젝트만 하지 않습니다. 직접 앱을 출시하고 운영하면서 제품 기획부터 운영까지의 노하우를 쌓아왔습니다.',
    accent: '#ec4899',
  },
  {
    icon: <FaIndustry />,
    title: '현장 적용 경험',
    text:
      'CCTV·안면인식·디지털트윈·AR 협업 등 실제 산업 현장에 적용한 경험. 데모가 아닌 운영 중인 시스템을 만들어왔습니다.',
    accent: '#10b981',
  },
];

const WhyUsSection = () => {
  const ref = useReveal();

  return (
    <section ref={ref} className={`${styles.section} reveal`}>
      <div className={styles.heading}>
        <p className={styles.eyebrow}>WHY PLAYRETURNS</p>
        <h2>저희를 선택하실 이유</h2>
        <p className={styles.subtitle}>
          기술의 폭, 운영의 깊이, 현장 검증 — 세 가지를 모두 갖춘 곳은 흔하지 않습니다.
        </p>
      </div>

      <div className={styles.grid}>
        {items.map((item) => (
          <div
            key={item.title}
            className={styles.card}
            style={{ '--accent': item.accent }}
          >
            <div className={styles.iconBox}>{item.icon}</div>
            <h3 className={styles.cardTitle}>{item.title}</h3>
            <p className={styles.cardText}>{item.text}</p>
          </div>
        ))}
      </div>
    </section>
  );
};

export default WhyUsSection;
