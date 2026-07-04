import styles from './About.module.css';
import {
  FaMobileAlt,
  FaCube,
  FaGlasses,
  FaBrain,
  FaVideo,
  FaPaintBrush,
  FaProjectDiagram,
} from 'react-icons/fa';
import { usePageTitle } from '../common/hooks/usePageTitle';
import ceoSignature from '../assets/images/ceo_signature.png';

const services = [
  { icon: <FaMobileAlt />, label: 'Android / iOS 어플리케이션 개발' },
  { icon: <FaPaintBrush />, label: 'Unity 엔진 기반 콘텐츠 개발' },
  { icon: <FaGlasses />, label: 'AR / VR / 메타버스 콘텐츠 개발' },
  { icon: <FaBrain />, label: 'AI 기능 융합 및 지능형 시스템 구현' },
  { icon: <FaProjectDiagram />, label: 'AI 매칭·평가 자동화 등 B2B 시스템 개발' },
  { icon: <FaVideo />, label: 'CCTV · 디지털트윈 · 실시간 협업 시스템' },
  { icon: <FaCube />, label: 'Unity Asset 기반 UI/모델링 + 맞춤 제작' },
];

const About = () => {
  usePageTitle('회사 소개');

  return (
    <section className={styles.about}>
      {/* ===== Hero ===== */}
      <div className={styles.hero}>
        <p className={styles.eyebrow}>ABOUT US</p>
        <h2>플레이리턴즈를 소개합니다</h2>
        <p className={styles.subtitle}>
          현실과 디지털을 연결하고, 경험을 가치로 바꾸는 기술을 만듭니다.
        </p>
      </div>

      {/* ===== Vision ===== */}
      <div className={`${styles.visionBlock} reveal`}>
        <p className={styles.smallLabel}>OUR VISION</p>
        <h3 className={styles.visionTitle}>
          “SEE SOMETHING,<br />
          KNOW SOMETHING”
        </h3>
        <p className={styles.visionText}>
          플레이리턴즈는 세상을 더 깊이 보고, 더 정확히 이해하는 기술을 만듭니다.
          단순한 콘텐츠 제작을 넘어 현실과 디지털을 연결하고, 경험을 가치로 바꾸는
          혁신을 지향합니다.
        </p>
      </div>

      {/* ===== What We Do ===== */}
      <div className={`${styles.servicesBlock} reveal`}>
        <div className={styles.sectionHeader}>
          <p className={styles.smallLabel}>WHAT WE DO</p>
          <h3>우리가 만드는 것</h3>
        </div>
        <div className={styles.serviceGrid}>
          {services.map((s, i) => (
            <div key={i} className={styles.serviceCard}>
              <div className={styles.serviceIcon}>{s.icon}</div>
              <p>{s.label}</p>
            </div>
          ))}
        </div>
      </div>

      {/* ===== Promise ===== */}
      <div className={`${styles.promiseBlock} reveal`}>
        <p className={styles.smallLabel}>OUR PROMISE</p>
        <h3>고객의 아이디어가 자라는 파트너</h3>
        <p>
          플레이리턴즈는 단순히 결과물을 만드는 것이 아니라,
          고객의 아이디어가 실제로 구현되고 성장할 수 있도록 돕는 파트너가 되겠습니다.
        </p>

        <div className={styles.signatureBlock}>
          <img
            src={ceoSignature}
            alt="대표이사 정유진 서명"
            className={styles.signature}
          />
          <p className={styles.signatureName}>
            <span className={styles.signatureRole}>대표이사</span>
            <span className={styles.signatureCeo}>정유진</span>
          </p>
        </div>
      </div>
    </section>
  );
};

export default About;
