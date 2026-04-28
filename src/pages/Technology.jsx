import { useState } from 'react';
import {
  FaVideo,
  FaUserCheck,
  FaGlasses,
  FaMobileAlt,
  FaBrain,
  FaCube,
  FaGlobe,
  FaEye,
} from 'react-icons/fa';
import styles from './Technology.module.css';
import ContactModal from '../components/common/Modals/ContactModal';
import { usePageTitle } from '../common/hooks/usePageTitle';

const techList = [
  {
    title: 'CCTV 활용 감지 기술',
    description:
      'CCTV와 센서를 결합하여 객체·이상행동·침입 등을 실시간 탐지합니다.',
    fields: ['보안', '안전 관리', '스마트시티', '공장 모니터링'],
    icon: <FaVideo />,
    accent: '#0ea5e9',
  },
  {
    title: '안면인식 기술',
    description:
      '딥러닝 기반 얼굴 인식 알고리즘으로 신원 확인, 출입 통제, 맞춤형 서비스를 제공합니다.',
    fields: ['보안 출입 시스템', '근태 관리', '맞춤형 서비스'],
    icon: <FaUserCheck />,
    accent: '#6366f1',
  },
  {
    title: 'AR 활용 원격협업 기술',
    description:
      'AR 글래스와 태블릿을 활용해 원격에서도 동일한 공간을 공유하며 협업할 수 있습니다.',
    fields: ['산업 현장', '교육/훈련', '원격 지원'],
    icon: <FaGlasses />,
    accent: '#06b6d4',
  },
  {
    title: 'Android · iOS 어플리케이션 개발',
    description:
      '네이티브와 크로스플랫폼 기반으로 다양한 모바일 앱을 설계·개발합니다.',
    fields: ['엔터테인먼트', '교육', '비즈니스', 'IoT 연동'],
    icon: <FaMobileAlt />,
    accent: '#22c55e',
  },
  {
    title: 'LLM & AI Agent 개발',
    description:
      'LLM과 AI 에이전트를 활용해 복잡한 문제 해결과 자동화된 의사결정을 지원합니다.',
    fields: ['대화형 서비스', '업무 자동화', '지능형 데이터 분석'],
    icon: <FaBrain />,
    accent: '#8b5cf6',
  },
  {
    title: '디지털트윈 개발',
    description:
      '현실 데이터를 실시간으로 반영하는 디지털 공간을 구축해 모니터링과 시뮬레이션이 가능합니다.',
    fields: ['스마트 팩토리', '건설', '물류 관리'],
    icon: <FaCube />,
    accent: '#f59e0b',
  },
  {
    title: '홈페이지 제작',
    description:
      '브랜딩 목적 홍보 페이지부터 맞춤형 서비스까지 다양한 웹사이트를 제작합니다.',
    fields: ['기업 홍보', '전시/이벤트', '온라인 서비스'],
    icon: <FaGlobe />,
    accent: '#14b8a6',
  },
  {
    title: '비전 AI',
    description:
      '영상·이미지 데이터를 기반으로 객체 탐지, 행동 분석, 이상 징후 파악 등 다양한 상황 인식을 지원합니다. 의료 영상(CT, MRI, X-ray) 분석과 같은 특수 분야에도 적용할 수 있습니다.',
    fields: ['스마트 시티', '보안 모니터링', '의료 영상 분석', '제조·품질 검사'],
    icon: <FaEye />,
    accent: '#ec4899',
  },
];

const Technology = () => {
  usePageTitle('보유 기술');
  const [showContact, setShowContact] = useState(false);

  return (
    <section className={styles.technology}>
      <div className={styles.hero}>
        <p className={styles.eyebrow}>OUR TECHNOLOGY</p>
        <h2>플레이리턴즈가 다루는 기술</h2>
        <p className={styles.subtitle}>
          CCTV·안면인식·AR·AI·디지털트윈 — 다양한 분야의 기술을 융합해 문제를
          해결합니다.
        </p>
      </div>

      <div className={styles.grid}>
        {techList.map((tech, idx) => (
          <article
            key={idx}
            className={`${styles.card} reveal`}
            style={{ '--accent': tech.accent }}
          >
            <div className={styles.iconWrap}>
              <span className={styles.icon}>{tech.icon}</span>
            </div>
            <h3 className={styles.title}>{tech.title}</h3>
            <p className={styles.desc}>{tech.description}</p>
            <div className={styles.fieldsBlock}>
              <span className={styles.fieldsLabel}>적용 분야</span>
              <div className={styles.tags}>
                {tech.fields.map((f) => (
                  <span key={f} className={styles.tag}>
                    {f}
                  </span>
                ))}
              </div>
            </div>
          </article>
        ))}
      </div>

      <div className={`${styles.contactBox} reveal`}>
        <p>이런 기술을 활용한 프로젝트를 검토하고 계신가요?</p>
        <button
          type="button"
          onClick={() => setShowContact(true)}
          className={styles.contactBtn}
        >
          기술 상담 문의하기
        </button>
      </div>

      {showContact && <ContactModal onClose={() => setShowContact(false)} />}
    </section>
  );
};

export default Technology;
