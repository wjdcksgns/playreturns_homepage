import styles from "./Technology.module.css";
import { FaVideo, FaUserCheck, FaGlasses, FaMobileAlt, FaBrain, FaCube, FaGlobe, FaEye } from "react-icons/fa";
import ContactModal from "../components/common/Modals/ContactModal"; // 경로 맞게 조정
import { useState } from "react";

const techList = [
  {
    title: "CCTV 활용 감지 기술",
    description: "CCTV와 센서를 결합하여 객체·이상행동·침입 등을 실시간 탐지합니다.",
    fields: "보안, 안전 관리, 스마트시티, 공장 모니터링",
    icon: <FaVideo />,
  },
  {
    title: "안면인식 기술",
    description: "딥러닝 기반 얼굴 인식 알고리즘으로 신원 확인, 출입 통제, 맞춤형 서비스를 제공합니다.",
    fields: "보안 출입 시스템, 근태 관리, 맞춤형 서비스",
    icon: <FaUserCheck />,
  },
  {
    title: "AR 활용 원격협업 기술",
    description: "AR 글래스와 태블릿을 활용해 원격에서도 동일한 공간을 공유하며 협업할 수 있습니다.",
    fields: "산업 현장, 교육/훈련, 원격 지원",
    icon: <FaGlasses />,
  },
  {
    title: "Android/iOS 어플리케이션 개발",
    description: "네이티브와 크로스플랫폼 기반으로 다양한 모바일 앱을 설계·개발합니다.",
    fields: "엔터테인먼트, 교육, 비즈니스, IoT 연동",
    icon: <FaMobileAlt />,
  },
  {
    title: "LLM & AI Agent 개발",
    description: "LLM과 AI 에이전트를 활용해 복잡한 문제 해결과 자동화된 의사결정을 지원합니다.",
    fields: "대화형 서비스, 업무 자동화, 지능형 데이터 분석",
    icon: <FaBrain />,
  },
  {
    title: "디지털트윈 개발",
    description: "현실 데이터를 실시간으로 반영하는 디지털 공간을 구축해 모니터링과 시뮬레이션이 가능합니다.",
    fields: "스마트 팩토리, 건설, 물류 관리",
    icon: <FaCube />,
  },
  {
    title: "홈페이지 제작",
    description: "브랜딩 목적 홍보 페이지부터 맞춤형 서비스까지 다양한 웹사이트를 제작합니다.",
    fields: "기업 홍보, 전시/이벤트, 온라인 서비스",
    icon: <FaGlobe />,
  },
  {
    title: "비전 AI",
    description: "영상·이미지 데이터를 기반으로 객체 탐지, 행동 분석, 이상 징후 파악 등 다양한 상황 인식을 지원합니다. 특히 의료 영상(CT, MRI, X- ray) 분석과 같은 특수 분야에도 적용할 수 있습니다.",
    fields: "스마트 시티, 보안 모니터링, 의료 영상 분석, 제조·품질 검사",
    icon: <FaEye />,
  },
];

function Technology() {
  const [showContact, setShowContact] = useState(false);
  return (
    <section className={styles.technology}>
      <div className={styles.hero}>
        <h2>Our Technology</h2>
        <p>플레이리턴즈가 보유한 핵심 기술과 적용 분야를 소개합니다.</p>
      </div>

      <div className={styles.grid}>
        {techList.map((tech, idx) => (
          <div key={idx} className={styles.card}>
            <div className={styles.icon}>{tech.icon}</div>
            <h3>{tech.title}</h3>
            <p className={styles.desc}>{tech.description}</p>
            <p className={styles.fields}>
              <strong>적용 분야</strong>
              <span className={styles.tags}>
                {tech.fields.split(",").map((field, i) => (
                  <span key={i} className={styles.tag}>{field.trim()}</span>
                ))}
              </span>
            </p>
          </div>
        ))}
      </div>

      {/* 문의하기 버튼 */}
      <div className={styles.contactBox}>
        <button onClick={() => setShowContact(true)} className={styles.contactBtn}>
          문의하기
        </button>
      </div>

      {showContact && <ContactModal onClose={() => setShowContact(false)} />}
    </section>
  );
}

export default Technology;
