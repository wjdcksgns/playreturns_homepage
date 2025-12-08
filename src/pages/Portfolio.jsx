import styles from "./Portfolio.module.css";
import { useState } from "react";
import ContactModal from "../components/common/Modals/ContactModal"; // 경로 맞게 조정

const projects = [
  {
    title: "AI 안면인식 솔루션",
    description: `실시간 안면인식으로 방문 이력과 증상·처방 정보를 통합 관리하는 스마트 헬스케어/보안 솔루션입니다.

• 비마커 기반 실시간 얼굴 인식으로 자연스러운 식별
• 방문·증상·처방 이력 통합 조회 및 시각화(약사/관리자 화면)
• 폐쇄망(Local Network)·암호화 등 개인정보 보호 강화
• 데이터 기반 맞춤 상담/추천 등 서비스 확장 지원`,
    images: [
      "/assets/images/portfolio/face_1.png",
      "/assets/images/portfolio/face_2.png",
      "/assets/images/portfolio/face_3.png",
    ],
  },
  {
    title: "슈퍼리듬스타 (Android)",
    description: `귀여운 도트 캐릭터와 스토리 진행이 결합된 리듬 액션 게임. 다양한 음악 장르와 손맛 나는 타격감으로 색다른 재미를 제공합니다.

• 'ONE! TWO! THREE! GO!' 큐에 맞춘 타이밍 플레이
• 스토리 × 리듬의 색다른 진행
• 다채로운 음악 장르와 그래픽/사운드 이펙트
• 리더보드로 점수 경쟁 및 친구와 순위 비교`,
    images: [
      "/assets/images/portfolio/rhythm_1.jpg",
      "/assets/images/portfolio/rhythm_2.jpg",
      "/assets/images/portfolio/rhythm_3.png",
      "/assets/images/portfolio/rhythm_4.jpg",
      "/assets/images/portfolio/rhythm_5.jpg",
      "/assets/images/portfolio/rhythm_6.jpg",
      "/assets/images/portfolio/rhythm_7.jpg",
      "/assets/images/portfolio/rhythm_8.png",
      "/assets/images/portfolio/rhythm_9.jpg",
    ],
  },
  {
    title: "AR 원격협업 플랫폼",
    description: `AI와 IoT 기반의 AR 원격협업 플랫폼. 실시간 영상·음성 스트리밍과 2D/3D 모델링 시뮬레이션을 통해 현장 인력과 원격 전문가가 협력할 수 있도록 지원합니다.

• 다자간 비대면 원격 협업 지원 (WebRTC 기반 영상/음성 스트리밍)
• 2D/3D 모델링 및 시뮬레이션을 통한 직관적 협업 지원
• IoT 데이터 연동으로 실시간 현장 정보 제공
• 현장 인력과 전문가 간 음성·영상·AR 모델·파일 공유 가능`,
    images: [
      "/assets/images/portfolio/ar_1.png",
      "/assets/images/portfolio/ar_2.png",
      "/assets/images/portfolio/ar_3.png",
      "/assets/images/portfolio/ar_4.png",
    ],
  },
];



function Portfolio() {
  const [selectedProject, setSelectedProject] = useState(null);
  const [currentIndex, setCurrentIndex] = useState(0); // ✅ 현재 이미지 인덱스
  const [showContact, setShowContact] = useState(false);

  const handlePrev = () => {
    setCurrentIndex((prev) =>
      prev === 0 ? selectedProject.images.length - 1 : prev - 1
    );
  };

  const handleNext = () => {
    setCurrentIndex((prev) =>
      prev === selectedProject.images.length - 1 ? 0 : prev + 1
    );
  };

  return (
    <section className={styles.portfolio}>
      <div className={styles.hero}>
        <h2>Our Portfolio</h2>
        <p>플레이리턴즈의 대표 프로젝트를 소개합니다.</p>
      </div>

      <div className={styles.list}>
        {projects.map((proj, idx) => (
          <div key={idx} className={styles.card}>
            <div className={styles.imageWrapper}>
              <img src={proj.images[0]} alt={proj.title} />
              <button
                className={styles.moreBtn}
                onClick={() => {
                  setSelectedProject(proj);
                  setCurrentIndex(0); // ✅ 항상 첫 번째 이미지로 초기화
                }}              >
                더보기
              </button>
            </div>
            <div className={styles.textWrapper}>
              <h3>{proj.title}</h3>
              <p>{proj.description}</p>
            </div>
          </div>
        ))}
      </div>


      {/* 모달 */}
      {selectedProject && (
        <div
          className={styles.modalOverlay}
          onClick={() => setSelectedProject(null)}
        >
          <div
            className={styles.modal}
            onClick={(e) => e.stopPropagation()}
          >
            <h3>{selectedProject.title}</h3>

            <div className={styles.slider}>

              <button className={`${styles.navBtn} ${styles.left}`} onClick={handlePrev}>
                ◀
              </button>
              <img
                src={selectedProject.images[currentIndex]}
                alt={`${selectedProject.title} ${currentIndex}`}
                className={styles.slideImage}
              />
              <button className={`${styles.navBtn} ${styles.right}`} onClick={handleNext}>
                ▶
              </button>

            </div>

            <p className={styles.counter}>
              {currentIndex + 1} / {selectedProject.images.length}
            </p>

            <button
              className={styles.closeBtn}
              onClick={() => setSelectedProject(null)}
            >
              닫기
            </button>
          </div>
        </div>
      )}

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

export default Portfolio;
