import styles from "./History.module.css";

const historyData = [
    {
        year: "2025",
        events: [
            "공사현장 시공 품질관리 어플리케이션 개발 (Android)",
            "카메라 영상과 UI 합성 안드로이드 기능 개발 (Android)",
        ],
    },
    {
        year: "2024",
        events: [
            "Magic Leap 2 기반 원격협업 플랫폼 개발 (AR, Android)",
            "낭산 김준연 선생 기념관 체험 콘텐츠 개발 (AR, WebGL)",
            "피코(VR)와 HW 연동 동영상 스트리밍 콘텐츠 개발 (VR, Android)",
        ],
    },
    {
        year: "2023",
        events: [
            "슈퍼리듬스타 모바일 게임 출시 (Android)",
            "CCTV 연계한 AI 기반 경계 시스템 개발 (AR, PC)",
            "태전그룹과 AI 기반 안면인식 기술을 활용한 고객 관리 PoC 수행 (현 개발팀 참여)"
        ],
    },
    {
        year: "2022",
        events: ["Nreal Light 기반 원격협업 플랫폼 개발 (AR, Android)",
            "삼성SDC와 원격협업 플랫폼 PoC 수행 (현 개발팀 참여)"
        ],
    },
    {
        year: "2021",
        events: [
            "홀로렌즈2 기반 원격협업 플랫폼 개발 (AR, UWP)",
            "CCTV 연계 모니터링 시스템 개발 (AR, PC)",
        ],
    },
    {
        year: "2020",
        events: ["박물관/식물원 체험 콘텐츠 개발 (VR)"],
    },
    {
        year: "2019",
        events: ["구청 CCTV 모니터링 디지털트윈 개발 (PC)"],
    },
];

function History() {
    return (
        <section className={styles.history}>
            <div className={styles.hero}>
                <h2>Our History</h2>
                <p>플레이리턴즈의 성장 발자취를 소개합니다.</p>
            </div>

            <div className={styles.timeline}>
                {historyData.map((item, idx) => (
                    <div key={idx} className={styles.timelineItem}>
                        <div className={styles.timelineYear}>{item.year}</div>
                        <ul className={styles.timelineEvents}>
                            {item.events.map((event, i) => (
                                <li key={i}>{event}</li>
                            ))}
                        </ul>
                    </div>
                ))}
            </div>
        </section>
    );
}

export default History;
