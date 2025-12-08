import styles from "./About.module.css";

function About() {
  return (
    <section className={styles.about}>
      {/* 비전 */}
      <div className={styles.block}>
        <h2 className={styles.title}>Our Vision</h2>
        <p className={styles.vision}>“SEE SOMETHING, KNOW SOMETHING”</p>
        <p className={styles.text}>
          플레이리턴즈는 세상을 더 깊이 보고, 더 정확히 이해하는 기술을 만듭니다.<br></br>
          우리는 단순한 콘텐츠 제작을 넘어 현실과 디지털을 연결하고 경험을
          가치로 바꾸는 혁신을 지향합니다.
        </p>
      </div>

      {/* 우리가 하는 일 */}
      <div className={styles.block}>
        <h2 className={styles.title}>What We Do</h2>
        <ul className={styles.services}>
          <li>Unity 엔진 기반 콘텐츠 개발</li>
          <li>AR/VR/메타버스 콘텐츠 개발 (Android / iOS / PC 지원)</li>
          <li>AI 기능 융합 및 지능형 시스템 구현</li>
          <li>CCTV · 디지털트윈 · 실시간 협업 시스템 개발</li>
          <li>Unity Asset 기반 UI/모델링 + 맞춤 제작 지원</li>
        </ul>
      </div>

      {/* 우리의 약속 */}
      <div className={styles.block}>
        <h2 className={styles.title}>Our Promise</h2>
        <p className={styles.text}>
          플레이리턴즈는 단순히 결과물을 만드는 것이 아니라,<br></br>
          고객의 아이디어가 실제로 구현되고 성장할 수 있도록 돕는
          파트너가 되겠습니다.
        </p>
      </div>
    </section>
  );
}

export default About;
