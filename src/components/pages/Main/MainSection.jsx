import styles from './MainSection.module.css';
import MainContents from './MainContents';

const MainSection = () => {
  return (
    <div className={styles.mainWrap}>
      <div className={styles.bg}>
        <video
          src={`${process.env.PUBLIC_URL || ''}/videos/bg_main.mp4`}
          muted
          autoPlay
          loop
          playsInline
        />
        <div className={styles.overlay} />
      </div>
      <div className={styles.section}>
        <MainContents />
      </div>
    </div>
  );
};

export default MainSection;
