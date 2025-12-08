import styles from './MainSection.module.css';

import FullHeightSection from '../../common/Sections/FullHeightSection';
import Container from '../../common/Layout/Container';
import MainContents from './MainContents';

const MainSection = () => {
  return (
    <div className={styles.mainWrap}>
      <div className={styles.bg}>
        <video src={`${process.env.PUBLIC_URL || ''}/videos/bg_main.mp4`} muted={true} autoPlay={true} loop={true}></video>
      </div>
      <FullHeightSection props={{className: styles.section}}>
        <Container>
          <MainContents />
        </Container>
      </FullHeightSection>
    </div>
  )
}

export default MainSection;