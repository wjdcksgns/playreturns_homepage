import { Container } from '@mui/system';
import styles from './IntroSection.module.css';

const IntroSection = ({children, bg}) => {

  return (
    <section className={styles.section} style={{background: `url(${bg}) no-repeat center/cover`}}>
      <div className="wrap">
        <Container>
          <div className={styles.contents}>
            <div className={styles.title}>
              {children}
            </div>
          </div>
        </Container>
      </div>
    </section>
  )
}

export default IntroSection;