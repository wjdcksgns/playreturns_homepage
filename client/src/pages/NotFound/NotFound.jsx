import styles from './NotFound.module.css';

import Container from "../../components/common/Layout/Container";
import MinHeightSection from "../../components/common/Sections/MinHeightSection"

const NotFound = () => {
  return (
    <main>
      <MinHeightSection className={styles.notFoundSection}>
        <div className={styles.space}>
          <div className={styles.stars}></div>
          <div className={styles.stars}></div>
          <div className={styles.stars}></div>
          <div className={styles.stars}></div>
          <div className={styles.stars}></div>
          <div className={styles.stars}></div>
          <div className={styles.msgBox}>
            <h2 className={styles.errorCode}>404</h2>
            <div className={styles.errorMsg}>
              <strong>
                죄송합니다. <br />
                페이지를 찾을 수 없습니다!
              </strong>
              <p>
                방문하시려는 페이지의 주소가 잘못 입력되었거나,<br />
                페이지의 주소가 변경 혹은 삭제되어 요청하신 페이지를 찾을 수 없습니다.<br />
                입력하신 주소가 정확한지 다시 한번 확인해 주시기 바랍니다.
              </p>
            </div>
          </div>
        </div>
        <Container>
        </Container>
      </MinHeightSection>
    </main>
  )
}

export default NotFound;