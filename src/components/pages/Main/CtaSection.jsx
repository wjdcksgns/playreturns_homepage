import { useState } from 'react';
import { Link } from 'react-router-dom';
import { useReveal } from '../../../common/hooks/useReveal';
import ContactModal from '../../common/Modals/ContactModal';
import styles from './CtaSection.module.css';

const CtaSection = () => {
  const ref = useReveal();
  const [showContact, setShowContact] = useState(false);

  return (
    <section ref={ref} className={`${styles.section} reveal`}>
      <div className={styles.box}>
        <div className={styles.bgPattern} aria-hidden="true" />
        <div className={styles.inner}>
          <p className={styles.eyebrow}>LET'S WORK TOGETHER</p>
          <h2 className={styles.title}>
            함께 만들고 싶은
            <br />
            프로젝트가 있으신가요?
          </h2>
          <p className={styles.subtitle}>
            아이디어 단계부터 운영까지 — 어느 단계든 편하게 문의 주세요.
            <br />
            보통 영업일 1~2일 내에 답변드립니다.
          </p>

          <div className={styles.actions}>
            <button
              type="button"
              className={styles.primaryBtn}
              onClick={() => setShowContact(true)}
            >
              프로젝트 문의하기
            </button>
            <Link to="/contact" className={styles.secondaryBtn}>
              연락처 정보 보기
            </Link>
          </div>
        </div>
      </div>

      {showContact && (
        <ContactModal onClose={() => setShowContact(false)} />
      )}
    </section>
  );
};

export default CtaSection;
