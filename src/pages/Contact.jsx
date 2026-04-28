import { useState } from 'react';
import {
  FaMapMarkerAlt,
  FaPhoneAlt,
  FaEnvelope,
  FaRegCopy,
  FaCheck,
} from 'react-icons/fa';
import styles from './Contact.module.css';
import ContactModal from '../components/common/Modals/ContactModal';
import { usePageTitle } from '../common/hooks/usePageTitle';

const infoItems = [
  {
    icon: <FaMapMarkerAlt />,
    label: 'Address',
    value: '경기도 수원시 영통구 광교산로 154-42, 경기대학교 창업보육센터 408호',
    copy: '경기도 수원시 영통구 광교산로 154-42, 경기대학교 창업보육센터 408호',
    accent: '#6366f1',
  },
  {
    icon: <FaPhoneAlt />,
    label: 'Phone',
    value: '010-2868-0655',
    copy: '01028680655',
    href: 'tel:01028680655',
    accent: '#22c55e',
  },
  {
    icon: <FaEnvelope />,
    label: 'Email',
    value: 'playreturns2025@gmail.com',
    sub: 'diksik2001@gmail.com',
    copy: 'playreturns2025@gmail.com',
    href: 'mailto:playreturns2025@gmail.com',
    accent: '#ec4899',
  },
];

const Contact = () => {
  usePageTitle('Contact');
  const [showContact, setShowContact] = useState(false);
  const [copiedIdx, setCopiedIdx] = useState(null);

  const onCopy = async (text, idx) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedIdx(idx);
      setTimeout(() => setCopiedIdx(null), 1500);
    } catch (e) {
      // 클립보드 권한 없는 환경 무시
    }
  };

  return (
    <section className={styles.contact}>
      <div className={styles.hero}>
        <p className={styles.eyebrow}>CONTACT US</p>
        <h2>플레이리턴즈와 함께할 준비가 되셨나요?</h2>
        <p className={styles.subtitle}>
          프로젝트 문의·기술 상담·협업 제안 등 무엇이든 편하게 연락 주세요.
        </p>
      </div>

      <div className={`${styles.container} reveal`}>
        <div className={styles.mapWrapper}>
          <iframe
            title="company-location"
            src="https://www.google.com/maps?q=37.300586,127.038294&hl=ko&z=17&output=embed"
            width="100%"
            height="100%"
            style={{ border: 0 }}
            allowFullScreen=""
            loading="lazy"
          />
        </div>

        <div className={styles.infoCard}>
          <h3>플레이리턴즈</h3>
          <p className={styles.tagline}>See something, know something.</p>

          <ul className={styles.infoList}>
            {infoItems.map((item, idx) => (
              <li
                key={idx}
                className={styles.infoItem}
                style={{ '--accent': item.accent }}
              >
                <div className={styles.infoIcon}>{item.icon}</div>
                <div className={styles.infoBody}>
                  <span className={styles.infoLabel}>{item.label}</span>
                  {item.href ? (
                    <a className={styles.infoValue} href={item.href}>
                      {item.value}
                    </a>
                  ) : (
                    <span className={styles.infoValue}>{item.value}</span>
                  )}
                  {item.sub && (
                    <span className={styles.infoSub}>{item.sub}</span>
                  )}
                </div>
                <button
                  type="button"
                  className={styles.copyBtn}
                  onClick={() => onCopy(item.copy, idx)}
                  aria-label={`${item.label} 복사`}
                  title="복사"
                >
                  {copiedIdx === idx ? <FaCheck /> : <FaRegCopy />}
                </button>
              </li>
            ))}
          </ul>

          <button
            type="button"
            className={styles.cta}
            onClick={() => setShowContact(true)}
          >
            메일 문의하기
          </button>
        </div>
      </div>

      {showContact && <ContactModal onClose={() => setShowContact(false)} />}
    </section>
  );
};

export default Contact;
