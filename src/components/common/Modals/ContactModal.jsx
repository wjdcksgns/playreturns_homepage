import emailjs from '@emailjs/browser';
import { useEffect, useState } from 'react';
import { FaUser, FaEnvelope, FaCommentDots, FaCheckCircle } from 'react-icons/fa';
import styles from './ContactModal.module.css';

const ContactModal = ({ onClose }) => {
  const [form, setForm] = useState({ name: '', email: '', message: '' });
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState(null); // 'success' | 'error' | null
  const [errorMsg, setErrorMsg] = useState('');

  useEffect(() => {
    const onKey = (e) => e.key === 'Escape' && onClose();
    document.addEventListener('keydown', onKey);
    document.body.style.overflow = 'hidden';
    return () => {
      document.removeEventListener('keydown', onKey);
      document.body.style.overflow = '';
    };
  }, [onClose]);

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!form.name || !form.email || !form.message) {
      setStatus('error');
      setErrorMsg('모든 항목을 입력해주세요.');
      return;
    }

    setLoading(true);
    setStatus(null);

    try {
      const result = await emailjs.send(
        'service_0w5wr3d',
        'template_5h0pwio',
        {
          ...form,
          time: new Date().toLocaleString('ko-KR'),
        },
        'jI1V7H7gx8L7NVV8w'
      );

      if (result.text === 'OK') {
        setStatus('success');
        setTimeout(() => onClose(), 1800);
      }
    } catch (error) {
      console.error('메일 전송 오류:', error);
      setStatus('error');
      setErrorMsg('메일 전송 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.');
    } finally {
      setLoading(false);
    }
  };

  if (status === 'success') {
    return (
      <div className={styles.overlay} onClick={onClose}>
        <div
          className={`${styles.modal} ${styles.successModal}`}
          onClick={(e) => e.stopPropagation()}
        >
          <div className={styles.successIcon}>
            <FaCheckCircle />
          </div>
          <h3 className={styles.successTitle}>문의가 전송되었습니다</h3>
          <p className={styles.successText}>
            담당자가 확인 후 빠르게 회신드리겠습니다.
            <br />
            감사합니다 :)
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className={styles.overlay} onClick={onClose}>
      <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
        <button
          type="button"
          className={styles.closeBtn}
          onClick={onClose}
          aria-label="닫기"
        >
          ×
        </button>

        <div className={styles.header}>
          <p className={styles.eyebrow}>CONTACT</p>
          <h3 className={styles.title}>프로젝트 문의하기</h3>
          <p className={styles.subtitle}>
            궁금하신 점이나 협업 제안이 있다면 편하게 남겨주세요.
          </p>
        </div>

        <form onSubmit={handleSubmit} className={styles.form} noValidate>
          <div className={styles.field}>
            <label htmlFor="cm-name" className={styles.label}>
              이름
            </label>
            <div className={styles.inputWrap}>
              <FaUser className={styles.inputIcon} />
              <input
                id="cm-name"
                type="text"
                name="name"
                placeholder="홍길동"
                value={form.name}
                onChange={handleChange}
                autoComplete="name"
                required
              />
            </div>
          </div>

          <div className={styles.field}>
            <label htmlFor="cm-email" className={styles.label}>
              이메일
            </label>
            <div className={styles.inputWrap}>
              <FaEnvelope className={styles.inputIcon} />
              <input
                id="cm-email"
                type="email"
                name="email"
                placeholder="example@email.com"
                value={form.email}
                onChange={handleChange}
                autoComplete="email"
                required
              />
            </div>
          </div>

          <div className={styles.field}>
            <label htmlFor="cm-message" className={styles.label}>
              문의 내용
            </label>
            <div className={styles.inputWrap}>
              <FaCommentDots
                className={`${styles.inputIcon} ${styles.textareaIcon}`}
              />
              <textarea
                id="cm-message"
                name="message"
                placeholder="프로젝트 개요·일정·예산·연락 가능 시간 등 자유롭게 적어주세요."
                rows="5"
                value={form.message}
                onChange={handleChange}
                required
              />
            </div>
          </div>

          {status === 'error' && (
            <p className={styles.errorMsg}>{errorMsg}</p>
          )}

          <div className={styles.actions}>
            <button
              type="button"
              onClick={onClose}
              className={styles.cancelBtn}
              disabled={loading}
            >
              취소
            </button>
            <button
              type="submit"
              className={styles.submitBtn}
              disabled={loading}
            >
              {loading ? (
                <>
                  <span className={styles.spinner} />
                  전송 중...
                </>
              ) : (
                '문의 보내기'
              )}
            </button>
          </div>

          <p className={styles.privacyNote}>
            전송된 정보는 문의 응대 목적으로만 사용되며, 별도 보관되지 않습니다.
          </p>
        </form>
      </div>
    </div>
  );
};

export default ContactModal;
