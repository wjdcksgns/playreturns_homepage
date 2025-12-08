import { useState, useEffect } from 'react';

import styles from './Footer.module.css';
import imgLogo from '../../../assets/images/logo_wh (2).png';

const Footer = () => {
  const [thisYear, setThisYear] = useState(0);

  const contactList = [
    {
      title: 'COMPANY',
      description: '플레이리턴즈',
    },
    {
      title: 'CEO',
      description: '정유진',
    },
    {
      title: 'ADDRESS',
      description: '경기도 수원시 영통구 광교산로 154-42, 경기대학교 창업보육센터 408호'
    }
  ]

  useEffect(() => {
    const now = new Date();
    
    setThisYear(now.getFullYear());
  }, []);

  return (
    <footer id="footer" className={styles.footer}>
      <div className="wrap">
        <div className="container">
          <div className={styles.contents}>
            <div className={styles.logo}>
              <h2>
                <img src={imgLogo} alt="logo" />
              </h2>
            </div>
            <div className={styles.info}>
              <address className={styles.contact}>
                {contactList.map(v => {
                  return (
                    <p key={v.title}>{v.title} : {v.description}</p>
                  )
                })}
              </address>
              <div className={styles.copyright}>
                <span>Copyright ⓒ {thisYear} PlayReturns All rights reserved.</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </footer>
  )
}

export default Footer;