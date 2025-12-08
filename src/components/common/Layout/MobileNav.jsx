import { useMemo, useEffect, useRef, useState } from 'react';
import styles from './MobileNav.module.css';

const MobileNav = ({children}) => {
  const navRef = useRef();
  const [openNav, setOpenNav] = useState(false);
  const handleCloseNav = e => e.target.tagName === 'A' && setOpenNav(false)

  const toggleNav = useMemo(() => {
    return {
      open(backdrop, navContainer) {
        document.body.style.overflow = 'hidden';
        navRef.current.style.display = 'block';
        
        setTimeout(() => {
          backdrop.style.opacity = 1;
          navContainer.style.right = 0;
        }, 0);
      },
      close(backdrop, navContainer) {
        const navContainerWidth = window.getComputedStyle(navContainer).getPropertyValue('width');

        backdrop.style.opacity = 0;
        navContainer.style.right = `-${navContainerWidth}`;
        
        setTimeout(() => {
          document.body.style.overflow = '';
          navRef.current.style.display = 'none';
        }, 300);
      }
    }
  }, []);

  useEffect(() => {
    const backdrop = navRef.current.firstChild;
    const navContainer = navRef.current.lastChild;
    const key = openNav ? 'open' : 'close';

    toggleNav[key](backdrop, navContainer);
  }, [openNav, toggleNav]);

  return (
    <div className={`${styles.mobileNav} ${openNav ? styles.open : ''}`}>
      <button className={styles.btnNav} onClick={() => setOpenNav(!openNav)}>
        <i></i>
        <i></i>
        <i></i>
      </button>
      <div ref={navRef} className={styles.navWrap}>
        <div className={styles.backdrop}></div>
        <div className={styles.navContainer} onClick={handleCloseNav}>
          {children}
        </div>
      </div>
    </div>
  )
}

export default MobileNav;