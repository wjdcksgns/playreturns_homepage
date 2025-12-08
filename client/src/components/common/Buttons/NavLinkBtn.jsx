import { useRef, useState } from 'react';
import { NavLink } from 'react-router-dom';
import { touchDevice } from '../../../common/js/common';
import styles from './NavLinkBtn.module.css';

const NavLinkBtn = ({children, path, props}) => {
  const tooltipRef = useRef(null);
  const [ showTooltip, setShowTooltip ] = useState(false);
  
  const getActiveClass = ({ isActive }) => styles.btnNavLink + (isActive ? ` ${styles.active}` : '');
  const hasTooltip = path === '/my_page' && !touchDevice();
  
  const tooltipEvents = {
    onMouseEnter() {
      setShowTooltip(hasTooltip);
    },
    onMouseMove({nativeEvent}) {
      if (!showTooltip) return;

      const { offsetX, offsetY } = nativeEvent;
      const gap = 15;
      const left = offsetX + gap;
      const top = offsetY + gap;
      
      tooltipRef.current.style.display = 'block';
      tooltipRef.current.style.left = `${left}px`;
      tooltipRef.current.style.top = `${top}px`;
    },
    onMouseLeave() {
      if (!showTooltip) return;
      tooltipRef.current.style.display = 'none';
      setShowTooltip(false);
    }
  }

  return (
    <NavLink to={path} className={getActiveClass} onMouseEnter {...props} {...tooltipEvents}>
      {children}
      {showTooltip && <div ref={tooltipRef} className={styles.tooltip}>마이페이지</div>}
    </NavLink>
  )
}

export default NavLinkBtn;