import { useCallback, useEffect, useRef } from 'react';
import { useSelector } from 'react-redux';

const FullHeightSection = ({children, props}) => {
  const { windowHeight } = useSelector((state) => state.viewport);
  const sectionRef = useRef(null);

  const getSectionHeight = useCallback(() => windowHeight - document.getElementById('header').offsetHeight, [windowHeight]);
  const setSectionHeight = height => sectionRef.current.style.height = `${height}px`;
  
  const handleResize = useCallback(() => {
    if (sectionRef.current.clientHeight === getSectionHeight()) return;
    setSectionHeight(getSectionHeight());
  }, [getSectionHeight]);

  useEffect(() => {
    setSectionHeight(getSectionHeight());
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [getSectionHeight, handleResize]);

  return (
    <section ref={sectionRef} {...props}>
      <div className="wrap">
        {children}
      </div>
    </section>
  )
}

export default FullHeightSection;