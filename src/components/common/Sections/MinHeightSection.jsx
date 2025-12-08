import { useCallback, useEffect, useRef } from "react";
import { useSelector } from 'react-redux';

const MinHeightSection = ({children, className}) => {
  const { windowHeight } = useSelector((state) => state.viewport);
  const subSectionRef = useRef(null);
  const setSectionMinHeight = minHeight => subSectionRef.current.style.minHeight = `${minHeight}px`;

  const getSectionMinHeight = useCallback(() => {
    const footerHeight = document.getElementById('footer').getBoundingClientRect().height;
    let minHeight = windowHeight - footerHeight;

    if (subSectionRef.current.previousElementSibling) {
      const introSection      = subSectionRef.current.previousElementSibling;
      const introHeight       = introSection.offsetHeight;
      const introMarginBottom = window.getComputedStyle(introSection).marginBottom;
      const introOuterHeight  = introHeight + Number(introMarginBottom.split('px')[0]);
  
      minHeight = windowHeight - (footerHeight + introOuterHeight);
    }

    return minHeight;
  }, [windowHeight]);

  const handleResize = useCallback(() => {
    if (subSectionRef.current.style.minHeight === `${getSectionMinHeight()}px`) return;
    setSectionMinHeight(getSectionMinHeight());
  }, [getSectionMinHeight])

  useEffect(() => {
    setSectionMinHeight(getSectionMinHeight());
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [getSectionMinHeight, handleResize])

  return (
    <section ref={subSectionRef} className={`section ${className}`}>
      <div className="wrap">
        {children}
      </div>
    </section>
  )
}

export default MinHeightSection;