import { useEffect, useRef } from 'react';

/**
 * 스크롤 시 요소가 뷰포트에 들어오면 'is-visible' 클래스를 추가.
 * .reveal 클래스와 함께 쓰면 fade-in-up 애니메이션이 실행됨.
 *
 * 사용:
 *   const ref = useReveal();
 *   <section ref={ref} className="reveal">...</section>
 */
export const useReveal = (options = {}) => {
  const ref = useRef(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return undefined;

    // prefers-reduced-motion 사용자에겐 즉시 표시
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      el.classList.add('is-visible');
      return undefined;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add('is-visible');
            observer.unobserve(entry.target);
          }
        });
      },
      {
        threshold: options.threshold ?? 0.12,
        rootMargin: options.rootMargin ?? '0px 0px -60px 0px',
      }
    );

    observer.observe(el);
    return () => observer.disconnect();
  }, [options.threshold, options.rootMargin]);

  return ref;
};
