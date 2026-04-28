import { useEffect } from 'react';

const SITE = '플레이리턴즈 PlayReturns';

/**
 * 브라우저 탭 제목을 페이지별로 자동 설정.
 * 사용:
 *   usePageTitle('회사 소개');  →  "회사 소개 | 플레이리턴즈 PlayReturns"
 *   usePageTitle();             →  "플레이리턴즈 PlayReturns"
 */
export const usePageTitle = (title) => {
  useEffect(() => {
    document.title = title ? `${title} | ${SITE}` : SITE;
  }, [title]);
};
