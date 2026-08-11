// 서울대 멘토-멘티 매칭 API 주소.
//
// 2026-08: 사무실 서버(FastAPI on Docker) → Google Cloud Run 으로 이전했다.
// 예전 주소는 https://api.playreturns.co.kr/snu 였는데, 그 도메인은 사주 앱
// Firebase 프로젝트에 묶여 있어서 사주 쪽 배포에 휘말려 끊긴 적이 있다.
// 그래서 매칭 시스템은 전용 도메인으로 분리했다.
//
// 백엔드는 루트(/analyze)와 /snu/analyze 를 모두 받도록 되어 있어서
// 이 값 끝에 /snu 를 붙여도 동작한다.
export const SNU_API_BASE_URL = 'https://snu.playreturns.co.kr';
