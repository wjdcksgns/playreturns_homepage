// 인터넷 익스플로러 확인
export const isIE = () => {
  const agent = navigator.userAgent.toLowerCase();
  
  return ((navigator.appName === 'Netscape') && (navigator.userAgent.search('Trident') !== -1)) || (agent.indexOf("msie") !== -1);
}

// 터치 디바이스 확인
export const touchDevice = () => ('ontouchstart' in document.documentElement);

// 화면 방향 확인
export const screenOrientation = {
  portrait: () => window.matchMedia('(orientation: portrait)').matches, // 세로모드
  landscape: () => window.matchMedia('(orientation: landscape)').matches, // 가로모드
}

// 날짜를 {yyyy-mm-dd hh:mm:ss}로 포맷
export const dateTimeFormat = (dateTime) => {
  const newDate = new Date(dateTime);
  const yyyy = newDate.getFullYear();
  const mm = newDate.getMonth() + 1;
  const dd = newDate.getDate();
  const h = newDate.getHours();
  const m = newDate.getMinutes();
  const s = newDate.getSeconds();
  const date = `${yyyy}-${twoDigitFormat(mm)}-${twoDigitFormat(dd)}`;
  const time = `${twoDigitFormat(h)}:${twoDigitFormat(m)}:${twoDigitFormat(s)}`;

  return `${date} ${time}`;
}

// 숫자 두자리수 포맷
export const twoDigitFormat = (num) => {
  const target = typeof(num) === 'string' ? num : String(num);
  
  return target.padStart(2, '0');
}

// 초를 {시h 분m 초s}로 포맷
export const secondsFormat = (seconds) => {
  const h = parseInt(seconds / 3600);
  const m = parseInt((seconds % 3600) / 60);
  const s = seconds % 60;
  const hour = h > 0 ? `${h}h ` : '';
  const min  = m > 0 ? `${m}m ` : '';
  const sec  = s > 0 ? `${s}s` : '0s';

  return `${hour}${min}${sec}`
}

export const sleep = (sec) => {
  const start = Date.now()
  let now = start;
  
  while (now - start < sec * 1000) {
    now = Date.now();
  }
}