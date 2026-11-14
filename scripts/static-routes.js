// 빌드 후: App.jsx의 각 경로(/idlefishing-privacy 등)에 index.html 복사본을 만든다.
//
// GitHub Pages는 모르는 경로에 404 상태 코드 + 404.html(리다이렉트 트릭)을 준다.
// 브라우저에선 페이지가 뜨지만, 구글 OAuth 동의 화면·스토어 심사 봇처럼 상태 코드만 보는
// 검사기는 "페이지 없음"으로 판단한다(2026-11-14 확인: 모든 정책 페이지가 404).
// 경로마다 진짜 파일을 두면 GitHub Pages가 200으로 바로 서빙한다. 화면 동작은 그대로.
const fs = require('fs');
const path = require('path');

const root = path.join(__dirname, '..');
const app = fs.readFileSync(path.join(root, 'src', 'App.jsx'), 'utf8');
const build = path.join(root, 'build');
const index = path.join(build, 'index.html');

const routes = [...app.matchAll(/path="\/([a-z0-9-]+)"/g)].map((m) => m[1]);
for (const r of routes) {
  const dir = path.join(build, r);
  fs.mkdirSync(dir, { recursive: true });
  fs.copyFileSync(index, path.join(dir, 'index.html'));
}
console.log(`[static-routes] ${routes.length}개 경로에 index.html 생성`);
