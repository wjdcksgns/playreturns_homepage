import styles from './MainContents.module.css';

const MainContents = () => {
  return (
    <div className={styles.contents}>
      <div className={styles.slogan}>
        <p>"See something, know something"<br></br><br></br>리맵(RE:MAP)을 출시했습니다!<br></br>구글스토어에서 지금 다운로드 해보세요~</p>
      </div>
      <svg className={styles.title}>
        <defs>
          <linearGradient id="logo-gradient" x1="50%" y1="0%" x2="75%" y2="100%" >
            <stop offset="0%" stopColor="#FF5798">
              <animate attributeName="stop-color" values="#CE93d8; #B39DDB; #81D4FA; #A5D6A7; #E6EE9C; #FFAB91; #EF9A9A; #FF5798; #CE93d8" dur="6s" repeatCount="indefinite"></animate>
            </stop>
            <stop offset="100%" stopColor="#FF5798">
              <animate attributeName="stop-color" values="#B39DDB; #81D4FA; #A5D6A7; #E6EE9C; #FFAB91; #EF9A9A; #FF5798; #CE93d8; #B39DDB" dur="6s" repeatCount="indefinite"></animate>
            </stop>
          </linearGradient>
        </defs>
        <text x="50%" y="40%" textAnchor="middle">
          PlayReturns
        </text>
        <text x="50%" y="85%" textAnchor="middle">
          플레이리턴즈
        </text>
      </svg>
    </div>
  )
}
// // 최초 버전
// const MainContents = () => {
//   return (
//     <div className={styles.contents}>
//       <div className={styles.slogan}>
//         <p>"See something, know something"<br></br><br></br>플레이리턴즈는 AR/VR, AI, 디지털트윈을 융합해<br></br>모바일·웹 애플리케이션과 솔루션을 개발하는 기업입니다.</p>
//       </div>
//       <svg className={styles.title}>
//         <defs>
//           <linearGradient id="logo-gradient" x1="50%" y1="0%" x2="75%" y2="100%" >
//             <stop offset="0%" stopColor="#FF5798">
//               <animate attributeName="stop-color" values="#CE93d8; #B39DDB; #81D4FA; #A5D6A7; #E6EE9C; #FFAB91; #EF9A9A; #FF5798; #CE93d8" dur="6s" repeatCount="indefinite"></animate>
//             </stop>
//             <stop offset="100%" stopColor="#FF5798">
//               <animate attributeName="stop-color" values="#B39DDB; #81D4FA; #A5D6A7; #E6EE9C; #FFAB91; #EF9A9A; #FF5798; #CE93d8; #B39DDB" dur="6s" repeatCount="indefinite"></animate>
//             </stop>
//           </linearGradient>
//         </defs>
//         <text x="50%" y="40%" textAnchor="middle">
//           PlayReturns
//         </text>
//         <text x="50%" y="85%" textAnchor="middle">
//           플레이리턴즈
//         </text>
//       </svg>
//     </div>
//   )
// }

export default MainContents;