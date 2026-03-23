// import { useEffect, useState } from 'react';
// import styles from './RemapSlider.module.css';

// const images = [
//     '/images/remap/remap_1.png',
//     '/images/remap/remap_2.png',
//     '/images/remap/remap_3.png',
//     '/images/remap/remap_4.png',
//     '/images/remap/remap_5.png',
// ];

// const RemapSlider = () => {
//     const [index, setIndex] = useState(0);

//     useEffect(() => {
//         const timer = setInterval(() => {
//             setIndex((prev) => (prev + 1) % images.length);
//         }, 2000);

//         return () => clearInterval(timer);
//     }, []);

//     return (
//         <div className={styles.slider}>
//             <img src={images[index]} alt="리맵 소개" />
//             <p className={styles.caption}>
//                 새롭게 개발한 리맵 솔루션으로 공간을 직관적으로 재구성했습니다.
//             </p>
//         </div>
//     );
// };

// export default RemapSlider;
