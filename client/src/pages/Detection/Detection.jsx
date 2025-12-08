import { useCallback, useEffect, useState, useRef } from 'react';
import { useSelector } from 'react-redux';
import styles from './Detection.module.css';

import ImgBgIntro from '../../assets/images/detection/bg_intro.jpg';
import IntroSection from "../../components/common/Sections/IntroSection";
import MinHeightSection from "../../components/common/Sections/MinHeightSection"
import Container from '../../components/common/Layout/Container';
import ArticleBox from '../../components/common/Boxes/ArticleBox';
import Btn from '../../components/common/Buttons/Btn';
import SwitchBtn from '../../components/common/Buttons/SwitchBtn';
import { FormControlLabel } from '@mui/material';
import api from '../../services/interceptor';

const Detection = () => {
  const videoRef = useRef(null);
  const cameraRef = useRef(null);
  const canvasRef = useRef(null);
  const contextRef = useRef(null);
  const userId = useSelector((state) => state.user.userId);
  
  const [hasCamera, setHasCamera] = useState(true);
  const [ctx, setCtx] = useState(null);
  const [rects, setRects] = useState([]);
  const [reverseMode, setReverseMode] = useState(false);

  const defaultWidth = 640;
  const defaultHeight = 480;
  const ratio = defaultHeight / defaultWidth;

  // 대비색 구하기
  const getComplementary = (color) => {
    const hexRgb = color.split('#')[1].match(/.{1,2}/g);
    const r = 255 - parseInt(hexRgb[0], 16);
    const g = 255 - parseInt(hexRgb[1], 16);
    const b = 255 - parseInt(hexRgb[2], 16);

    return `rgb(${r}, ${g}, ${b})`;
  };

  // Rectangle 체크
  const collides = (rects, x, y) => {
    let isCollision = false;

    rects.forEach((v, i) => {
      const left = v.x;
      const right = left + v.w;
      const top = v.y
      const bottom = top + v.h;

      if (right >= x && left <= x && bottom >= y && top <= y)
      isCollision = rects[i];
    })
    
    return isCollision;
  }

  // Rectangle 클릭
  const handleClick = (e) => {
    const { offsetX, offsetY } = e.nativeEvent;
    const rect = collides(rects, offsetX, offsetY);
    const hasProtocol = rect.url.split('://').length > 1;
    const url = hasProtocol ? rect.url : `https://${rect.url}`
    
    if (rect) window.open(url)
  }

  // Rectangle hover
  const handleMouseMove = (e) => {
    const { offsetX, offsetY } = e.nativeEvent;
    const rect = collides(rects, offsetX, offsetY);
    const title = rect ? `${rect.url} 바로가기` : '';
    const cursor = rect ? 'pointer' : 'default';

    canvasRef.current.title = title;
    canvasRef.current.style.cursor = cursor;
  }

  const handleResize = useCallback(() => {
    if (!hasCamera) return;
    const videoWidth = videoRef.current.clientWidth;
    const canvas = canvasRef.current;
    canvas.width = videoWidth;
    canvas.height = videoWidth * ratio;
  }, [hasCamera, ratio])

  // 캔버스 세팅
  useEffect(() => {
    if (!ctx) {
      const videoWidth = videoRef.current.clientWidth;
      const canvas = canvasRef.current;
      canvas.width = videoWidth;
      canvas.height = videoWidth * ratio;

      const context = canvas.getContext('2d');
      contextRef.current = context;
      setCtx(context);
    }
  }, [ctx, ratio])

  // 스트리밍 실행
  useEffect(() => {
    const getUserMedia = async () => {
      try {
        const video = videoRef.current;
        const constraints = {
          audio: false,
          video: {
            width: { ideal: 640 },
            height: { ideal: 480 },
            facingMode: "environment",
          }
        };
        const stream = await navigator.mediaDevices.getUserMedia(constraints);

        video.srcObject = stream;
        video.onloadedmetadata = () => video.play();
      }
      catch (err) {
        setHasCamera(false);
        console.log(`${err.name}: ${err.message}`);
      }
    };
    getUserMedia();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [handleResize])

  // detection 실행
  useEffect(() => {
    if (!hasCamera) return;
    const capture = setInterval(async () => {
      const videoWidth = videoRef.current.clientWidth;
      const videoHeight = videoWidth * ratio;

      const camera = cameraRef.current;
      camera.width = videoWidth;
      camera.height = videoHeight;
      camera.getContext("2d").drawImage(videoRef.current, 0, 0);

      const imgBlob = camera.toDataURL("image/png")

      const data = {
        userId: userId,
        imageBlob: imgBlob.split(',')[1]
      }

      try {
        const response = await api.post('/api/labeling-server/object/detection', data);

        ctx.lineWidth = 4;
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
        if (response.data.status === 200) {
          if (response.data.data.length > 0) {
            const ratioX = videoWidth / defaultWidth;
            const ratioY = videoHeight / defaultHeight;
            const objList = response.data.data;

            const rectList = objList.map(obj => {
              const label = obj.parentname;
              const color = obj.rec_color;
              const startX = ~~(obj.rec_start_x);
              const startY = ~~(obj.rec_start_y);
              const endX = ~~(obj.rec_end_x);
              const endY = ~~(obj.rec_end_y);
              const defaultX = startX * ratioX;
              const y = startY * ratioY;
              const w = (endX * ratioX) - defaultX;
              const h = (endY * ratioY) - y;
              const x = reverseMode ? videoWidth - w - defaultX : defaultX;
              const labelH = 34;
              const labelY = y - labelH;

              ctx.strokeStyle = color;
              ctx.strokeRect(x, y, w, h);

              ctx.fillStyle = color;
              ctx.fillRect(x - (ctx.lineWidth / 2), labelY, w + ctx.lineWidth, labelH);
        
              ctx.font = '16px sansSerif';
              ctx.fillStyle = getComplementary(color);
              ctx.fillText(label, x + 10, y - 10);

              return {
                label: label,
                url: obj.rec_url,
                color: color, 
                x: x, y: y, w: w, h: h,
              }
            })

            setRects(rectList);
          }
        }
        else {
          clearInterval(capture);
        }
      }
      catch(err) {
        console.log(`${err.name}: ${err.message}`);
        clearInterval(capture);
      }
    }, 1000);
    return () => clearInterval(capture);
  }, [ctx, hasCamera, ratio, reverseMode, userId]);
  
  return (
    <main>
      <IntroSection bg={ImgBgIntro}>
        <h2>학습 확인</h2>
        <small>카메라를 통해 학습된 모델의 성능을 확인할 수 있습니다</small>
      </IntroSection>
      <MinHeightSection className={styles.detectionSection}>
        <Container>
          {hasCamera ? (
            <ArticleBox size='md'>
              <div className={styles.cameraCtrl}>
                <FormControlLabel control={
                  <SwitchBtn Checked={false} onChange={e => setReverseMode(e.currentTarget.checked)} />
                } label="카메라 좌우반전" className={styles.switchLabel} />
              </div>
              <div className={styles.detectionBox}>
                <video ref={videoRef} autoPlay={true} className={`${reverseMode ? styles.reverseMode : ''}`}></video>
                <canvas className={styles.detectionCamera} ref={cameraRef}></canvas>
                <canvas className={styles.detectionCanvas} ref={canvasRef} onClick={handleClick} onMouseMove={handleMouseMove}></canvas>
              </div>
            </ArticleBox>
          ) : (
            <ArticleBox size='sm'>
              <div className={styles.infoBox}>
                <p>
                  카메라가 연결 되어있지 않습니다. <br />
                  카메라를 연결 후 아래 연결 버튼을 눌러주세요.
                </p>
                <Btn onClick={() => window.location.reload()} size='md'>연결</Btn>
              </div>
            </ArticleBox>
          )}
        </Container>
      </MinHeightSection>
    </main>
  )
}

export default Detection;