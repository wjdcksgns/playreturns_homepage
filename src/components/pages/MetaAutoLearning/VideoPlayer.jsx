import { useState, useContext, useCallback, useEffect, useRef } from 'react';
import { VideoTabContext } from '../../../context/MetaAutoLearning/VideoTabContext';

import styles from './VideoPlayer.module.css';
import RectangleCanvas from './RectangleCanvas';
import TooltipBtn from '../../common/Buttons/TooltipBtn';

import AddCircleIcon from '@mui/icons-material/AddCircle';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import StopIcon from '@mui/icons-material/Stop';
import PauseIcon from '@mui/icons-material/Pause';
import { touchDevice } from '../../../common/js/common';

const VideoPlayer = () => {
  const {currentStep, video, setVideo, cards, setResetModalOpen, setDurationModalOpen, maxDuration} = useContext(VideoTabContext);
  const videoRef = useRef(null);

  const [videoSize, setVideoSize] = useState({
    width: 0,
    height: 0
  })

  const updateVideo = useCallback(() => {
    setVideo(prev => ({
      ...prev,
      isPaused: videoRef.current.paused,
      currentTime: videoRef.current.currentTime
    }));
  }, [setVideo]);

  const handlePlayVideo = () => {
    if (videoRef.current.currentTime === videoRef.current.duration) videoRef.current.currentTime = 0;
    videoRef.current.play();

    updateVideo();
  };

  const handleStopVideo = useCallback(() => {
    videoRef.current.currentTime = 0;
    videoRef.current.pause();

    updateVideo();
  }, [updateVideo]);

  const handlePauseVideo = () => {
    videoRef.current.pause();

    updateVideo();
  };

  useEffect(() => {
    currentStep > 1 && handleStopVideo();
  }, [currentStep, handleStopVideo])

  const handleMetadata = e => {
    const { videoWidth, videoHeight, duration } = e.currentTarget;

    if (~~duration > maxDuration) {
      setDurationModalOpen(true);
    }
    else {
      const ratio = videoHeight / videoWidth;
      const width = videoRef.current.clientWidth;
      const height = ~~(width * ratio)
      
      setVideoSize({
        width: width,
        height: height
      })
    }
  }

  const handleResize = () => {
    const { clientWidth, clientHeight } = videoRef.current;

    setVideoSize({
      width: clientWidth,
      height: clientHeight
    })
  }

  useEffect(() => {
    if (videoRef) window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [videoRef])

  return (
    <>
      <div className={styles.videoBox}>
        <video ref={videoRef} className={styles.video} src={video.source} muted onLoadedMetadata={handleMetadata} />
        <div className={styles.fileCover} style={{boxShadow: touchDevice() && 'none'}}>
          {(videoSize.width && videoSize.height) && (cards.map(card => <RectangleCanvas key={card.id} id={card.id} size={videoSize} disabled={card.step < 1 || !card.color} />))}
          {!touchDevice() && (
            <button className={styles.btnFile} onClick={() => setResetModalOpen(true)}>
              <span className={styles.btnFileLabel}>Change File</span>
              <AddCircleIcon className={styles.btnFileIcon} />
            </button>
          )}
        </div>
      </div>
      <div className={styles.videoControls}>
        <TooltipBtn title="재생" props={{onClick: handlePlayVideo, disabled: cards.some(v => v.step > 0) || !video.isPaused}}>
          <PlayArrowIcon />
        </TooltipBtn>
        <TooltipBtn title="정지" props={{onClick: handleStopVideo, disabled: cards.some(v => v.step > 0) || (video.isPaused && video.currentTime === 0)}}>
          <StopIcon />
        </TooltipBtn>
        <TooltipBtn title="일시정지" props={{onClick: handlePauseVideo, disabled: cards.some(v => v.step > 0) || video.isPaused}}>
          <PauseIcon />
        </TooltipBtn>
      </div>
    </>
  )
}

export default VideoPlayer;