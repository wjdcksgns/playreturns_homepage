import { useRef, useState } from 'react';
import { VideoTabContext } from '../../../context/MetaAutoLearning/VideoTabContext';
import styles from './VideoTabContents.module.css';

import { useSelector } from 'react-redux';

import Slider from 'react-slick';
import 'slick-carousel/slick/slick.css';
import 'slick-carousel/slick/slick-theme.css';
import '../../../common/css/slick-custom.css';

import Btn from '../../common/Buttons/Btn';
import TooltipBtn from '../../common/Buttons/TooltipBtn';
import ConfirmModal from '../../common/Modals/ConfirmModal';
import AlertModal from '../../common/Modals/AlertModal';
import LearningCard from './LearningCard';
import VideoPlayer from './VideoPlayer';

import UploadIcon from '@mui/icons-material/Upload';
import AddIcon from '@mui/icons-material/Add';
import RemoveIcon from '@mui/icons-material/Remove';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';
import api from '../../../services/interceptor';

const VideoTabContents = () => {
  const userId = useSelector((state) => state.user.userId);
  const { windowWidth } = useSelector((state) => state.viewport);

  const fileRef = useRef(null);
  const sliderRef = useRef(null);

  const [currentStep, setCurrentStep] = useState(0);
  const [video, setVideo] = useState({
    source: '',
    isPaused: true,
    currentTime: 0,
  });
  const [cards, setCards] = useState([{
    id: 0,
    step: 0,
    color: '',
    canvasSize: { width: 0, height: 0 },
    rectangle: { startX: 0, startY: 0, endX: 0, endY: 0 },
    label: '',
    url: '',
    tags: [],
    current: true,
  }]);
  const [resetModalOpen, setResetModalOpen] = useState(false);
  const [durationModalOpen, setDurationModalOpen] = useState(false);
  const [submitModalOpen, setSubmitModalOpen] = useState(false);

  const maxDuration = 10;
  const sliderSettings = {
    dots: true,
    infinite: false,
    speed: 500,
    slidesToShow: 1,
    slidesToScroll: 1,
    arrows: true,
    beforeChange: (current, next) => {
      const newCards = [...cards];

      newCards.forEach((v, i) => v.current = i === next);
      setCards(newCards);
    },
    responsive: [
      {
        breakpoint: 768,
        settings: {
          arrows: false,
        }
      }
    ]
  };

  const handleChangeFile = () => {
    const file = fileRef.current.files[0];
    const url = URL.createObjectURL(file);

    setVideo({
      source: url,
      isPaused: true,
      currentTime: 0,
    });
    setCurrentStep(1);
  }
  const handleClickFile = () => fileRef.current.click();
  const handleCreateCard = () => {
    const newCardId = cards[cards.length - 1].id + 1;
    const newCard = { 
      id: newCardId,
      step: 0,
      color: '',
      canvasSize: { width: 0, height: 0 },
      rectangle: { startX: 0, startY: 0, endX: 0, endY: 0 },
      label: '',
      url: '',
      tags: [],
      current: false
    };

    setCards(prevCards => [...prevCards, newCard]);
  }
  const handleDeleteCard = () => {
    const currentCardIdx = sliderRef.current.innerSlider.state.currentSlide;
    const newCards = cards.filter((v, i) => i !== Number(currentCardIdx));

    setCards(newCards);
  }
  const handleResetModalOk = () => {
    fileRef.current.value = '';
    setCurrentStep(0);
    setVideo({
      source: '',
      isPaused: true,
      currentTime: 0,
    });
    setCards([{
      id: 0,
      step: 0,
      color: '',
      canvasSize: { width: 0, height: 0 },
      rectangle: { startX: 0, startY: 0, endX: 0, endY: 0 },
      label: '',
      url: '',
      tags: [],
      current: true,
    }]);
    handleResetModalClose();
    handleClickFile();
  }
  const handleResetModalClose = (e, reason) => {
    if (reason && reason === 'backdropClick') return;
    setResetModalOpen(false);
  }
  const handleDurationModalClose = (e, reason) => {
    if (reason && reason === 'backdropClick') return;
    fileRef.current.value = '';
    setCurrentStep(0);
    setVideo({
      source: '',
      isPaused: true,
      currentTime: 0,
    });
    setCards([{
      id: 0,
      step: 0,
      color: '',
      canvasSize: { width: 0, height: 0 },
      rectangle: { startX: 0, startY: 0, endX: 0, endY: 0 },
      label: '',
      url: '',
      tags: [],
      current: true,
    }]);
    setDurationModalOpen(false);
  }
  const handleSubmitModalClose = (e, reason) => {
    if (reason && reason === 'backdropClick') return;
    fileRef.current.value = '';
    setCurrentStep(0);
    setVideo({
      source: '',
      isPaused: true,
      currentTime: 0,
    });
    setCards([{
      id: 0,
      step: 0,
      color: '',
      canvasSize: { width: 0, height: 0 },
      rectangle: { startX: 0, startY: 0, endX: 0, endY: 0 },
      label: '',
      url: '',
      tags: [],
      current: true,
    }]);
    setSubmitModalOpen(false);
  }
  const handleSubmit = e => {
    e.preventDefault();
    const copiedData = [...cards];
    const data = copiedData.map(v => {
      const formData = new FormData();

      formData.append('userId', userId);
      formData.append('objName', v.label);
      formData.append('url', v.url);
      formData.append('tags', [v.label, ...v.tags].join());
      formData.append('canvasSize', JSON.stringify(v.canvasSize));
      formData.append('rectangle', JSON.stringify(v.rectangle));
      formData.append('color', v.color);
      formData.append('file', fileRef.current.files[0]);

      return formData;
    })

    data.forEach(async v => {
      try {
        const response = await api.post('/api/meta-auto-learning/object/video', v);
        if (response.data.status === 200) setSubmitModalOpen(true);
      }
      catch (err) {
        console.log(err);
      }
    })
  }

  return (
    <VideoTabContext.Provider value={{currentStep, setCurrentStep, video, setVideo, cards, setCards, setResetModalOpen, setDurationModalOpen, maxDuration}}>
      <div className={styles.tabBox}>
        <div className={styles.viewArea}>
          <input ref={fileRef} className={styles.inputFile} type="file" accept=".mp4" onChange={handleChangeFile} />
          {!video.source ? (
            <div className={styles.uploadFile}>
              <UploadIcon className={styles.uploadIcon} />
              <div className={styles.uploadInfo}>
                <ErrorOutlineIcon />
                <p>업로드할 영상의 길이는 {maxDuration}초 이하여야 합니다.</p>
              </div>
              <Btn onClick={handleClickFile} size="md" fullWidth={true}>영상 업로드</Btn>
            </div>
          ) : (
            <VideoPlayer />
          )}
        </div>
        <div className={styles.setArea}>
          {currentStep > 1 && (
          <div className={styles.cardControls}>
            <TooltipBtn title="삭제" props={{onClick: handleDeleteCard, disabled: cards.length === 1}}>
              <RemoveIcon />
            </TooltipBtn>
            <TooltipBtn title="추가" props={{onClick: handleCreateCard, disabled: cards.length === 10}}>
              <AddIcon />
            </TooltipBtn>
          </div>
          )}
          <div style={{padding: cards.length > 1 && windowWidth > 768 ? '0 30px' : '0'}}>
            <Slider {...sliderSettings} ref={sliderRef}>
              {cards.map((card) => <LearningCard key={card.id} id={card.id} />)}
            </Slider>
          </div>
          {cards.every(v => v.step > 2) && (<div className={styles.btnWrap}>
            <Btn onClick={handleSubmit} size="md">등록하기</Btn>
          </div>)}
        </div>
        <ConfirmModal title="새로운 영상 등록" open={resetModalOpen} handleOk={handleResetModalOk} handleClose={handleResetModalClose}>
          기존에 설정한 모든 데이터가 삭제됩니다.<br />계속 하시겠습니까?
        </ConfirmModal>
        <AlertModal title="업로드 오류" open={durationModalOpen} handleClose={handleDurationModalClose}>
          영상의 길이가 {maxDuration}초를 초과합니다.<br />{maxDuration}초 이하의 영상을 업로드 해주세요.
        </AlertModal>
        <AlertModal title="등록 완료" open={submitModalOpen} handleClose={handleSubmitModalClose}>
          설정값이 저장되었습니다.<br />목록 페이지에서 학습이 가능합니다.
        </AlertModal>
      </div>
    </VideoTabContext.Provider>
  )
}

export default VideoTabContents;