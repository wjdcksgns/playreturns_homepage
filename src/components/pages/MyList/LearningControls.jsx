import { useState, useContext, useEffect } from 'react';
import { useSelector } from 'react-redux';
import { MyListContext } from '../../../context/MyList/MyListContext';

import styles from './LearningControls.module.css';

import Btn from '../../common/Buttons/Btn';
import AlertModal from '../../common/Modals/AlertModal';
import api from '../../../services/interceptor';
import ConfirmModal from '../../common/Modals/ConfirmModal';

const LearningControls = () => {
  const userId = useSelector((state) => state.user.userId);
  const {rows, setQueryFlag, setSelectionModel, setSelectedRows, isLearning, learningDuration, learningTotalTime, learningRemainTime} = useContext(MyListContext);
  const [noDatasetAlertOpen, setNoDatasetAlertOpen] = useState(false);
  const [beforeLearningConfirmOpen, setBeforeLearningConfirmOpen] = useState(false);
  const [cancelable, setCancelable] = useState(false);
  const [queryInterval, setQueryInterval] = useState();

  const checkBeforeLearning = () => {
    const noDatasetList = rows.filter(v => v.dataSet.percent < 100);

    noDatasetList.length > 0 ? setNoDatasetAlertOpen(true) : setBeforeLearningConfirmOpen(true);
  }
  
  const handleStartLearning = async () => {
    const queryInterval = setInterval(() => setQueryFlag(true), 1000);

    setBeforeLearningConfirmOpen(false);
    setSelectionModel(rows.map(v => v.id));
    setSelectedRows(rows);
    setQueryInterval(queryInterval);
    
    const data = {
      'userId': userId,
      'objects': rows
    }

    try {
      const response = await api.patch('/api/labeling-server/object/video/learning', data);

      if (response.data.status === 200) {
        setSelectionModel([]);
        setSelectedRows([]);
      }
    }
    catch (err) {
      console.log(err)
    }
  }

  const handleCancelLearning = async () => {
    const data = {
      'userId': userId,
      'objects': rows
    }

    setQueryFlag(true);
    
    try {
      const response = await api.patch('/api/labeling-server/object/video/stop-learning', data);
      console.log(response.data);
    }
    catch (err) {
      console.log(err)
    }
  }

  const handleAlertClose = (e, reason) => {
    if (reason && reason === 'backdropClick') return;
    setNoDatasetAlertOpen(false);
  }

  const handleConfirmClose = (e, reason) => {
    if (reason && reason === 'backdropClick') return;
    setBeforeLearningConfirmOpen(false);
  }

  useEffect(() => {
    if (isLearning && learningDuration > 10) setCancelable(true);
  }, [isLearning, learningDuration]);

  useEffect(() => {
    if (isLearning) clearInterval(queryInterval);
    return () => clearInterval(queryInterval);
  }, [isLearning, queryInterval]);

  const learningBtn = {
    text: isLearning ? '학습 취소' : '오브젝트 학습',
    handleClick: isLearning ? handleCancelLearning : checkBeforeLearning,
    disabled: isLearning ? !cancelable || !rows.length : !rows.length
  }

  return (
    <>
      <div className={styles.rowsControls}>
        <Btn onClick={learningBtn.handleClick} size="md" props={{disabled: learningBtn.disabled}}>{learningBtn.text}</Btn>
      </div>
      {!!rows.length && !isLearning && <div className={styles.estimatedInfo}>오브젝트 학습 예상 소요 시간 : {learningTotalTime}</div>}
      {!!rows.length && isLearning && <div className={styles.estimatedInfo}>오브젝트 학습 완료까지 남은 시간 : {learningRemainTime}</div>}
      <AlertModal title="데이터셋 구축 미완료" open={noDatasetAlertOpen} handleClose={handleAlertClose}>
        먼저 모든 오브젝트의 데이터셋이 구축되어야 합니다.<br />
        데이터셋 구축 완료 후 학습을 진행해주시기 바랍니다.
      </AlertModal>
      <ConfirmModal title="학습 취소 시 주의사항" open={beforeLearningConfirmOpen} handleOk={handleStartLearning} handleClose={handleConfirmClose}>
        학습을 취소할 경우 기존 학습된 리스트도 다시 학습해야 합니다.<br />
        계속 진행하시겠습니까?
      </ConfirmModal>
    </>
  )
}

export default LearningControls;