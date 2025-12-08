import { useSelector, useDispatch } from 'react-redux';
import { useEffect, useState } from 'react';

import styles from './MyPage.module.css';
import ImgBgIntro from '../../assets/images/my_page/bg_intro.jpg';

import ArticleBox from '../../components/common/Boxes/ArticleBox';
import Container from '../../components/common/Layout/Container';
import IntroSection from "../../components/common/Sections/IntroSection";
import MinHeightSection from '../../components/common/Sections/MinHeightSection';
import Btn from '../../components/common/Buttons/Btn';
import ConfirmModal from '../../components/common/Modals/ConfirmModal';
import AlertModal from '../../components/common/Modals/AlertModal';
import IconInput from '../../components/common/Inputs/IconInput';

import PersonIcon from '@mui/icons-material/Person';
import ModelTrainingIcon from '@mui/icons-material/ModelTraining';
import CalendarViewWeekIcon from '@mui/icons-material/CalendarViewWeek';
import LockIcon from '@mui/icons-material/Lock';

import { update } from '../../features/user/userSlice';
import api from '../../services/interceptor';

const MyPage = () => {
  const dispatch = useDispatch();
  const [userData, setUserData] = useState({ name: '', email: '', epoch: '', batch: '' });
  const [editData, setEditData] = useState({ name: '', epoch: '', batch: '' });
  const [changeData, setChangeData] = useState({ currentPw: '', newPw: '', confirmPw: '' });
  const [editModalOpen, setEditModalOpen] = useState(false);
  const [changeModalOpen, setChangeModalOpen] = useState(false);
  const [successAlertOpen, setSuccessAlertOpen] = useState(false);
  const [successType, setSuccessType] = useState('');
  const [errMsg, setErrMsg] = useState({ edit: '', change: '' });

  const { windowWidth } = useSelector((state) => state.viewport);
  const userId = useSelector((state) => state.user.userId);

  const editInputs = [
    {
      id: "name",
      type: "text",
      label: "FULL NAME",
      placeholder: "한글 또는 영문 대/소문자 입력",
      value: editData.name,
      icon: <PersonIcon />
    },
    {
      id: "epoch",
      type: "number",
      label: "LEARNING EPOCHS",
      placeholder: "학습 횟수 입력 (20~500회)",
      value: editData.epoch,
      icon: <ModelTrainingIcon />
    },
    {
      id: "batch",
      type: "number",
      label: "BATCH SIZE",
      placeholder: "배치 사이즈 입력 (2~32)",
      value: editData.batch,
      icon: <CalendarViewWeekIcon />
    }
  ]
  const changeInputs = [
    {
      id: "currentPw",
      type: "password",
      label: "CURRENT PASSWORD",
      placeholder: "영문 대/소문자, 숫자, 특수문자 중 3가지 이상 조합 (8~16자)",
      value: changeData.currentPw,
      icon: <LockIcon />
    },
    {
      id: "newPw",
      type: "password",
      label: "NEW PASSWORD",
      placeholder: "영문 대/소문자, 숫자, 특수문자 중 3가지 이상 조합 (8~16자)",
      value: changeData.newPw,
      icon: <LockIcon />
    },
    {
      id: "confirmPw",
      type: "password",
      label: "CONFIRM PASSWORD",
      placeholder: "영문 대/소문자, 숫자, 특수문자 중 3가지 이상 조합 (8~16자)",
      value: changeData.confirmPw,
      icon: <LockIcon />
    }
  ]

  const setInputValue = {
    edit(id, value) {
      setEditData(prevFormValue => ({
        ...prevFormValue,
        [id]: value
      }))
    },
    change(id, value) {
      setChangeData(prevFormValue => ({
        ...prevFormValue,
        [id]: value
      }))
    }
  }

  const handleChange = (e, type) => {
    const { id, value } = e.currentTarget;

    setInputValue[type](id, value);
  }

  const validation = {
    edit: {
      name() {
        return /^[가-힣|a-z|A-Z|]+$/.test(editData.name);
      },
      epoch() {
        return /^[0-9]+$/.test(editData.epoch) && (editData.epoch >= 20 && editData.epoch <= 500);
      },
      batch() {
        return /^[0-9]+$/.test(editData.batch) && (editData.batch >= 2 && editData.batch <= 32);
      }
    },
    change: {
      async currentPw() {
        const response = await api.post('/api/user/check/password', { userId: userId, userPw: changeData.currentPw });
        return response.data;
      },
      newPw() {
        return changeData.currentPw !== changeData.newPw;
      },
      regExpPw() {
        return /^(?=.*[a-z|A-Z])(?=.*\d)(?=.*[{}[\]/?.,;:|)*~`!^\-_+<>@#$%&\\=('"])[A-Za-z\d{}[\]/?.,;:|)*~`!^\-_+<>@#$%&\\=('"]{8,16}$/g.test(changeData.newPw);
      },
      confirmPw() {
        return changeData.newPw === changeData.confirmPw;
      },
    }
  }

  const errorMessages = {
    name: '한글 또는 영문 대/소문자만 입력 가능합니다.',
    epoch: '20 이상 500 이하의 숫자만 입력 가능합니다.',
    batch: '2 이상 32 이하의 숫자만 입력 가능합니다.',
    currentPw: '비밀번호가 일치하지 않습니다.',
    newPw: '사용 중인 비밀번호와 다른 비밀번호를 입력해주세요.',
    regExpPw: '8~16자의 영문 대소문자, 숫자, 특수문자 중 3가지 조합으로 입력해주세요.',
    confirmPw: '새 비밀번호와 확인용 비밀번호가 일치하지 않습니다.'
  }

  const validator = async (type) => {
    for (const key in validation[type]) {
      const isPass = await validation[type][key]();
      
      if (!isPass) {
        const id = key === 'regExpPw' ? 'newPw' : key;

        setInputValue[type](id, '');
        setErrMsg((prevErrMsg) => ({...prevErrMsg, [type]: errorMessages[key]}));
        document.getElementById(id).focus();
        return false;
      }
    }
    setErrMsg((prevErrMsg) => ({...prevErrMsg, [type]: ''}));
    return true;
  }

  // 회원정보 수정
  const handleEditModalOk = async (e) => {
    const valid = await validator('edit');
    
    if (valid) {
      try {
        const response = await api.patch(`/api/user/${userId}`, editData);
        setEditModalOpen(false);
        
        if (response.data.status === 200) {
          dispatch(update(editData.name));
          setUserData(prevFormValue => ({
            ...prevFormValue,
            name: editData.name,
            epoch: editData.epoch,
            batch: editData.batch,
          }));
          setSuccessType(e.target.innerText);
          setSuccessAlertOpen(true);
        }
      }
      catch (err) {
        console.log(err);
      }
    }
  }

  const handleEditModalClose = (e, reason) => {
    if (reason && reason === 'backdropClick') return;
    setEditModalOpen(false);
    setErrMsg((prevErrMsg) => ({...prevErrMsg, edit: ''}));
    setEditData({ name: userData.name, epoch: userData.epoch, batch: userData.batch });
  }

  // 비밀번호 변경
  const handleChangeModalOk = async (e) => {
    const valid = await validator('change'); 

    if (valid) {
      try {
        const data = {
          userId: userId,
          currentPw: changeData.currentPw,
          newPw: changeData.newPw,
        }
        const response = await api.put('/api/user/password', data);
        setChangeModalOpen(false);
  
        if (response.data.status === 200) {
          setChangeData({ currentPw: '', newPw: '', confirmPw: '' });
          setSuccessType(e.target.innerText);
          setSuccessAlertOpen(true);
        }
      }
      catch (err) {
        console.log(err);
      }
    }
  }

  const handleChangeModalClose = (e, reason) => {
    if (reason && reason === 'backdropClick') return;
    setChangeModalOpen(false);
    setErrMsg((prevErrMsg) => ({...prevErrMsg, change: ''}));
  }

  const handleAlertClose = (e, reason) => {
    if (reason && reason === 'backdropClick') return;
    setSuccessAlertOpen(false);
  }

  // 회원정보 조회
  useEffect(() => {
    const getUserData = async () => {
      try {
        const response = await api.get(`/api/user/${userId}`);
        const { user } = response.data;
        console.log(user);
        setUserData(user);
        setEditData({ name: user.name, epoch: user.epoch, batch: user.batch });
      }
      catch (err) {
        console.log(err);
      }
    }
    getUserData();
  }, [userId])

  const getAlertMsg = () => {
    const data = successType === '수정' ? '회원정보' : '비밀번호';
    const title = `${data} ${successType} 성공`;
    const contents = `${data}가 성공적으로 ${successType}되었습니다.`;

    return {
      title: title,
      contents: contents
    }
  }

  return (
    <main>
      <IntroSection bg={ImgBgIntro}>
        <h2>마이페이지</h2>
      </IntroSection>
      <MinHeightSection>
        <Container>
          <ArticleBox size="sm">
            <div className={styles.titleBox}>
              <h2>회원 정보</h2>
            </div>
            <dl className={styles.userBox}>
              <dt>이름</dt>
              <dd>{userData.name}</dd>
              <dt>이메일</dt>
              <dd>{userData.email}</dd>
              <dt>학습 횟수</dt>
              <dd>{userData.epoch}회</dd>
              <dt>배치 사이즈</dt>
              <dd>{userData.batch}</dd>
            </dl>
            <div className={styles.btnBox}>
              <Btn onClick={() => setEditModalOpen(true)} size="md" fullWidth={windowWidth <= 375}>회원정보 수정</Btn>
              <Btn onClick={() => setChangeModalOpen(true)} size="md" fullWidth={windowWidth <= 375}>비밀번호 변경</Btn>
            </div>
          </ArticleBox>
        </Container>
        <ConfirmModal title="회원정보 수정" open={editModalOpen} handleOk={handleEditModalOk} handleClose={handleEditModalClose} btnTxt={{ok: '수정', close: '취소'}}>
          <div className={styles.inputGroup}>
            {editInputs.map(input => (
              <IconInput key={input.id} input={input} props={{onChange: e => handleChange(e, 'edit')}} icon={input.icon} />
            ))}
          </div>
          {errMsg.edit && <p className={styles.errMsg}>{errMsg.edit}</p>}
        </ConfirmModal>
        <ConfirmModal title="비밀번호 변경" open={changeModalOpen} handleOk={handleChangeModalOk} handleClose={handleChangeModalClose} btnTxt={{ok: '변경', close: '취소'}}>
          <div className={styles.inputGroup}>
            {changeInputs.map(input => (
              <IconInput key={input.id} input={input} props={{onChange: e => handleChange(e, 'change')}} icon={input.icon} />
            ))}
          </div>
          {errMsg.change && <p className={styles.errMsg}>{errMsg.change}</p>}
        </ConfirmModal>
        <AlertModal title={getAlertMsg().title} open={successAlertOpen} handleClose={handleAlertClose}>
          {getAlertMsg().contents}
        </AlertModal>
      </MinHeightSection>
    </main>
  )
}

export default MyPage;