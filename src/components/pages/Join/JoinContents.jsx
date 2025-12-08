import { useState } from 'react';
import styles from './JoinContents.module.css';

import axios from 'axios';

import ArticleBox from '../../common/Boxes/ArticleBox';
import IconInput from '../../common/Inputs/IconInput';
import Btn from '../../common/Buttons/Btn';
import LinkBtn from '../../common/Buttons/LinkBtn';

import PersonIcon from '@mui/icons-material/Person';
import EmailIcon from '@mui/icons-material/Email';
import LockIcon from '@mui/icons-material/Lock';
import TaskAltIcon from '@mui/icons-material/TaskAlt';

const JoinContents = () => {
  const [formValue, setFormValue] = useState({ userFullName: '', userEmail: '', userPw: '', userConfirmPw: '' })
  const [errMsg, setErrMsg] = useState({ userFullName: '', userEmail: '', userPw: '', userConfirmPw: '' })
  const [isCompleted, setIsCompleted] = useState(false);
  const inputs = [
    {
      id: "userFullName",
      type: "text",
      label: "FULL NAME",
      placeholder: "한글 또는 영문 대/소문자 입력",
      value: formValue.userFullName,
      infoMsg: errMsg.userFullName,
      icon: <PersonIcon />
    },
    {
      id: "userEmail",
      type: "text",
      label: "EMAIL",
      placeholder: "이메일 주소 입력",
      value: formValue.userEmail,
      infoMsg: errMsg.userEmail,
      icon: <EmailIcon />
    },
    {
      id: "userPw",
      type: "password",
      label: "PASSWORD",
      placeholder: "영문 대/소문자, 숫자, 특수문자 중 3가지 이상 조합 (8~16자)",
      infoMsg: errMsg.userPw,
      icon: <LockIcon />
    },
    {
      id: "userConfirmPw",
      type: "password",
      label: "CONFIRM PASSWORD",
      placeholder: "영문 대/소문자, 숫자, 특수문자 중 3가지 이상 조합 (8~16자)",
      value: formValue.userConfirmPw,
      infoMsg: errMsg.userConfirmPw,
      icon: <LockIcon />
    }
  ]

  const validation = {
    userFullName: {
      regErr: '한글 또는 영문 대/소문자만 입력 가능합니다.',
      regExp(value) {
        return /^[가-힣|a-z|A-Z|]+$/.test(value);
      },
      getMsg(value) {
        return this.regExp(value) ? '' : this.regErr;
      }
    },
    userEmail: {
      regErr: '올바른 이메일 형식이 아닙니다.',
      matErr: '이미 가입된 이메일입니다',
      regExp(value) {
        return /^[0-9a-zA-Z]([-_.]?[0-9a-zA-Z])*@[0-9a-zA-Z]([-_.]?[0-9a-zA-Z])*\.[a-zA-Z]{2,3}$/i.test(value);
      },
      async getMsg(value) {
        if (this.regExp(value))
          return await axios.get(`/api/user/exists/${value}`)
            .then(res => res.data ? this.matErr : '')
            .catch(err => console.log(err));
        else
          return this.regErr
      },
    },
    userPw: {
      regErr: '8~16자의 영문 대소문자, 숫자, 특수문자 중 3가지 조합으로 입력해주세요.',
      regExp(value) {
        return /^(?=.*[a-z|A-Z])(?=.*\d)(?=.*[{}[\]/?.,;:|)*~`!^\-_+<>@#$%&\\=('"])[A-Za-z\d{}[\]/?.,;:|)*~`!^\-_+<>@#$%&\\=('"]{8,16}$/g.test(value);
      },
      getMsg(value) {
        return this.regExp(value) ? '' : this.regErr;
      }
    },
    userConfirmPw: {
      regErr: '8~16자의 영문 대소문자, 숫자, 특수문자 중 3가지 조합으로 입력해주세요.',
      matErr: '비밀번호가 일치하지 않습니다.',
      regExp(value) {
        return /^(?=.*[a-z|A-Z])(?=.*\d)(?=.*[{}[\]/?.,;:|)*~`!^\-_+<>@#$%&\\=('"])[A-Za-z\d{}[\]/?.,;:|)*~`!^\-_+<>@#$%&\\=('"]{8,16}$/g.test(value);
      },
      getMsg(value) {
        return this.regExp(value) ? (formValue.userPw === value ? '' : this.matErr) : this.regErr;
      }
    }
  }

  const validator = (id, value) => {
    id === 'userEmail' ? 
    validation[id].getMsg(value).then(v => setErrMsg(prevErrMsg => ({ ...prevErrMsg, [id]: v }))) :
    setErrMsg(prevErrMsg => ({ ...prevErrMsg, [id]: validation[id].getMsg(value) }));
  }

  const handleChange = (e) => {
    const { id, value } = e.currentTarget;

    setFormValue(prevFormValue => ({
      ...prevFormValue,
      [id]: value
    }));

    validator(id, value);
  }
  
  const handleJoin = {
    success() {
      setIsCompleted(true);
    },
    error() {
      setFormValue(prevFormValue => ({...prevFormValue, userEmail: ''}))
      setErrMsg(prevErrMsg => ({...prevErrMsg, userEmail: '이미 가입된 이메일입니다.'}))
    }
  }

  const handleSubmit = (e) => {
    e.preventDefault();

    axios
      .post('/api/user/join', {
        userName: formValue.userFullName,
        userEmail: formValue.userEmail,
        userPw: formValue.userPw,
      })
      .then(res => {
        const status = res.data.status === 200 ? 'success' : 'error';

        handleJoin[status]();
      })
      .catch(err => console.log(err));
  }

  return (
    <ArticleBox size="sm">
      {!isCompleted ? (
        <form onSubmit={handleSubmit}>
          <div className={styles.inputGroup}>
            {inputs.map(input => (
              <IconInput key={input.id} input={input} props={{onChange: handleChange}} icon={input.icon} info={{color: 'red', msg: input.infoMsg}} />
            ))}
          </div>
          <Btn size="md" fullWidth={true} props={{disabled: Object.values(formValue).some(v => !v) || Object.values(errMsg).some(v => v)}}>회원가입</Btn>
        </form>
      ) : (
        <div className={styles.completedInfo}>
          <div className={styles.completeBox}>
            <TaskAltIcon className={styles.checkIcon} />
            <h2>회원가입이 완료되었습니다</h2>
            <p>메타스페이스의 회원이 되신 것을 환영합니다 <br />로그인 후 서비스를 이용하실 수 있습니다</p>
          </div>
          <dl className={styles.userBox}>
            <dt>이름</dt>
            <dd>{formValue.userFullName}</dd>
            <dt>이메일</dt>
            <dd>{formValue.userEmail}</dd>
          </dl>
          <div className={styles.linkBox}>
            <LinkBtn path="/" size="md" border={true}>메인화면</LinkBtn>
            <LinkBtn path="/login" size="md">로그인</LinkBtn>
          </div>
        </div>
      )}
    </ArticleBox>
  )
}

export default JoinContents;