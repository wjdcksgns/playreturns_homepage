import { useState } from 'react';
import { useNavigate } from "react-router-dom";

import axios from "axios";

import { useDispatch } from 'react-redux';
import { login } from '../../../features/user/userSlice';

import styles from './LoginContents.module.css';

import ArticleBox from '../../common/Boxes/ArticleBox';
import IconInput from '../../common/Inputs/IconInput';
import Btn from '../../common/Buttons/Btn';
import LinkBtn from '../../common/Buttons/LinkBtn';

import EmailIcon from '@mui/icons-material/Email';
import LockIcon from '@mui/icons-material/Lock';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';

const LoginContents = () => {
  const navigate = useNavigate();
  const dispatch = useDispatch();
  const [formValue, setFormValue] = useState({
    userEmail: '',
    userPw: ''
  })
  const [showErrMsg, setShowErrMsg] = useState(false);
  const inputs = [
    {
      id: "userEmail",
      type: "text",
      label: "EMAIL",
      placeholder: "이메일을 입력해주세요.",
      value: formValue.userEmail,
      onChange: (e) => {
        const value = e.currentTarget.value;
        setFormValue((prevFormValue)=> ({
          ...prevFormValue,
          userEmail: value
        }))
      },
      icon: <EmailIcon />
    },
    {
      id: "userPw",
      type: "password",
      label: "PASSWORD",
      placeholder: "비밀번호를 입력해주세요.",
      value: formValue.userPw,
      onChange: (e) => {
        const value = e.currentTarget.value;
        setFormValue((prevFormValue)=> ({
          ...prevFormValue,
          userPw: value
        }))
      },
      icon: <LockIcon />
    }
  ]

  const handleLogin = {
    success(user) {
      setShowErrMsg(false);
      dispatch(login(user));
      navigate('/');
    },
    error(e) {
      setFormValue({
        userEmail: '',
        userPw: ''
      });
      setShowErrMsg(true);
      e.target[0].focus();
    }
  }

  const handleSubmit = (e) => {
    e.preventDefault();
    axios.post('/api/user/login', formValue)
      .then(res => {
        const success = res.data.status === 200;
        console.log(res)

        if (success) {
          handleLogin.success(res.data.user);
        }
        else {
          handleLogin.error(e);
        }
      }).catch(err => console.log(err));
  }

  return (
    <ArticleBox size="sm">
      <form onSubmit={handleSubmit}>
        <div className={styles.inputGroup}>
          {inputs.map(input => <IconInput key={input.id} input={input} props={{onChange: input.onChange}} icon={input.icon} />)}
          {showErrMsg && (
            <div className={styles.errorMsg}>
              <ErrorOutlineIcon />
              <small>이메일 또는 비밀번호가 일치하지 않습니다.</small>
            </div>
          )}
        </div>
        <Btn size="md" fullWidth={true} props={{disabled: !formValue.userEmail || !formValue.userPw}}>로그인</Btn>
      </form>
      <LinkBtn path="/join" size="md" fullWidth={true}>회원가입</LinkBtn>
    </ArticleBox>
  )
}

export default LoginContents;