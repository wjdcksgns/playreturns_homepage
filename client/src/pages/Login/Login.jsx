import { useEffect } from 'react';
import { useNavigate } from "react-router-dom";
import { useSelector } from 'react-redux';

import ImgBgIntro from '../../assets/images/login/bg_intro.jpg';

import IntroSection from '../../components/common/Sections/IntroSection';
import LoginSection from '../../components/pages/Login/LoginSection';

const Login = () => {
  const userId = useSelector((state) => state.user.userId);
  const navigate = useNavigate();

  useEffect(() => {
    userId && navigate(-1);
  }, [userId, navigate])
  
  return (
    <main>
      <IntroSection bg={ImgBgIntro}>
        <h2>로그인</h2>
      </IntroSection>
      <LoginSection />
    </main>
  )
}

export default Login;