import { useEffect } from 'react';
import { useNavigate } from "react-router-dom";
import { useSelector } from 'react-redux';

import ImgBgIntro from '../../assets/images/join/bg_intro.jpg';

import IntroSection from '../../components/common/Sections/IntroSection';
import JoinSection from '../../components/pages/Join/JoinSection';

const Join = () => {
  const userId = useSelector((state) => state.user.userId);
  const navigate = useNavigate();

  useEffect(() => {
    userId && navigate(-1);
  }, [userId, navigate])

  return (
    <main>
      <IntroSection bg={ImgBgIntro}>
        <h2>회원가입</h2>
      </IntroSection>
      <JoinSection />
    </main>
  )
}

export default Join;