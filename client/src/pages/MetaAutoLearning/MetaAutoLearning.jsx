import { useEffect } from 'react';
import { useNavigate } from "react-router-dom";
import { useSelector } from 'react-redux';

import ImgBgIntro from '../../assets/images/meta_auto_learning/bg_intro.jpg';
import IntroSection from "../../components/common/Sections/IntroSection";
import MetaAutoLearningSection from "../../components/pages/MetaAutoLearning/MetaAutoLearningSection";

const MetaAutoLearning = () => {
  const userId = useSelector((state) => state.user.userId);
  const navigate = useNavigate();

  useEffect(() => {
    !userId && navigate('/login');
  }, [userId, navigate])
  
  return (
    <main>
      <IntroSection bg={ImgBgIntro}>
        <h2>데이터 등록</h2>
        <small>동영상과 이미지를 활용해서 나만의 학습 데이터를 등록할 수 있습니다</small>
      </IntroSection>
      <MetaAutoLearningSection />
    </main>
  )
}

export default MetaAutoLearning;