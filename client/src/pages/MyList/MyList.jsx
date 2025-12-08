import { useEffect } from 'react';
import { useNavigate } from "react-router-dom";
import { useSelector } from 'react-redux';

import ImgBgIntro from '../../assets/images/my_list/bg_intro.jpg';
import IntroSection from '../../components/common/Sections/IntroSection';
import MyListSection from '../../components/pages/MyList/MyListSection';

const MyList = () => {
  const userId = useSelector((state) => state.user.userId);
  const navigate = useNavigate();

  useEffect(() => {
    !userId && navigate('/login');
  }, [userId, navigate])

  return (
    <main>
      <IntroSection bg={ImgBgIntro}>
        <h2>학습 목록</h2>
        <small>나만의 학습 데이터로 모델을 학습할 수 있습니다</small>
      </IntroSection>
      <MyListSection />
    </main>
  )
}

export default MyList;