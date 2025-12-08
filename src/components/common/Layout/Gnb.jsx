import { useSelector } from 'react-redux';

import styles from './Gnb.module.css';
import NavList from './NavList';
import NavLinkBtn from '../Buttons/NavLinkBtn';

const Gnb = () => {
  const userId = useSelector((state) => state.user.userId);
  const key = userId ? 'login' : 'logout';
  const gnbMenu = {
    login: [
      {
        id: 1,
        title: '데이터 등록',
        path: '/meta_auto_learning',
      },
      {
        id: 2,
        title: '학습 목록',
        path: '/my_list',
      },
      {
        id: 3,
        title: '학습 확인',
        path: '/detection',
      }
    ],
    logout: []
  }

  return (
    <NavList>
      {gnbMenu[key].map(gnb => {
        return (
          <li key={gnb.id} className={styles.navItem}>
            <NavLinkBtn path={gnb.path}>{gnb.title}</NavLinkBtn>
          </li>
        )
      })}
    </NavList>
  )
}

export default Gnb;