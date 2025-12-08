import { useSelector, useDispatch } from 'react-redux'
import { logout } from '../../../features/user/userSlice'

import styles from './Users.module.css';
import NavList from './NavList';
import SlideLinkBtn from '../Buttons/SlideLinkBtn';
import api from "../../../services/interceptor";
import NavLinkBtn from '../Buttons/NavLinkBtn';

const Users = () => {
  const userId = useSelector((state) => state.user.userId);
  const userName = useSelector((state) => state.user.userName);
  const key = userId ? 'login' : 'logout';
  const dispatch = useDispatch();

  //   const userMenu = {
  //     login: [
  //       {
  //         id: 1,
  //         title: '로그아웃',
  //         path: '/',
  //         onClick: async () => {
  //           try {
  //             const response = await api.delete('/api/user/logout');
  //             if (response.data.status === 200) dispatch(logout());
  //           }
  //           catch (err) {
  //             console.log(err);
  //           }
  //         }
  //       }
  //     ],
  //     logout: [
  //       {
  //         id: 1,
  //         title: '로그인',
  //         path: '/login',
  //       },
  //       {
  //         id: 2,
  //         title: '회원가입',
  //         path: '/join',
  //       }
  //     ]
  //   }

  //   return (
  //     <NavList>
  //       {userId && (
  //         <li className={styles.userProfile}>
  //           <NavLinkBtn path={'/my_page'}>{userName}</NavLinkBtn>님
  //         </li>
  //       )}
  //       {userMenu[key].map(user => {
  //         return (
  //           <li key={user.id} className={styles.navItem}>
  //             <SlideLinkBtn path={user.path} props={{ onClick: user.onClick }}>{user.title}</SlideLinkBtn>
  //           </li>
  //         )
  //       })}
  //     </NavList>
  //   )
}

export default Users;