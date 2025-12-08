import styles from './NavList.module.css';

const NavList = ({children}) => {
  return (
    <nav>
      <ul className={styles.navList}>
        {children}
      </ul>
    </nav>
  )
}

export default NavList;