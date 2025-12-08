import styles from './TabMenu.module.css';

const TabMenu = ({children}) => {
  return (
    <ul className={styles.tabMenu}>
      {children}
    </ul>
  )
}

export default TabMenu;