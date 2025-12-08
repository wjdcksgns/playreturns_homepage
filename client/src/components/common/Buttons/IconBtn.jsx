import styles from './IconBtn.module.css';

const IconBtn = ({children, props}) => {
  return <button className={styles.btnIcon} {...props}>{children}</button>
}

export default IconBtn;