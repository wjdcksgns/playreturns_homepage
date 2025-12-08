import styles from './LineBtn.module.css';

const LineBtn = ({children, onClick, active, props}) => {
  return (
    <button className={`${styles.btnLine} ${active ? styles.active : ''}`} onClick={onClick} {...props}>{children}</button>
  )
}

export default LineBtn;