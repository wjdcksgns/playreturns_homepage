import styles from './InfoInputBox.module.css';

const InfoInputBox = ({id, label, info, children}) => {
  return (
    <div className={styles.inputBox}>
      <label htmlFor={id} className={styles.label}>{label}</label>
      {children}
      <small className={styles.infoMsg} style={{color: info.color}}>{info.msg}</small>
    </div>
  )
}

export default InfoInputBox;