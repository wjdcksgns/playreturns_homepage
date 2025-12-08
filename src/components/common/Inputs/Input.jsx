import styles from './Input.module.css';

const Input = ({type, id, value, placeholder, maxLength, onChange}) => {
  return (
    <input type={type} id={id} name={id} value={value} placeholder={placeholder} maxLength={maxLength} onChange={onChange} className={styles.input} />
  )
}

export default Input;