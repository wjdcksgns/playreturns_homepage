import styles from './Tags.module.css';

const Tags = ({type, id, value, placeholder, maxLength, onChange, onKeyDown, tags}) => {
  return (
    <div className={styles.tags}>
      {tags.map((tag, idx) => <span key={idx} className={styles.tag}>#{tag}</span>)}
      <input type={type} id={id} name={id} value={value} placeholder={placeholder} maxLength={maxLength} className={styles.input} onChange={onChange} onKeyDown={onKeyDown} />
    </div>
  )
}

export default Tags;