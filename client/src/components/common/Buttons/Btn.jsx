import styles from './Btn.module.css';

const Btn = ({children, onClick, size, fullWidth, props}) => {
  let sizeClass;
  switch (size) {
    case 'md':
      sizeClass = styles.md;
      break;
    case 'sm':
      sizeClass = styles.sm;
      break;
    default:
      sizeClass = '';
      break;
  }
  const className = `${styles.btn} ${sizeClass} ${fullWidth ? styles.fullWidth : ''}`;
  
  return <button className={className} onClick={onClick} {...props}>{children}</button>
}

export default Btn;