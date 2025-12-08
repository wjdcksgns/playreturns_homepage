import { Link } from 'react-router-dom';
import styles from './Btn.module.css';

const LinkBtn = ({children, path, size, border, fullWidth, props}) => {
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
  const className = `${styles.btn} ${sizeClass} ${border ? styles.border : ''} ${fullWidth ? styles.fullWidth : ''}`;

  return <Link to={path} className={className} {...props}>{children}</Link>
}

export default LinkBtn;