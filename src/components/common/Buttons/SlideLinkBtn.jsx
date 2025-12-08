import { Link } from 'react-router-dom';
import styles from './SlideLinkBtn.module.css';

const SlideLinkBtn = ({children, path, props}) => {
  return (
    <Link to={path} className={styles.btnSlideLink} {...props}>
      <span>{children}</span>
    </Link>
  )
}

export default SlideLinkBtn;