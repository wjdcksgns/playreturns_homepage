import styles from './ArticleBox.module.css';

const ArticleBox = ({children, size}) => {
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
  const className = `${styles.box} ${sizeClass}`;

  return <article className={className}>{children}</article>
}

export default ArticleBox;