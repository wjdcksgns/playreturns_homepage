import styles from './TooltipBtn.module.css';
import WhiteTooltip from '../Tooltip/WhiteTooltip';

const TooltipBtn = ({children, title, props}) => {
  return (
    <>
      {props && props.disabled ? (
        <button className={styles.btnIcon} {...props}>
          {children}
        </button>
      ) : (
        <WhiteTooltip title={title}>
          <button className={styles.btnIcon} {...props}>
            {children}
          </button>
        </WhiteTooltip>
      )}
    </>
  )
}

export default TooltipBtn;