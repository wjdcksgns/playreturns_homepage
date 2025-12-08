import styles from './TooltipBadgeBtn.module.css';

import CloseIcon from '@mui/icons-material/Close';
import BlackTooltip from '../Tooltip/BlackTooltip';

const TooltipBadgeBtn = ({children, title, props, deleteBadge}) => {
  return (
    <div className={styles.tooltipBadge}>
      <BlackTooltip title={title}>
        <button className={styles.btnTooltipBadge} {...props}>{children}</button>
      </BlackTooltip>
      <button className={styles.btnDeleteBadge} onClick={deleteBadge}>
        <CloseIcon />
      </button>
    </div>
  )
}

export default TooltipBadgeBtn;