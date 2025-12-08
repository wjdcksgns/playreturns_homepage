import { styled } from '@mui/material/styles';
import Tooltip, { tooltipClasses } from '@mui/material/Tooltip';

const DarkTooltip = styled(({ className, ...props }) => (
  <Tooltip {...props} classes={{ popper: className }} />
))(({ theme }) => ({
  [`& .${tooltipClasses.arrow}`]: {
    color: 'rgba(0, 0, 0, .7)',
  },
  [`& .${tooltipClasses.tooltip}`]: {
    backgroundColor: 'rgba(0, 0, 0, .7)',
    color: theme.palette.common.white,
    boxShadow: theme.shadows[3],
    fontSize: 14,
    borderRadius: 20,
    padding: '5px 13px 3px',
  },
}));

const BlackTooltip = ({children, title}) => {
  return (
    <DarkTooltip title={title} arrow>
      {children}
    </DarkTooltip>
  )
}

export default BlackTooltip;