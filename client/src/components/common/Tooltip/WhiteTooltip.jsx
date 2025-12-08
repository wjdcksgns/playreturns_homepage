import { styled } from '@mui/material/styles';
import Tooltip, { tooltipClasses } from '@mui/material/Tooltip';

const LightTooltip = styled(({ className, ...props }) => (
  <Tooltip {...props} classes={{ popper: className }} />
))(({ theme }) => ({
  [`& .${tooltipClasses.arrow}`]: {
    color: theme.palette.common.white,
  },
  [`& .${tooltipClasses.tooltip}`]: {
    backgroundColor: theme.palette.common.white,
    color: theme.palette.common.black,
    boxShadow: theme.shadows[3],
    fontSize: 14,
    borderRadius: 20,
    padding: '5px 13px 3px',
  },
}));

const WhiteTooltip = ({children, title}) => {
  return (
    <LightTooltip title={title} arrow>
      {children}
    </LightTooltip>
  )
}

export default WhiteTooltip;