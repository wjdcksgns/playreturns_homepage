import * as React from 'react';
import { alpha, styled } from '@mui/material/styles';
import Switch from '@mui/material/Switch';

const PinkSwitch = styled(Switch)(({ theme }) => ({
  '& .MuiSwitch-switchBase.Mui-checked': {
    color: '#ff5798',
    '&:hover': {
      backgroundColor: alpha('#ff5798', theme.palette.action.hoverOpacity),
    },
  },
  '& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': {
    backgroundColor: '#ff5798',
  },
}));

const SwitchBtn = ({Checked, onChange}) => {
  return <PinkSwitch defaultChecked={Checked} onChange={onChange} />;
}

export default SwitchBtn