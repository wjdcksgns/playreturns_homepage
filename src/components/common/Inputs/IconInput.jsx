import styles from './IconInput.module.css';

import { createTheme, ThemeProvider } from '@mui/material/styles';
import TextField from '@mui/material/TextField';

const theme = createTheme({
  palette: {
    pink: {
      main: '#ff5798',
    },
  },
});

const IconInput = ({input, props, icon, info}) => {
  return (
    <ThemeProvider theme={theme}>
      <div className={styles.inputBox}>
        <div className={styles.iconInput}>
          {icon}
          <TextField fullWidth id={input.id} type={input.type} label={input.label} value={input.value} placeholder={input.placeholder} {...props} variant="standard" color="pink" />
        </div>
        {info?.msg && <small className={styles.infoMsg} style={{color: info.color}}>{info.msg}</small>}
      </div>
    </ThemeProvider>
  )
}

export default IconInput;