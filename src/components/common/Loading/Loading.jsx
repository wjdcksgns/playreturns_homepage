import styles from './Loading.module.css';

import { createTheme, ThemeProvider } from '@mui/material/styles';
import CircularProgress from '@mui/material/CircularProgress';

const theme = createTheme({
  palette: {
    pink: {
      main: '#ff5798',
    },
  },
});

const Loading = ({children}) => {
  return (
    <ThemeProvider theme={theme}>
      <div className={styles.loadingBox}>
        <CircularProgress className={styles.loadingIcon} color="pink" />
        {children && (
          <p className={styles.loadingTxt}>
            {children}
          </p>
        )}
      </div>
    </ThemeProvider>
  )
}

export default Loading;