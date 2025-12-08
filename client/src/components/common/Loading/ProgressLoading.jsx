import styles from './ProgressLoading.module.css'

import { createTheme, ThemeProvider } from '@mui/material/styles';

import CircularProgress from '@mui/material/CircularProgress';
import Typography from '@mui/material/Typography';
import Box from '@mui/material/Box';

const theme = createTheme({
  palette: {
    pink: {
      main: '#ff5798',
    },
  },
});

const ProgressLoading = (props) => {
  return (
    <ThemeProvider theme={theme}>
      <div>
        <Box className={styles.loadingBox}>
          <CircularProgress variant="determinate" color="pink" className={styles.loadingCircle} {...props} />
          <Box className={styles.progressBox}>
            <Typography variant="caption" component="div" color="text.secondary" className={styles.progressTxt}>
              {`${Math.round(props.value)}%`}
            </Typography>
          </Box>
        </Box>
        <p className={styles.loadingTxt}>
          {props.children}
        </p>
      </div>
    </ThemeProvider>
  );
}

export default ProgressLoading;