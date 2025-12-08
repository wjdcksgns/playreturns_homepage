import { createTheme, ThemeProvider } from '@mui/material/styles';
import { TextField } from "@mui/material"

const theme = createTheme({
  palette: {
    pink: {
      main: '#ff5798',
    },
  },
});

/**
 * @param {object} props id, label, value, placeholder, onChange, onKeyup
 */
const OutlineInput = ({props}) => {
  return (
    <ThemeProvider theme={theme}>
      <TextField
        sx={{width: '100%', background: 'white'}}
        size="small"
        variant="outlined"
        color="pink"
        fullWidth
        {...props}
      />
    </ThemeProvider>
  )
}

export default OutlineInput