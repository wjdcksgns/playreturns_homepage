import { createTheme, ThemeProvider } from '@mui/material/styles';
import { FormControl, InputLabel, MenuItem, Select } from "@mui/material";

const theme = createTheme({
  palette: {
    pink: {
      main: '#ff5798',
    },
  },
});

const SelectBox = ({label, props, options}) => {
  return (
    <ThemeProvider theme={theme}>
      <FormControl sx={{width: '100%', background: 'white'}} size="small" color="pink">
        <InputLabel id={label.id}>{label.title}</InputLabel>
        <Select
          labelId={label.id}
          label={label.title}
          {...props}
        >
          {options.map(option => <MenuItem key={option.id} value={option.value}>{option.title}</MenuItem>)}
        </Select>
      </FormControl>
    </ThemeProvider>
  )
}

export default SelectBox;