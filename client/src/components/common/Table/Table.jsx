import { DataGrid, koKR } from '@mui/x-data-grid';

const Table = ({props}) => {
  return (
    <DataGrid
      localeText={koKR.components.MuiDataGrid.defaultProps.localeText}
      sx={{background: 'white'}}
      autoHeight
      disableColumnFilter
      disableColumnSelector
      disableDensitySelector
      checkboxSelection
      {...props}
    />
  )
}

export default Table;