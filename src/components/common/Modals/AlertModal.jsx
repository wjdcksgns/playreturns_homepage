import Btn from '../Buttons/Btn';

import Dialog from '@mui/material/Dialog';
import DialogActions from '@mui/material/DialogActions';
import DialogContent from '@mui/material/DialogContent';
import DialogContentText from '@mui/material/DialogContentText';
import DialogTitle from '@mui/material/DialogTitle';

const AlertModal = ({children, title, open, handleClose}) => {
  return (
    <Dialog open={open} onClose={handleClose}>
      <DialogTitle>{title}</DialogTitle>
      <DialogContent>
        <DialogContentText>{children}</DialogContentText>
      </DialogContent>
      <DialogActions sx={{padding: '16px 24px'}}>
        <Btn onClick={handleClose} size="sm">확인</Btn>
      </DialogActions>
    </Dialog>
  )
}

export default AlertModal;