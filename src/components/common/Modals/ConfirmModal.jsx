import { Children, isValidElement } from 'react';
import Btn from '../Buttons/Btn';

import Dialog from '@mui/material/Dialog';
import DialogActions from '@mui/material/DialogActions';
import DialogContent from '@mui/material/DialogContent';
import DialogContentText from '@mui/material/DialogContentText';
import DialogTitle from '@mui/material/DialogTitle';

const ConfirmModal = ({children, title, open, handleOk, handleClose, btnTxt}) => {
  const childArr = Children.toArray(children);
  const isElement = isValidElement(childArr[0]);

  return (
    <Dialog open={open} onClose={handleClose}>
      <DialogTitle>{title}</DialogTitle>
      <DialogContent>
        {isElement ? children : <DialogContentText>{children}</DialogContentText>}
      </DialogContent>
      <DialogActions sx={{padding: '16px 24px'}}>
        <Btn onClick={handleOk} size="sm">{btnTxt ? btnTxt.ok : '네'}</Btn>
        <Btn onClick={handleClose} size="sm">{btnTxt ? btnTxt.close : '아니오'}</Btn>
      </DialogActions>
    </Dialog>
  )
}

export default ConfirmModal;