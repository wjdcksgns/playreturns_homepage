import { useContext } from 'react';
import { MyListContext } from '../../../context/MyList/MyListContext';
import { sleep } from '../../../common/js/common';

import ConfirmModal from '../../common/Modals/ConfirmModal';
import api from '../../../services/interceptor';

const ConfirmDeleteModal = () => {
  const {setQueryFlag, selectedRows, setRows, modalOpen, setModalOpen} = useContext(MyListContext);

  const handleModalOk = () => {
    selectedRows.forEach(async v => {
      try {
        const response = await api.delete(`/api/meta-auto-learning/object/video/${v.id}`);

        setQueryFlag(true)
        if (response.data.status === 200) {
          setRows(prevRows => prevRows.filter(row => !selectedRows.includes(row)));
        }
        handleModalClose();
      }
      catch (err) {
        console.log(err);
      }
      
      sleep(0.5);
    })
  }

  const handleModalClose = (e, reason) => {
    if (reason && reason === 'backdropClick') return;
    setModalOpen(false);
  }

  return (
    <ConfirmModal title="데이터 삭제" open={modalOpen} handleOk={handleModalOk} handleClose={handleModalClose}>
      선택된 모든 데이터가 삭제됩니다.<br />계속 하시겠습니까?
    </ConfirmModal>
  )
}

export default ConfirmDeleteModal;