import { useContext } from 'react';
import { useSelector } from 'react-redux';
import { MyListContext } from '../../../context/MyList/MyListContext';

import styles from './DatasetControls.module.css';

import Btn from '../../common/Buttons/Btn';
import DeleteIcon from '@mui/icons-material/Delete';
import api from '../../../services/interceptor';

const DatasetControls = () => {
  const userId = useSelector((state) => state.user.userId);
  const {setQueryFlag, selectedRows, setSelectedRows, selectionModel, setSelectionModel, setModalOpen} = useContext(MyListContext);
  const handleModalOpen = () => setModalOpen(true);

  const getData = () => {
    return {
      'userId': userId,
      'objects': selectedRows
    }
  }

  const handleDataset = async () => {
    setSelectedRows([]);
    setSelectionModel([]);
    setQueryFlag(true);
    
    try {
      const response = await api.patch('/api/labeling-server/object/video/dataset', getData())
      response.data.status !== 200 && console.log(response.data)
    }
    catch (err) {
      console.log(err);
    }
  }

  return (
    <div className={styles.rowsControls}>
      <Btn onClick={handleModalOpen} size="sm" props={{disabled: !selectionModel.length}}><DeleteIcon /></Btn>
      <div className={styles.controlBox}>
        <Btn onClick={handleDataset} size="sm" props={{disabled: !selectedRows.length || selectedRows.some(v => v.dataSet.percent !== 0)}}>데이터셋 구축</Btn>
      </div>
    </div>
  )
}

export default DatasetControls;