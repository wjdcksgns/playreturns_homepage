import { useContext, useState } from 'react';
import { MyListContext } from '../../../context/MyList/MyListContext';

import styles from './DataTable.module.css';

import Table from '../../common/Table/Table';
import BlackTooltip from '../../common/Tooltip/BlackTooltip';
import IconButton from '@mui/material/IconButton';

import PanoramaFishEyeIcon from '@mui/icons-material/PanoramaFishEye';
import EditIcon from '@mui/icons-material/Edit';
import CloseIcon from '@mui/icons-material/Close';
import CircularProgressWithLabel from '../../common/Loading/circularProgressWithLabel';

import { secondsFormat } from '../../../common/js/common';

const columns = [
  { field: 'id', headerName: 'No', hide: true },
  { field: 'objName', headerName: '오브젝트 이름', width: 120, headerAlign: 'center' },
  { field: 'dateTime', headerName: '등록일시', width: 160, headerAlign: 'center', align: 'center' },
  {
    field: 'tags',
    headerName: '태그',
    width: 200,
    headerAlign: 'center',
    valueGetter: params => params.value.map(tag => `#${tag}`).join(' ')
  },
  { field: 'url', headerName: 'URL', width: 160, headerAlign: 'center' },
  { 
    field: 'dataSet',
    headerName: '데이터셋',
    width: 100,
    headerAlign: 'center',
    align: 'center',
    renderCell: params => {
      const { percent, totalTime } = params.value;

      switch(percent) {
        case 0:
          return (
            <div className={styles.datasetTime}>
              예상시간<br />{secondsFormat(totalTime)}
            </div>
          )
        case 100:
          return <PanoramaFishEyeIcon sx={{color: 'green'}} />
        default:
          return <CircularProgressWithLabel value={percent < 0 ? 0 : percent} />
      }
    },
  },
  { 
    field: 'learning',
    headerName: '학습',
    width: 100,
    headerAlign: 'center',
    align: 'center',
    renderCell: params => {
      return (
        !params.value ? 
        <CloseIcon sx={{color: 'red'}} /> : 
        <PanoramaFishEyeIcon sx={{color: 'green'}} />
      )
    },
  },
  { 
    field: 'progress',
    headerName: '학습률',
    width: 100,
    headerAlign: 'center',
    align: 'center',
    renderCell: params => {
      if (params.value === 0) {
        return '-';
      }
      else {
        return `${params.value}%`;
      }
    }
  },
  {
    field: 'edit',
    headerName: '수정',
    sortable: false,
    width: 70,
    headerAlign: 'center',
    align: 'center',
    renderCell: () => {
      const onClick = e => {
        e.stopPropagation();

        return alert('수정하기 기능 작업중입니다.');
      }
      return (
        <BlackTooltip title="수정하기">
          <IconButton onClick={onClick} >
            <EditIcon />
          </IconButton>
        </BlackTooltip>
      )
    }
  },
];

const DataTable = () => {
  const {selectionModel, setSelectionModel, searchFilter, setSelectedRows, rows, isLearning} = useContext(MyListContext);
  const [pageSize, setPageSize] = useState(10);

  const handleChangePageSize = newPageSize => setPageSize(newPageSize);
  const handleRowSelectable = params => params.row.dataSet.percent === 0 || params.row.dataSet.percent === 100;
  const handleChangeSelectionModel = id => {
    const setSelectedId = new Set(id);
    const getSelectedRow = rows.filter(row => setSelectedId.has(row.id));

    setSelectedRows(getSelectedRow);
    setSelectionModel(id);
  }

  return (
    <div className={styles.dataTable} style={{opacity: isLearning ? 0.25 : 1}}>
      <Table
        props={{
          columns: columns,
          rows: rows,
          getRowId: row => row.id,
          pageSize: pageSize,
          rowsPerPageOptions: [5, 10, 50],
          filterModel: {items: searchFilter},
          selectionModel: selectionModel,
          onPageSizeChange: handleChangePageSize,
          isRowSelectable: handleRowSelectable,
          onSelectionModelChange: handleChangeSelectionModel
        }}
      />
    </div>
  )
}

export default DataTable;