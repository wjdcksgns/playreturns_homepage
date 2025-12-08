import { useState } from 'react';
import { MyListContext } from '../../../context/MyList/MyListContext';

import { useSelector } from 'react-redux';
import { useQuery } from "react-query";

import ArticleBox from '../../common/Boxes/ArticleBox';
import SearchArea from './SearchArea';
import DatasetControls from './DatasetControls';
import DataTable from './DataTable';
import ConfirmDeleteModal from './ConfirmDeleteModal';
import Loading from '../../common/Loading/Loading';
import LearningControls from './LearningControls';
import ProgressLoading from '../../common/Loading/ProgressLoading';

import { dateTimeFormat, secondsFormat } from '../../../common/js/common';
import api from '../../../services/interceptor';

const MyListContents = () => {
  const [rows, setRows] = useState([]);
  const [searchFilter, setSearchFilter] = useState([]);
  const [selectionModel, setSelectionModel] = useState([]);
  const [selectedRows, setSelectedRows] = useState([]);
  const [modalOpen, setModalOpen] = useState(false);
  const [queryFlag, setQueryFlag] = useState(true);
  const [isLearning, setIsLearning] = useState(false);
  const [learningDuration, setLearningDuration] = useState(0);
  const [learningTotalTime, setLearningTotalTime] = useState(0);
  const [learningRemainTime, setLearningRemainTime] = useState(0);
  const [learningPercent, setLearningPercent] = useState(0);

  const userId = useSelector((state) => state.user.userId);

  const getNewRows = (rowsData) => {
    return rowsData.map(v => {
      return {
        id: v.uuid,
        objName: v.name,
        dateTime: dateTimeFormat(v.datetime),
        tags: v.tags.split(','),
        url: v.url,
        dataSet: {percent: v.dataset, totalTime: v.dataset_total},
        learning: v.learning,
        progress: v.progress,
      }
    })
  }

  const sumRowsData = (rowsData, key) => {
    return rowsData.reduce((acc, cur) => acc += ~~cur[key], 0)
  }

  const { isLoading, error, data } = useQuery([userId],
    async () => {
      try {
        const res = await api.get(`/api/meta-auto-learning/my-list/${userId}`);

        if (res.data.status === 200) {
          if (res.data.data) {
            const now = new Date();
            const flag = res.data.data.some(v => (v.dataset !== 0 && v.dataset !== 100) || v.learning_ready);
            const newRows = getNewRows(res.data.data);
            const learningTotal = sumRowsData(res.data.data, 'learning_total') + 17 + (8 * res.data.data.length);
            const learningTotalTime = secondsFormat(learningTotal);
            const learningTimer = sumRowsData(res.data.data, 'learning_timer') / res.data.data.length;
            const learningRemain = (learningTotal + learningTimer) - (now.getTime() / 1000);
            const learningRemainTime = secondsFormat(~~learningRemain);
            const learningDuration = learningTotal - ~~learningRemain;
            let learningPercent = 100 - ((learningRemain / learningTotal) * 100)
                learningPercent = learningPercent > 99 ? 99 : learningPercent;

            setQueryFlag(flag)
            setIsLearning(res.data.data.some(v => v.learning_ready))
            setRows([...newRows])
            setLearningTotalTime(learningTotalTime)
            setLearningRemainTime(learningRemainTime)
            setLearningPercent(learningPercent)
            setLearningDuration(learningDuration)

            return [...newRows]
          }
          else {
            setQueryFlag(false)
            setIsLearning(false)
            setRows([])
            setLearningTotalTime(0)
            setLearningRemainTime(0)
            setLearningPercent(0)

            return []
          }
        }
      }
      catch (err) {
        return console.log(err);
      }
    }, {
      enabled: !!userId && queryFlag,
      refetchInterval: 1000
    })

  if (error) {
    console.log(error)
  }

  return (
    <MyListContext.Provider value={{rows, setRows, setQueryFlag, searchFilter, setSearchFilter, selectionModel, setSelectionModel, selectedRows, setSelectedRows, modalOpen, setModalOpen, isLearning, setIsLearning, learningDuration, learningTotalTime, learningRemainTime}}>
      <ArticleBox size="md">
        <SearchArea />
        <DatasetControls />
        {isLoading && (
          <Loading>
            로딩중입니다.<br />
            잠시만 기다려주세요.
          </Loading>
        )}
        <div style={{position: 'relative'}}>
          {isLearning && (
            <div style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: '100%',
              zIndex: 100,
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
            }}>
              <ProgressLoading value={learningPercent}>
                오브젝트 학습 중입니다.<br />
                잠시만 기다려주세요.
              </ProgressLoading>
            </div>
          )}
          {data && <DataTable />}
        </div>
        <LearningControls />
      </ArticleBox>
      <ConfirmDeleteModal />
    </MyListContext.Provider>
  )
}

export default MyListContents;