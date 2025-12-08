import { useState } from 'react';

import TabMenu from '../../common/Tabs/TabMenu';
import TabContents from '../../common/Tabs/TabContents';
import LineBtn from '../../common/Buttons/LineBtn';
import AlertModal from '../../common/Modals/AlertModal';
import VideoTabContents from './VideoTabContents';

const MetaAutoLearningContents = () => {
  const [activeTab, setActiveTab] = useState(1);
  const [open, setOpen] = useState(false);
  
  const tabs = [
    {
      id: 1,
      name: '동영상',
      contents: (
        <VideoTabContents />
      ),
    },
    {
      id: 2,
      name: '이미지',
      contents: (
        <>
          이미지 자동
        </>
      ),
    },
  ]

  const handleModalOpen = () => setOpen(true);
  const handleModalClose = (e, reason) => {
    if (reason && reason === 'backdropClick') return;
    setOpen(false);
  }
  const handleClickTab = (id) => id === 1 ? setActiveTab(id) : handleModalOpen();

  return (
    <>
      <TabMenu>
        {tabs.map(tab => {
          return (
            <li key={tab.id}>
              <LineBtn
                onClick={() => handleClickTab(tab.id)}
                active={tab.id === activeTab}
                props={{
                  'data-tab-target': tab.name,
                }}
              >
                {tab.name}
              </LineBtn>
            </li>
          )
        })}
      </TabMenu>
      {tabs.map(tab => tab.id === activeTab && <TabContents key={tab.id} id={tab.id} name={tab.name}>{tab.contents}</TabContents>)}
      <AlertModal title="이미지 업로드" open={open} handleClose={handleModalClose}>
        해당 서비스는 준비중입니다.<br />더 좋은 서비스로 찾아뵙겠습니다.
      </AlertModal>
    </>
  )
}

export default MetaAutoLearningContents;