import { useContext, useRef, useEffect, useState } from 'react';
import { MyListContext } from '../../../context/MyList/MyListContext';

import styles from './SearchArea.module.css';

import Btn from '../../common/Buttons/Btn';
import TooltipBadgeBtn from '../../common/Buttons/TooltipBadgeBtn';
import OutlineInput from '../../common/Inputs/OutlineInput';
import SelectBox from '../../common/Inputs/SelectBox';

import { IconButton } from '@mui/material';

import SearchIcon from '@mui/icons-material/Search';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';

const SearchArea = () => {
  const {setSelectionModel, setSearchFilter} = useContext(MyListContext);

  const historyBoxRef = useRef();
  
  const [searchField, setSearchField] = useState('objName');
  const [searchValue, setSearchValue] = useState('');
  const [searchHistory, setSearchHistory] = useState([]);
  const [showHistoryToggleBtn, setShowHistoryToggleBtn] = useState(false);
  const [openHistoryBox, setOpenHistoryBox] = useState(false);

  const handleChangeSearchField = e => setSearchField(e.target.value);
  const handleChangeSearchValue = e => setSearchValue(e.target.value);
  const handleSearch = () => {
    setSearchFilter([{
      columnField: searchField,
      value: searchValue.replace(/ /g,""),
      operatorValue: "contains",
    }]);
    if (searchValue.trim()) {
      setSearchHistory(prevHistory => {
        let field = '';
        switch (searchField) {
          case 'objName':
            field = '오브젝트 이름';
            break;
          case 'tags':
            field = '태그';
            break;
          case 'url':
            field = 'URL';
            break;
          default:
            field = '';
            break;
        }
        
        const newHistory = {
          field: field,
          value: searchValue
        }

        const historyArr = [newHistory, ...prevHistory]
        const historyFilter = historyArr.filter((v1, idx) => historyArr.findIndex(v2 => v1.field === v2.field && v1.value === v2.value) === idx);

        return historyFilter.slice(0, 10);
      });
    }
    setSearchValue('');
    setSelectionModel([]);
  }
  const handleKeyUpSearch = (e) => {
    if (e.keyCode === 13) {
      e.preventDefault();

      handleSearch();
    }
  }
  const handleToggleHistoryBox = () => setOpenHistoryBox(!openHistoryBox)

  useEffect(() => {
    if (searchHistory.length === 0) return;
    setShowHistoryToggleBtn(historyBoxRef.current.childNodes[0].offsetHeight > 78);
  }, [searchHistory]);

  const selectBox = {
    label: {
      id: 'searchFilterLabel',
      title: '검색필터'
    },
    props: {
      id: 'searchFilter',
      value: searchField,
      onChange: handleChangeSearchField
    },
    options: [
      {id: 1, value: 'objName', title: '오브젝트 이름'},
      {id: 2, value: 'tags', title: '태그'},
      {id: 3, value: 'url', title: 'URL'}
    ]
  }
  const searchInput = {
    props: {
      id: 'searchKeyword',
      label: '검색어',
      value: searchValue,
      placeholder: '검색어를 입력해주세요',
      onChange: handleChangeSearchValue,
      onKeyUp: handleKeyUpSearch
    }
  }

  return (
    <div className={styles.searchArea}>
      <div className={styles.inputBox}>
        <div className={styles.searchFilter}>
          <SelectBox {...selectBox} />
        </div>
        <div className={styles.searchBox}>
          <OutlineInput {...searchInput} />
          {searchHistory.length !== 0 && (
            <div ref={historyBoxRef} className={styles.historyBox} style={{maxHeight: openHistoryBox ? 'none' : '78px'}}>
              <div className={styles.historyBadges}>
                {searchHistory.map((keyword, idx) => {
                  return (
                    <TooltipBadgeBtn key={idx} title={keyword.field}
                      props={{
                        onClick: () => {
                          let field = '';
                          switch (keyword.field) {
                            case '오브젝트 이름':
                              field = 'objName';
                              break;
                            case '태그':
                              field = 'tags';
                              break;
                            case 'URL':
                              field = 'url';
                              break;
                            default:
                              field = '';
                              break;
                          }
                          setSearchField(field);
                          setSearchFilter([
                            {
                              columnField: field,
                              value: keyword.value.replace(/ /g,""),
                              operatorValue: "contains",
                            }
                          ]);
                          setSearchHistory(prevHistory => {
                            const newHistory = {
                              field: keyword.field,
                              value: keyword.value
                            }
                            
                            const historyArr = [newHistory, ...prevHistory]
                            const historyFilter = historyArr.filter((v1, idx) => {
                              return historyArr.findIndex(v2 => v1.field === v2.field && v1.value === v2.value) === idx;
                            });
                            
                            return historyFilter.slice(0, 10);
                          });
                          setSearchValue('');
                          setSelectionModel([]);
                        }
                      }}
                      deleteBadge={() => {
                        searchHistory.splice(idx, 1)
                        setSearchHistory([...searchHistory])
                        searchHistory.length === 0 && setSearchFilter([]);
                      }}
                    >
                      {keyword.value}
                    </TooltipBadgeBtn>
                  )
                })}
              </div>
              {showHistoryToggleBtn && (
                <IconButton className={styles.btnToggleHistory} onClick={handleToggleHistoryBox} >
                  <ExpandMoreIcon sx={{transform: `rotate(${openHistoryBox ? 180 : 0}deg)`}} />
                </IconButton>
              )}
            </div>
          )}
        </div>
      </div>
      <div className={styles.btnBox}>
        <Btn onClick={handleSearch} size="sm"><SearchIcon /></Btn>
      </div>
    </div>
  )
}

export default SearchArea;