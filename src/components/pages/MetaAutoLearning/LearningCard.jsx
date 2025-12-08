import { useCallback, useContext, useState } from 'react';
import { VideoTabContext } from '../../../context/MetaAutoLearning/VideoTabContext';

import { useSelector } from 'react-redux';

import Btn from '../../common/Buttons/Btn';
import ColorInput from '../../common/Inputs/ColorInput';
import InfoInput from '../../common/Inputs/InfoInput';
import TagInput from '../../common/Inputs/TagInput';
import StepCard from './StepCard';

import api from '../../../services/interceptor';

const LearningCard = ({id}) => {
  const userId = useSelector((state) => state.user.userId);
  const {currentStep, setCurrentStep, cards, setCards} = useContext(VideoTabContext);
  const targetIdx = cards.findIndex((i) => i.id === id);
  const card = cards[targetIdx];

  const infoColor = {
    success: '#6a6a6a',
    danger: '#EB1D36'
  }

  const [tag, setTag] = useState('');
  const [inputInfo, setInputInfo] = useState({
    label: {
      msg: '한글, 영문 최대 10글자까지 가능합니다.',
      color: infoColor.success,
    },
    url: {
      msg: '인식 후 해당 URL을 실행합니다.',
      color: infoColor.success,
    },
    tags: {
      msg: '최대 10개만 추가할 수 있습니다.',
      color: infoColor.success,
    }
  });
  const updateCards = useCallback((key, value) => {
    if (card[key] === value) return;
    else {
      const newCards = [...cards];
      newCards[targetIdx][key] = value;

      setCards(newCards);
    }
  }, [card, cards, setCards, targetIdx]);
  const validation = {
    null: {
      check: () => !tag || tag.length === 0 || tag === '',
      errMsg: '태그를 입력해주세요.',
    },
    match: {
      check: () => card.tags.some(v => v === tag) || card.label === tag,
      errMsg: '이미 등록된 태그입니다',
    },
    max: {
      check: () => card.tags.length >= 9,
      errMsg: '태그는 최대 10개까지 등록할 수 있습니다.'
    },
  }
  const validator = (type) => {
    if (validation[type].check()) {
      setInputInfo((prev) => ({
        ...prev,
        tags: {
          msg: validation[type].errMsg,
          color: infoColor.danger
        }
      }))
      return;
    }
  }
  const exists = {
    pass: () => {
      setInputInfo((prev) => ({
        ...prev,
        label: {
          msg: '한글, 영문 최대 10글자까지 가능합니다.',
          color: infoColor.success,
        }
      }))
      updateCards('step', card.label && card.url && (card.rectangle.startX !== card.rectangle.endX || card.rectangle.startY !== card.rectangle.endY) ? 3 : 2);
    },
    fail: () => {
      setInputInfo((prev) => ({
        ...prev,
        label: {
          msg: '이미 등록된 이름입니다.',
          color: infoColor.danger,
        }
      }))
      updateCards('step', 2);
    }
  }

  const handleStartLearning = () => {
    updateCards('step', 1);
    setCurrentStep(2);
  }
  const handleCreateTag = (e) => {
    const enter = e.keyCode === 13;
    const spaceBar = e.keyCode === 32;
    const backSpace = e.keyCode === 8;

    if (spaceBar || enter) {
      e.preventDefault();

      validator('null');
      validator('match');
      validator('max');
      if (!validation.null.check() && !validation.match.check() && !validation.max.check()) {
        setInputInfo((prev) => ({
          ...prev,
          tags: {
            msg: '최대 10개만 추가할 수 있습니다.',
            color: infoColor.success,
          }
        }))
        const newTags = [...card.tags, tag];
        updateCards('tags', newTags);
      }
      setTag('');
    }
    if (backSpace && !e.currentTarget.value) {
      e.preventDefault();
      const newTags = card.tags.slice(0, card.tags.length - 1);
      updateCards('tags', newTags);
    }
  }
  const handleChangeLabel = async (e) => {
    const label = e.currentTarget.value;
    
    updateCards('label', label);

    if (!label) {
      setInputInfo((prev) => ({
        ...prev,
        label: {
          msg: '오브젝트 이름을 입력해주세요.',
          color: infoColor.danger,
        }
      }))
      updateCards('step', 2);
    } else {
      const isAllPass = cards.every((v, i) => cards.findIndex(v2 => v.label === v2.label) === i);
      if (isAllPass) {
        try {
          const response = await api.get(`/api/meta-auto-learning/object/exists/${userId}/${label}`);
          const key = response.data ? 'fail' : 'pass';

          exists[key]();
        }
        catch (err) {
          console.log(err);
        }
      }
      else {
        exists.fail();
      }
    }
  }
  const handleChangeUrl = (e) => {
    updateCards('url', e.currentTarget.value);
    updateCards('step', card.label && inputInfo.label.color === infoColor.success && card.url && (card.rectangle.startX !== card.rectangle.endX || card.rectangle.startY !== card.rectangle.endY) ? 3 : 2);
  }

  return (
    <div style={{margin: '0 10px'}} data-id={id} data-step={card.step}>
      <StepCard number="1">
        <Btn size="md" fullWidth={true} onClick={handleStartLearning} props={{disabled: currentStep < 1 || card.step > 0}}>학습할 오브젝트 지정</Btn>
        {card.step > 0 && <ColorInput id={id} changeColor={(color) => updateCards('color', color)} />}
      </StepCard>
      {card.step > 1 && (
        <StepCard number="2">
          <InfoInput type="text" id={`objLabel${id}`} label="오브젝트 이름 지정" value={card.label} onChange={handleChangeLabel} placeholder="오브젝트 이름을 입력해주세요." maxLength="10" info={inputInfo.label} />
          <InfoInput type="text" id={`objUrl${id}`} label="URL" value={card.url} onChange={handleChangeUrl} placeholder="URL을 입력해주세요." info={inputInfo.url} />
          <TagInput type="text" id={`objTags${id}`} label="태그 편집" value={tag} onChange={e => setTag(e.currentTarget.value)} onKeyDown={handleCreateTag} placeholder="태그를 입력해주세요." maxLength="10" info={inputInfo.tags} tags={card.label ? [card.label, ...card.tags] : card.tags} />
        </StepCard>
      )}
    </div>
  )
}

export default LearningCard;