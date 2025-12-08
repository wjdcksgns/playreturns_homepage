import { useState } from 'react';
import styles from './ColorInput.module.css';

const ColorInput = ({changeColor, id}) => {
  const [checkedRadio, setCheckedRadio] = useState(null);
  const [checkedColor, setCheckedColor] = useState('');

  const setCurrentColor = (color) => {
    setCheckedColor(color);
    changeColor(color);
  }

  const handleChangeColor = (e) => {
    e.currentTarget.id === `color${id}` && document.querySelector(`label[for="colorPicker${id}"]`).click();

    if (e.currentTarget.id === `colorPicker${id}`) setCheckedRadio(`color${id}`);
    else setCheckedRadio(e.currentTarget.id);

    setCurrentColor(e.currentTarget.value);
  }

  const colorInputs = [
    {
      id: 1,
      color: 'red',
      bg: '#ff0000',
      value: '#ff0000',
    },
    {
      id: 2,
      color: 'orange',
      bg: '#ffbb00',
      value: '#ffbb00',
    },
    {
      id: 3,
      color: 'yellow',
      bg: '#ffe400',
      value: '#ffe400',
    },
    {
      id: 4,
      color: 'green',
      bg: '#1ddb16',
      value: '#1ddb16',
    },
    {
      id: 5,
      color: 'blue',
      bg: '#0054ff',
      value: '#0054ff',
    },
    {
      id: 6,
      color: 'navy',
      bg: '#050099',
      value: '#050099',
    },
    {
      id: 7,
      color: 'purple',
      bg: '#5f00ff',
      value: '#5f00ff',
    },
    {
      id: 8,
      color: 'color',
      bg: 'conic-gradient(#FF0000, #FF5E00, #FFBB00, #FFE400, #ABF200, #1DDB16, #00D8FF, #0054FF, #0100FF, #5F00FF, #FF00DD, #FF007F)',
      value: checkedColor,
    },
  ]

  return (
    <div className={styles.colorTool}>
      {colorInputs.map(input => {
        return (
          <div key={input.id} className={styles.inputColor}>
            <input type="radio" id={input.color + id} name={`choose-color${id}`} onChange={handleChangeColor} value={input.value} checked={checkedRadio === (input.color + id)} />
            <label htmlFor={input.color + id} style={{background: input.bg}}></label>
            {input.color === 'color' && 
            <>
              <input className={styles.colorPicker} type="color" id={`colorPicker${id}`} value={checkedColor} onChange={handleChangeColor} />
              <label htmlFor={`colorPicker${id}`}></label>
            </>}
          </div>
        )
      })}
      <span className={styles.colorInfo}>{checkedColor || '컬러를 선택해주세요.'}</span>
    </div>
  )
}

export default ColorInput;