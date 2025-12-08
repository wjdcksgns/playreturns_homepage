import { useCallback } from 'react';
import { useEffect, useContext, useRef, useState } from 'react';
import { touchDevice } from '../../../common/js/common';
import { VideoTabContext } from '../../../context/MetaAutoLearning/VideoTabContext';

const RectangleCanvas = ({id, size, disabled}) => {
  const {cards, setCards} = useContext(VideoTabContext);
  const targetIdx = cards.findIndex((i) => i.id === id);
  const card = cards[targetIdx];
  const { width, height } = size;
  const { startX, startY, endX, endY } = card.rectangle;

  const canvasRef = useRef(null);
  const contextRef = useRef(null);

  const [ctx, setCtx] = useState();
  const [isDrawing, setIsDrawing] = useState(false);

  const updateCards = useCallback((key, value) => {
    if (card[key] === value) return;
    else {
      const newCards = [...cards];
      newCards[targetIdx][key] = value;

      setCards(newCards);
    }
  }, [card, cards, setCards, targetIdx]);
  
  const getComplementary = useCallback(() => {
    const hexRgb = card.color.split('#')[1].match(/.{1,2}/g);
    const r = 255 - parseInt(hexRgb[0], 16);
    const g = 255 - parseInt(hexRgb[1], 16);
    const b = 255 - parseInt(hexRgb[2], 16);

    return `rgb(${r}, ${g}, ${b})`;
  }, [card.color]);

  const drawRectangle = useCallback(() => {
    ctx.lineWidth = 4;
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    
    if (startX === endX || startY === endY) return;
    ctx.strokeStyle = card.color;
    ctx.strokeRect(startX, startY, endX - startX, endY - startY);
  }, [card.color, ctx, endX, endY, startX, startY]);

  const drawLabelBox = useCallback(() => {
    if (startX === endX || startY === endY) return;
    if (card.label) {
      const x = startX < endX ? startX : endX;
      const y = startY < endY ? startY : endY;
      const width = startX < endX ? endX - startX : startX - endX;
      const height = 34;
      
      ctx.fillStyle = card.color;
      ctx.fillRect(x - (ctx.lineWidth / 2), y - height, width + ctx.lineWidth, height);

      ctx.font = '16px sansSerif';
      ctx.fillStyle = getComplementary();
      ctx.fillText(card.label, x + 10, y - 10);
    }
  }, [card.color, card.label, ctx, endX, endY, getComplementary, startX, startY]);

  const startDrawing = (offsetX, offsetY) => {
    setIsDrawing(true);
    updateCards('rectangle', {
      startX: offsetX,
      startY: offsetY,
      endX: offsetX,
      endY: offsetY,
    })
  }
  const draw = (offsetX, offsetY) => {
    if (ctx && isDrawing) {
      updateCards('rectangle', {
        ...card.rectangle,
        endX: offsetX,
        endY: offsetY,
      })
      drawRectangle();
    }
  }
  const endDrawing = () => {
    if (!disabled) {
      updateCards('step', card.label && card.url && (startX !== endX || startY !== endY) ? 3 : 2);
      updateCards('rectangle', {
        startX: startX < endX ? startX : endX,
        startY: startY < endY ? startY : endY,
        endX: startX > endX ? startX : endX,
        endY: startY > endY ? startY : endY,
      })
      
      finishDrawing();
    }
  }
  const finishDrawing = () => !disabled && setIsDrawing(false);
  
  const canvasEvents = {
    mouse: {
      onMouseDown: ({nativeEvent}) => {
        if (!disabled) {
          const { offsetX, offsetY } = nativeEvent;
    
          startDrawing(offsetX, offsetY);
        }
      },
      onMouseMove: ({nativeEvent}) => {
        if (!disabled) {
          const { offsetX, offsetY } = nativeEvent;
          
          draw(offsetX, offsetY);
        }
      },
      onMouseUp: endDrawing,
      onMouseLeave: finishDrawing,
    },
    touch: {
      onTouchStart: ({nativeEvent}) => {
        if (!disabled) {
          const { clientX, clientY } = nativeEvent.targetTouches[0];
          const { left, top } = canvasRef.current.getBoundingClientRect();
          const offsetX = clientX - left;
          const offsetY = clientY - top;

          startDrawing(offsetX, offsetY);
        }
      },
      onTouchMove: ({nativeEvent}) => {
        if (!disabled) {
          const { clientX, clientY } = nativeEvent.targetTouches[0];
          const { left, top } = canvasRef.current.getBoundingClientRect();
          const offsetX = clientX - left;
          const offsetY = clientY - top;
          
          draw(offsetX, offsetY);
        }
      },
      onTouchEnd: endDrawing,
      onTouchCancel: finishDrawing,
    }
  }

  useEffect(() => {
    document.body.style.overflow = touchDevice() && isDrawing ? 'hidden' : '';
  }, [isDrawing])

  useEffect(() => {
    if (!ctx) {
      const canvas = canvasRef.current;
      canvas.width = width;
      canvas.height = height;

      const context = canvas.getContext('2d');
      contextRef.current = context;

      setCtx(context);
      updateCards('canvasSize', {
        width: width,
        height: height,
      })
    }
  }, [ctx, width, height, updateCards])

  useEffect(() => {
    const widthMatch = card.canvasSize.width === width;
    const heightMatch = card.canvasSize.height === height;

    if (widthMatch && heightMatch) return;
    const canvas = canvasRef.current;
    canvas.width = width;
    canvas.height = height;
    
    const ratioX = width / card.canvasSize.width;
    const ratioY = height / card.canvasSize.height;

    updateCards('canvasSize', {
      width: width,
      height: height,
    });

    updateCards('rectangle', {
      startX: startX * ratioX,
      startY: startY * ratioY,
      endX: endX * ratioX,
      endY: endY * ratioY,
    });
  }, [card.canvasSize.height, card.canvasSize.width, endX, endY, height, startX, startY, updateCards, width])
  
  useEffect(() => {
    if (isDrawing) canvasRef.current.parentElement.dataset.drawing = '';
    else {
      delete canvasRef.current.parentElement.dataset.drawing;
      if (ctx) {
        drawRectangle();
        drawLabelBox();
      }
    }
  }, [ctx, drawLabelBox, drawRectangle, isDrawing]);

  return (
    <canvas
        ref={canvasRef}
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          zIndex: card.current ? 200 : 100,
          cursor: disabled ? 'default' : 'crosshair'
        }}
        data-id={id}
        {...(touchDevice() ? canvasEvents.touch : canvasEvents.mouse)}
      ></canvas>
  )
}

export default RectangleCanvas;