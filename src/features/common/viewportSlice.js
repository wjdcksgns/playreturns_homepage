import { createSlice } from '@reduxjs/toolkit'

const initialState = {
  windowWidth: window.innerWidth,
  windowHeight: window.innerHeight,
}

export const viewportSlice = createSlice({
  name: 'viewport',
  initialState,
  reducers: {
    setViewport(state, action) {
      state.windowWidth = action.payload.windowWidth;
      state.windowHeight = action.payload.windowHeight;
    },
    resetViewport(state) {
      state.windowWidth = null;
      state.windowHeight = null;
    },
  },
})

export const { setViewport, resetViewport } = viewportSlice.actions;

export default viewportSlice.reducer