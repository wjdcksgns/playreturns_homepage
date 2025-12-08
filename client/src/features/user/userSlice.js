import { createSlice } from '@reduxjs/toolkit'

const initialState = {
  userId: null,
  userName: null,
}

export const userSlice = createSlice({
  name: 'user',
  initialState,
  reducers: {
    login(state, action) {
      state.userId = action.payload.id;
      state.userName = action.payload.name;
    },
    logout(state) {
      state.userId = null;
      state.userName = null;
    },
    update(state, action) {
      state.userName = action.payload;
    }
  },
})

export const { login, logout, update } = userSlice.actions;

export default userSlice.reducer