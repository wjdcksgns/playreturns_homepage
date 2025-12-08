import { configureStore } from '@reduxjs/toolkit';
import storage from 'redux-persist/lib/storage/session';
import { combineReducers } from 'redux';
import { persistReducer } from 'redux-persist';
import thunk from 'redux-thunk';

import userReducer from '../features/user/userSlice';
import viewportReducer from '../features/common/viewportSlice';

const reducers = combineReducers({
  user: userReducer,
  viewport: viewportReducer,
});

const persistConfig = {
  key: 'root',
  storage,
};

const persistedReducer = persistReducer(persistConfig, reducers);

export const store = configureStore({
  reducer: persistedReducer,
  devTools: process.env.NODE_ENV !== 'production',
  middleware: [thunk],
});