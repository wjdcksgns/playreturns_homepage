import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';

import './common/css/reset.css';
import './common/css/index.css';

import { getCookie } from './common/js/cookie'

import axios from 'axios';

import App from './App';
import reportWebVitals from './reportWebVitals';

import { store } from './app/store'
import { Provider } from 'react-redux'
import { PersistGate } from 'redux-persist/integration/react';
import { persistStore } from 'redux-persist';

axios.defaults.baseURL = process.env.REACT_APP_API_URL;
axios.defaults.withCredentials = true;
axios.defaults.xsrfCookieName = 'csrf_access_token'
axios.defaults.xsrfHeaderName = 'X-CSRF-TOKEN'
axios.defaults.headers.common['X-CSRF-TOKEN'] = getCookie('csrf_access_token');

const root = ReactDOM.createRoot(document.getElementById('root'));

let persistor = persistStore(store);

root.render(
  <Provider store={store}>
    <PersistGate loading={null} persistor={persistor}>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </PersistGate>
  </Provider>
);

reportWebVitals();