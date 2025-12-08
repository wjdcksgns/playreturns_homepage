import axios from 'axios';
import { getCookie } from '../common/js/cookie';


// axios 인스턴스 생성
const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL,
  xsrfCookieName: 'csrf_access_token',
  xsrfHeaderName: 'X-CSRF-TOKEN',
  withCredentials: true,
});

// axios request interceptor
api.interceptors.request.use(
  (config) => {
    config.headers['X-CSRF-TOKEN'] = getCookie('csrf_access_token');

    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// axios response interceptor
api.interceptors.response.use(
  (response) => {
    return response;
  },
  async (error) => {
    const originalRequest = error.config;
    
    if (error.response.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;
      
      try {
        const refreshInstance = axios.create({
          baseURL: process.env.REACT_APP_API_URL,
          xsrfCookieName: 'csrf_refresh_token',
          xsrfHeaderName: 'X-CSRF-TOKEN',
          withCredentials: true,
        });
        const userId = JSON.parse(JSON.parse(sessionStorage.getItem('persist:root')).user).userId;
        const refreshResponse = await refreshInstance.get(`/api/user/refresh/${userId}`, {
          headers: {
            'X-CSRF-TOKEN': getCookie('csrf_refresh_token')
          }
        });

        if (refreshResponse.data.status === 200) {
          return api(originalRequest);
        }
        else {
          console.log(refreshResponse.data)
        }
      } catch (error) {
        console.error(error);

        window.location.href = '/login';
      }
    }
    
    return Promise.reject(error);
  }
);

export default api;