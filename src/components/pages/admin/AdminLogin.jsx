import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import styles from './AdminLogin.module.css';

const ADMIN_ID = 'SNUSR01';
const ADMIN_PW = 'snusr01)!';

const AdminLogin = () => {
    const navigate = useNavigate();
    const [id, setId] = useState('');
    const [pw, setPw] = useState('');

    const handleLogin = () => {
        if (id === ADMIN_ID && pw === ADMIN_PW) {
            sessionStorage.setItem('admin', 'true');
            navigate('/admin/upload');
        } else {
            alert('접근 권한이 없습니다.');
        }
    };

    return (
        <div className={styles.wrapper}>
            <h2>서울대학교 멘토-멘티 매칭</h2>

            <div className={styles.form}>
                <input
                    placeholder="ID"
                    value={id}
                    onChange={(e) => setId(e.target.value)}
                />
                <input
                    type="password"
                    placeholder="Password"
                    value={pw}
                    onChange={(e) => setPw(e.target.value)}
                />

                <button onClick={handleLogin}>로그인</button>
            </div>
        </div>
    );
};

export default AdminLogin;
