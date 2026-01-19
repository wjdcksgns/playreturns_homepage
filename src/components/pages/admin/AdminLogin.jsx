import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import styles from './AdminLogin.module.css';

const AdminLogin = () => {
    const navigate = useNavigate();
    const [id, setId] = useState('');
    const [pw, setPw] = useState('');
    const [error, setError] = useState('');

    const handleLogin = async (e) => {
        e.preventDefault();

        if (!id || !pw) {
            setError('아이디와 비밀번호를 입력해주세요.');
            return;
        }

        try {
            const res = await fetch('https://api.playreturns.co.kr/auth/admin/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ id, pw }),
            });

            if (!res.ok) throw new Error();

            const data = await res.json();
            sessionStorage.setItem('adminToken', data.token);
            navigate('/admin/upload');

        } catch {
            setError('계정이 존재하지 않습니다. 접근 권한이 없습니다.');
        }
    };


    return (
        <div className={styles.page}>
            <div className={styles.container}>
                {/* 좌측 로고 영역 */}
                <div className={styles.left}>
                    <img src="/snu_logo.png" alt="서울대학교 로고" />
                </div>


                {/* 우측 로그인 영역 */}
                <div className={styles.right}>
                    <h2>서울대학교 멘토-멘티 매칭</h2>

                    <form className={styles.form} onSubmit={handleLogin}>
                        <div className={styles.inputGroup}>
                            <input
                                placeholder="ID"
                                value={id}
                                onChange={(e) => {
                                    setId(e.target.value);
                                    setError('');
                                }}
                            />
                        </div>

                        <div className={styles.inputGroup}>
                            <input
                                type="password"
                                placeholder="Password"
                                value={pw}
                                onChange={(e) => {
                                    setPw(e.target.value);
                                    setError('');
                                }}
                            />
                        </div>

                        {error && (
                            <div className={styles.errorMsg}>
                                <small>{error}</small>
                            </div>
                        )}

                        <button type="submit">로그인</button>
                    </form>
                </div>
            </div>
        </div>
    );
};

export default AdminLogin;
