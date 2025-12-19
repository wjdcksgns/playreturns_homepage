import { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import styles from './AdminUpload.module.css';
import { excelToCsvFile } from './adminUtils';


const API_BASE_URL = 'https://api.playreturns.co.kr/snu';


const AdminUpload = () => {
    const navigate = useNavigate();

    const [analysisDone, setAnalysisDone] = useState(false);
    const mentorInputRef = useRef(null);
    const menteeInputRef = useRef(null);

    const [mentorFile, setMentorFile] = useState(null);
    const [menteeFile, setMenteeFile] = useState(null);

    // 🔐 로그인 체크
    useEffect(() => {
        if (sessionStorage.getItem('admin') !== 'true') {
            alert('접근 권한이 없습니다.');
            navigate('/admin/login');
        }
    }, [navigate]);

    const handleMentorSelect = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        if (file.name.match(/\.(xlsx|xls)$/i)) {
            const csvFile = await excelToCsvFile(file, 'mentor_raw.csv');
            setMentorFile(csvFile);
        } else {
            const renamedFile = new File(
                [file],
                'mentor_raw.csv',
                { type: file.type }
            );
            setMentorFile(renamedFile);
        }
    };

    const handleMenteeSelect = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        if (file.name.match(/\.(xlsx|xls)$/i)) {
            const csvFile = await excelToCsvFile(file, 'mentee_raw.csv');
            setMenteeFile(csvFile);
        } else {
            const renamedFile = new File(
                [file],
                'mentee_raw.csv',
                { type: file.type }
            );
            setMenteeFile(renamedFile);
        }
    };

    const canAnalyze = mentorFile && menteeFile;

    // ✅ 반드시 컴포넌트 안에 있어야 함
    const handleAnalyze = async () => {
        if (!mentorFile || !menteeFile) return;

        try {
            const formData = new FormData();
            formData.append('mentor', mentorFile);
            formData.append('mentee', menteeFile);

            const response = await fetch(`${API_BASE_URL}/analyze`, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error('분석 요청 실패');
            }

            const result = await response.json();

            alert('분석이 완료되었습니다.');
            console.log('서버 응답:', result);
            setAnalysisDone(true);
        } catch (error) {
            console.error(error);
            alert('분석 중 오류가 발생했습니다.');
        }
    };

    const handleDownload = () => {
        window.location.href = `${API_BASE_URL}/download`;
    };


    return (
        <div className={styles.wrapper}>
            <h2>멘토-멘티 매칭 분석</h2>

            <div className={styles.uploadBox}>
                <h3>멘토 파일 업로드</h3>
                <button onClick={() => mentorInputRef.current.click()}>
                    파일 선택
                </button>
                <input
                    ref={mentorInputRef}
                    type="file"
                    accept=".csv,.xlsx,.xls"
                    onChange={handleMentorSelect}
                    hidden
                />
                {mentorFile && (
                    <p className={styles.fileName}>
                        선택된 파일: {mentorFile.name}
                    </p>
                )}
            </div>

            <div className={styles.uploadBox}>
                <h3>멘티 파일 업로드</h3>
                <button onClick={() => menteeInputRef.current.click()}>
                    파일 선택
                </button>
                <input
                    ref={menteeInputRef}
                    type="file"
                    accept=".csv,.xlsx,.xls"
                    onChange={handleMenteeSelect}
                    hidden
                />
                {menteeFile && (
                    <p className={styles.fileName}>
                        선택된 파일: {menteeFile.name}
                    </p>
                )}
            </div>

            <button
                className={`${styles.analyzeBtn} ${canAnalyze ? styles.active : ''}`}
                disabled={!canAnalyze}
                onClick={handleAnalyze}
            >
                분석 시작
            </button>
            {analysisDone && (
                <button
                    className={styles.downloadBtn}
                    onClick={handleDownload}
                >
                    결과 파일 다운로드
                </button>
            )}

        </div>
    );
};

export default AdminUpload;
