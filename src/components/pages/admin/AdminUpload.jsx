import { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import styles from './AdminUpload.module.css';
import { excelToCsvFile } from './adminUtils';




const API_BASE_URL = 'https://api.playreturns.co.kr/snu';



const AdminUpload = () => {
    const navigate = useNavigate();

    const mentorInputRef = useRef(null);
    const menteeInputRef = useRef(null);

    const blacklistInputRef = useRef(null);
    const [blacklistFile, setBlacklistFile] = useState(null);


    const [mentorFile, setMentorFile] = useState(null);
    const [menteeFile, setMenteeFile] = useState(null);

    const [analysisDone, setAnalysisDone] = useState(false);

    // 팝업 / 로딩 상태
    const [showConfirm, setShowConfirm] = useState(false);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [progress, setProgress] = useState(0);

    const [progressText, setProgressText] = useState('');

    // 🔐 로그인 체크 (서버 인증 기준)
    useEffect(() => {
        const token = sessionStorage.getItem('adminToken');
        if (token !== 'ADMIN_OK') {
            alert('접근 권한이 없습니다.');
            navigate('/admin/login');
        }
    }, [navigate]);


    /* =========================
       파일 선택 핸들러
    ========================= */
    const handleMentorSelect = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        if (file.name.match(/\.(xlsx|xls)$/i)) {
            const csvFile = await excelToCsvFile(file, 'mentor_raw.csv');
            setMentorFile(csvFile);
        } else {
            const renamedFile = new File([file], 'mentor_raw.csv', {
                type: file.type,
            });
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
            const renamedFile = new File([file], 'mentee_raw.csv', {
                type: file.type,
            });
            setMenteeFile(renamedFile);
        }
    };

    const handleBlacklistSelect = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        if (file.name.match(/\.(xlsx|xls)$/i)) {
            const csvFile = await excelToCsvFile(file, 'blocklist.csv');
            setBlacklistFile(csvFile);
        } else {
            const renamedFile = new File([file], 'blocklist.csv', {
                type: file.type,
            });
            setBlacklistFile(renamedFile);
        }
    };


    const canAnalyze = mentorFile && menteeFile;

    /* =========================
       실제 분석 실행
    ========================= */
    const startAnalyze = async () => {
        if (!mentorFile || !menteeFile) return;

        setIsAnalyzing(true);
        setProgress(0);
        setProgressText('업로드된 파일을 확인하는 중입니다...');

        const timer = setInterval(() => {
            setProgress((prev) => {
                const next = prev + 10;

                if (next === 20) {
                    setProgressText('멘토 데이터를 전처리하는 중입니다...');
                } else if (next === 40) {
                    setProgressText('멘티 데이터를 전처리하는 중입니다...');
                } else if (next === 60) {
                    setProgressText('멘토-멘티 매칭 점수를 계산하는 중입니다...');
                } else if (next === 80) {
                    setProgressText('매칭 결과를 정리하는 중입니다...');
                }

                return next < 90 ? next : 90;
            });
        }, 400);

        try {
            const formData = new FormData();
            formData.append('mentor', mentorFile);
            formData.append('mentee', menteeFile);
            // ✅ 선택 사항
            if (blacklistFile) {
                formData.append('blacklist', blacklistFile);
            }

            const response = await fetch(`${API_BASE_URL}/analyze`, {
                method: 'POST',
                headers: {
                    'admin-token': sessionStorage.getItem('adminToken'),
                },
                body: formData,
            });


            if (!response.ok) {
                throw new Error('분석 요청 실패');
            }

            clearInterval(timer);

            // ✅ 100% 먼저 보여주기
            setProgress(100);
            setProgressText('분석을 마무리하고 있습니다...');

            // ✅ 100% 상태를 잠깐 유지
            setTimeout(() => {
                setIsAnalyzing(false);
                setAnalysisDone(true);
                alert('분석이 완료되었습니다.');
            }, 600);
        } catch (error) {
            clearInterval(timer);
            setIsAnalyzing(false);
            alert('분석 중 오류가 발생했습니다.');
        }
    };


    const handleDownload = async () => {
        try {
            const res = await fetch(`${API_BASE_URL}/download`, {
                headers: {
                    'admin-token': sessionStorage.getItem('adminToken'),
                },
            });

            if (!res.ok) throw new Error();

            const blob = await res.blob();
            const url = window.URL.createObjectURL(blob);

            const a = document.createElement('a');
            a.href = url;
            a.download = 'match_result.xlsx';
            document.body.appendChild(a);
            a.click();

            a.remove();
            window.URL.revokeObjectURL(url);

        } catch {
            alert('다운로드 중 오류가 발생했습니다.');
        }
    };


    return (
        <div
            className={styles.page}
            style={{
                backgroundImage: `url(${process.env.PUBLIC_URL}/upload_bg.jpg)`,
            }}
        >
            <div className={styles.wrapper}>
                <h2>멘토-멘티 매칭 분석</h2>

                {/* =========================
               파일 업로드 영역
            ========================= */}
                <div className={styles.uploadGrid}>
                    {/* 멘토 */}
                    <div className={styles.uploadBox}>
                        <h3>멘토 파일 업로드</h3>

                        <div
                            className={`${styles.uploadSquare} ${mentorFile ? styles.checked : ''
                                }`}
                            onClick={() => mentorInputRef.current.click()}
                        >
                            {mentorFile ? '✓' : '+'}
                        </div>

                        <input
                            ref={mentorInputRef}
                            type="file"
                            accept=".csv,.xlsx,.xls"
                            onChange={handleMentorSelect}
                            hidden
                        />

                        {mentorFile && (
                            <div className={styles.fileInfo}>
                                <span>{mentorFile.name}</span>
                                <button
                                    className={styles.removeBtn}
                                    onClick={() => {
                                        setMentorFile(null);
                                        mentorInputRef.current.value = '';
                                    }}
                                >
                                    ×
                                </button>
                            </div>
                        )}
                    </div>

                    {/* 멘티 */}
                    <div className={styles.uploadBox}>
                        <h3>멘티 파일 업로드</h3>

                        <div
                            className={`${styles.uploadSquare} ${menteeFile ? styles.checked : ''
                                }`}
                            onClick={() => menteeInputRef.current.click()}
                        >
                            {menteeFile ? '✓' : '+'}
                        </div>

                        <input
                            ref={menteeInputRef}
                            type="file"
                            accept=".csv,.xlsx,.xls"
                            onChange={handleMenteeSelect}
                            hidden
                        />

                        {menteeFile && (
                            <div className={styles.fileInfo}>
                                <span>{menteeFile.name}</span>
                                <button
                                    className={styles.removeBtn}
                                    onClick={() => {
                                        setMenteeFile(null);
                                        menteeInputRef.current.value = '';
                                    }}
                                >
                                    ×
                                </button>
                            </div>
                        )}
                    </div>

                    {/* 블락리스트 (선택) */}
                    <div className={styles.uploadBox}>
                        <h3>블락리스트 업로드 (선택)</h3>

                        <div
                            className={`${styles.uploadSquare} ${blacklistFile ? styles.checked : ''}`}
                            onClick={() => blacklistInputRef.current.click()}
                        >
                            {blacklistFile ? '✓' : '+'}
                        </div>

                        <input
                            ref={blacklistInputRef}
                            type="file"
                            accept=".csv,.xlsx,.xls"
                            onChange={handleBlacklistSelect}
                            hidden
                        />

                        {blacklistFile && (
                            <div className={styles.fileInfo}>
                                <span>{blacklistFile.name}</span>
                                <button
                                    className={styles.removeBtn}
                                    onClick={() => {
                                        setBlacklistFile(null);
                                        blacklistInputRef.current.value = '';
                                    }}
                                >
                                    ×
                                </button>
                            </div>
                        )}
                    </div>
                </div>

                {/* =========================
               분석 시작 버튼
            ========================= */}
                <button
                    className={`${styles.analyzeBtn} ${canAnalyze ? styles.active : ''
                        }`}
                    disabled={!canAnalyze || isAnalyzing}
                    onClick={() => setShowConfirm(true)}
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

                {/* =========================
                분석 시작 확인 팝업
                ========================= */}
                {showConfirm && (
                    <div className={styles.modalOverlay}>
                        <div className={styles.confirmModal}>
                            <h4>매칭 분석 시작</h4>

                            <p className={styles.modalDesc}>
                                멘토–멘티 매칭 분석을 시작하시겠습니까?
                                <br />
                                {blacklistFile ? (
                                    <>
                                        블랙리스트 파일이 적용되어<br />
                                        해당 인원은 매칭 대상에서 자동 제외됩니다.
                                    </>
                                ) : (
                                    <>
                                        블랙리스트 없이<br />
                                        전체 인원을 대상으로 매칭이 진행됩니다.
                                    </>
                                )}
                            </p>

                            <div className={styles.modalActions}>
                                <button
                                    className={styles.cancelBtn}
                                    onClick={() => setShowConfirm(false)}
                                >
                                    취소
                                </button>
                                <button
                                    className={styles.confirmBtn}
                                    onClick={() => {
                                        setShowConfirm(false);
                                        startAnalyze();
                                    }}
                                >
                                    분석 시작
                                </button>
                            </div>
                        </div>
                    </div>
                )}


                {/* =========================
               분석 중 로딩 오버레이
            ========================= */}
                {isAnalyzing && (
                    <div className={styles.loadingOverlay}>
                        <div className={styles.loadingBox}>
                            <p>{progressText}</p>

                            <div className={styles.progressBar}>
                                <div
                                    className={styles.progress}
                                    style={{ width: `${progress}%` }}
                                />
                            </div>

                            <span>{progress}%</span>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );

};

export default AdminUpload;
