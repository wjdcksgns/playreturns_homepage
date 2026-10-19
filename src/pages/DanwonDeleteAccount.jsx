import styles from './Privacy.module.css';

const DanwonDeleteAccount = () => {
    return (
        <div className={styles.privacyWrap}>
            <h1>단원 기록 계정 삭제 안내</h1>
            <p className={styles.date}>시행일자: 2026년 10월 19일</p>

            <p>
                단원 기록 앱(개발자: 플레이리턴즈)의 계정과 서버에 보관된 기록은 아래 두 가지 방법으로 삭제할 수 있습니다.
            </p>

            <h2>1. 앱에서 바로 삭제</h2>
            <ul>
                <li>앱에 로그인한 뒤 <strong>설정 → 계정 삭제</strong>를 누르고 한 번 더 확인합니다.</li>
                <li>계정과 서버에 올린 기록이 그 자리에서 삭제됩니다.</li>
            </ul>

            <h2>2. 이메일로 요청</h2>
            <ul>
                <li>앱을 쓸 수 없는 경우 playreturns2025@gmail.com 으로 "단원 기록 계정 삭제 요청" 제목의 메일을 보내 주세요.</li>
                <li>본문에 가입한 이메일 주소와 이름을 적어 주세요. 본인 확인을 위해 가입한 이메일 주소에서 보내 주시면 가장 빠릅니다.</li>
                <li>처리 기간: 요청 접수 후 영업일 기준 7일 이내</li>
            </ul>

            <h2>3. 삭제되는 데이터</h2>
            <ul>
                <li>계정 정보: 이름, 소속, 이메일 주소, 휴대전화번호, 암호화된 비밀번호</li>
                <li>서버에 올린 녹음, 사진, 메모와 폴더 정보</li>
                <li>녹음을 받아 적은 내용과 회의록 초안</li>
                <li>로그인 상태를 유지하는 인증 정보</li>
            </ul>

            <h2>4. 유의사항</h2>
            <ul>
                <li>삭제한 계정과 기록은 복구할 수 없습니다.</li>
                <li>앱을 지우는 것만으로는 서버에 올린 기록이 삭제되지 않습니다.</li>
                <li>기기에만 남아 있는 기록은 앱을 삭제하면 함께 지워집니다.</li>
                <li>서버 접속 기록(IP 주소, 접속 일시)은 보안 목적상 3개월 보관 후 삭제됩니다.</li>
            </ul>

            <h2>5. 문의</h2>
            <p>
                이름: 정찬훈 <br />
                소속: 플레이리턴즈(PlayReturns) <br />
                이메일: playreturns2025@gmail.com <br />
                연락처: 010-2868-0655
            </p>
        </div>
    );
};

export default DanwonDeleteAccount;
