import styles from './Privacy.module.css';

const PetsangDeleteAccount = () => {
    return (
        <div className={styles.privacyWrap}>
            <h1>펫상 계정 삭제 안내</h1>
            <p className={styles.date}>시행일자: 2026년 5월 14일</p>

            <p>
                펫상 앱의 계정 및 개인정보 삭제를 원하시는 경우 아래 안내에 따라 요청하실 수 있습니다.
                회사는 요청 접수 후 관련 법령에 따라 지체 없이 처리합니다.
            </p>

            <h2>1. 삭제되는 데이터</h2>
            <ul>
                <li>익명 계정 정보(Firebase 익명 사용자 ID)</li>
                <li>등록된 반려동물 정보(이름, 종류, 성별, 나이, 프로필 사진)</li>
                <li>AI 분석 결과 히스토리(관상·MBTI·닮은꼴·운세·궁합)</li>
                <li>FCM 푸시 알림 토큰</li>
            </ul>

            <h2>2. 삭제 요청 방법</h2>
            <ul>
                <li>이메일 요청: 아래 이메일로 "펫상 계정 삭제 요청" 제목으로 연락 주세요.</li>
                <li>이메일 본문에 앱에서 확인 가능한 사용자 ID 또는 기기 정보를 포함해 주시기 바랍니다.</li>
                <li>처리 기간: 요청 접수 후 영업일 기준 7일 이내 처리됩니다.</li>
            </ul>

            <h2>3. 유의사항</h2>
            <ul>
                <li>계정 삭제 시 모든 분석 기록 및 반려동물 정보가 영구적으로 삭제되며 복구가 불가능합니다.</li>
                <li>반려동물 사진은 AI 분석 완료 후 24시간 이내 자동 삭제되므로 별도 요청이 필요하지 않습니다.</li>
                <li>앱 삭제만으로는 서버에 저장된 데이터가 삭제되지 않으니 반드시 삭제 요청을 해주시기 바랍니다.</li>
                <li>삭제 후에도 법령에서 별도 보관을 요구하는 정보는 해당 기간 동안 보관될 수 있습니다.</li>
            </ul>

            <h2>4. 개인정보 보호책임자 연락처</h2>
            <p>
                이름: 정찬훈 <br />
                소속: 플레이리턴즈(PlayReturns) <br />
                이메일: playreturns2025@gmail.com <br />
                연락처: 010-2868-0655
            </p>
        </div>
    );
};

export default PetsangDeleteAccount;
