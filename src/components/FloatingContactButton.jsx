import { useState } from "react";
import { FaEnvelope } from "react-icons/fa"; // 이메일 아이콘
import ContactModal from "./common/Modals/ContactModal"; // 경로 맞게 조정
import styles from "./FloatingContactButton.module.css";

function FloatingContactButton() {
    const [showContact, setShowContact] = useState(false);

    return (
        <>
            <button
                className={styles.floatingBtn}
                onClick={() => setShowContact(true)}
            >
                <FaEnvelope size={20} />
            </button>

            {showContact && (
                <ContactModal onClose={() => setShowContact(false)} />
            )}
        </>
    );
}

export default FloatingContactButton;
