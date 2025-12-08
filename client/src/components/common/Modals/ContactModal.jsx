import emailjs from "@emailjs/browser";
import { useState } from "react";
import styles from "./ContactModal.module.css";

function ContactModal({ onClose }) {
    const [form, setForm] = useState({ name: "", email: "", message: "" });
    const [loading, setLoading] = useState(false);

    const handleChange = (e) => {
        setForm({ ...form, [e.target.name]: e.target.value });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();

        if (!form.name || !form.email || !form.message) {
            alert("모든 항목을 입력해주세요!");
            return;
        }

        setLoading(true);

        try {
            // ✅ EmailJS 전송
            const result = await emailjs.send(
                "service_0w5wr3d", // ← Service ID
                "template_5h0pwio", // ← Template ID
                {
                    ...form,
                    time: new Date().toLocaleString("ko-KR"), // 문의 시각 자동 추가
                },
                "jI1V7H7gx8L7NVV8w" // ← Public Key
            );

            if (result.text === "OK") {
                alert("문의가 성공적으로 전송되었습니다!\n담당자가 확인 후 회신드리겠습니다 😊");
                onClose();
            }
        } catch (error) {
            console.error("메일 전송 오류:", error);
            alert("메일 전송 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className={styles.overlay} onClick={onClose}>
            <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
                <h3>문의하기</h3>
                <form onSubmit={handleSubmit} className={styles.form}>
                    <input
                        type="text"
                        name="name"
                        placeholder="이름"
                        value={form.name}
                        onChange={handleChange}
                        required
                    />
                    <input
                        type="email"
                        name="email"
                        placeholder="이메일"
                        value={form.email}
                        onChange={handleChange}
                        required
                    />
                    <textarea
                        name="message"
                        placeholder="문의 내용을 입력하세요"
                        rows="5"
                        value={form.message}
                        onChange={handleChange}
                        required
                    />
                    <div className={styles.actions}>
                        <button type="submit" disabled={loading}>
                            {loading ? "전송 중..." : "보내기"}
                        </button>
                        <button type="button" onClick={onClose}>닫기</button>
                    </div>
                </form>
            </div>
        </div>
    );
}

export default ContactModal;
