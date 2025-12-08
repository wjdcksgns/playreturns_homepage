import express from "express";
import bodyParser from "body-parser";
import cors from "cors";
import nodemailer from "nodemailer";
import dotenv from "dotenv";

dotenv.config();

const app = express();
app.use(cors());
app.use(bodyParser.json());

app.post("/contact", async (req, res) => {
    const { name, email, message } = req.body;

    try {
        const transporter = nodemailer.createTransport({
            service: "gmail", // 가비아 SMTP면 host/port 지정 필요
            auth: {
                user: process.env.MAIL_USER,
                pass: process.env.MAIL_PASS,
            },
        });

        await transporter.sendMail({
            from: `"${name}" <${email}>`,
            to: process.env.MAIL_USER,
            subject: `[플레이리턴즈 문의] ${name}`,
            html: `
    <h3>플레이리턴즈 문의</h3>
    <p><strong>이름:</strong> ${name}</p>
    <p><strong>이메일:</strong> ${email}</p>
    <p><strong>문의 내용:</strong><br/>${message.replace(/\n/g, "<br/>")}</p>
  `,
        });


        res.json({ success: true, message: "메일 전송 성공!" });
    } catch (error) {
        console.error("메일 전송 오류:", error);
        res.status(500).json({ success: false, error: "메일 전송 실패" });
    }
});

const PORT = 4000;
app.listen(PORT, () => console.log(`✅ Server running on http://localhost:${PORT}`));
