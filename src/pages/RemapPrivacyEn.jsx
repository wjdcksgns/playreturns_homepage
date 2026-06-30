import styles from './Privacy.module.css';

const RemapPrivacyEn = () => {
    return (
        <div className={styles.privacyWrap}>
            <h1>RE:MAP Privacy Policy</h1>
            <p className={styles.date}>Effective date: December 4, 2025</p>

            <p>
                PlayReturns (hereinafter the "Company") values users' personal information and complies with
                the Personal Information Protection Act and related laws. This Privacy Policy is intended to
                transparently explain the purposes of processing, retention periods, and protective measures
                for personal information collected while using the RE:MAP app (the "App").
            </p>

            <h2>1. Personal Information We Collect</h2>
            <ul>
                <li>At sign-up: Google login account information (email address, profile name, unique user ID)</li>
                <li>During use: Location information (GPS-based), and photos, videos, and text records the user registers</li>
                <li>Automatically collected: Device information (OS, model), access logs, app usage history, error logs</li>
            </ul>

            <h2>2. Purpose of Collection and Use</h2>
            <ul>
                <li>Providing map-based record saving and viewing features</li>
                <li>User account identification and login authentication (Google OAuth)</li>
                <li>Photo, video, and text upload and record management features</li>
                <li>Operating and improving the service, and error monitoring</li>
                <li>Security and prevention of fraudulent use</li>
                <li>Compliance with legal obligations and dispute resolution</li>
            </ul>

            <h2>3. Retention and Use Period</h2>
            <ul>
                <li>Upon account withdrawal, data is deleted immediately and destroyed in an unrecoverable manner.</li>
                <li>Records created by the user in the app (photos, videos, text, etc.) are deleted together upon withdrawal.</li>
                <li>However, where laws require separate retention, data is retained for the required period.</li>
            </ul>

            <h2>4. Provision to Third Parties and Outsourcing</h2>
            <p>As a rule, the Company does not provide personal information to third parties without the user's consent.</p>
            <p>However, to provide the service, processing may be outsourced to external providers as follows:</p>
            <ul>
                <li>Google Cloud Platform: Map service (Google Maps API), login authentication</li>
                <li>Cloud server providers (e.g., AWS): Data storage and service operation</li>
            </ul>

            <h2>5. Security Measures</h2>
            <ul>
                <li>Encrypted storage and transmission of important information</li>
                <li>Minimized access privileges and internal security policies</li>
                <li>Regular server inspections and security updates</li>
                <li>Monitoring of data access logs and anomaly detection</li>
            </ul>

            <h2>6. User Rights</h2>
            <p>
                Users may view, correct, or delete their personal information at any time, and may request account
                withdrawal. The Company will act without delay on the user's request in accordance with applicable laws.
            </p>

            <h2>7. Personal Information Protection Officer</h2>
            <p>
                Name: Chanhoon Jung <br />
                Affiliation: PlayReturns <br />
                Email: playreturns2025@gmail.com <br />
                Phone: +82-10-2868-0655
            </p>

            <h2>8. Changes to This Privacy Policy</h2>
            <p>
                If this Privacy Policy is changed, we will provide advance notice through an in-app notice or a notice on our website.
            </p>
        </div>
    );
};

export default RemapPrivacyEn;
