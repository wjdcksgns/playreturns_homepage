import styles from './Privacy.module.css';

const RemapTermsEn = () => {
    return (
        <div className={styles.privacyWrap}>
            <h1>RE:MAP Terms of Service</h1>
            <p className={styles.date}>Effective date: April 20, 2026</p>

            <h2>Article 1 (Purpose)</h2>
            <p>
                These Terms govern the conditions and procedures for using the RE:MAP app (the "Service") provided by
                PlayReturns (the "Company"), as well as the rights, obligations, and responsibilities between the
                Company and users.
            </p>

            <h2>Article 2 (Definitions)</h2>
            <ul>
                <li>"Service" means the map-based travel record app RE:MAP and related additional services provided by the Company.</li>
                <li>"User" means a person who agrees to these Terms and uses the Service.</li>
                <li>"Content" means all forms of material—text, photos, videos, etc.—created or registered by a user within the Service.</li>
                <li>"Subscription" means using premium services such as AI features through a monthly recurring payment.</li>
            </ul>

            <h2>Article 3 (Effect and Amendment of Terms)</h2>
            <ul>
                <li>These Terms take effect by being posted on the Service screen or otherwise notified to users.</li>
                <li>The Company may amend these Terms within the scope not violating applicable laws, and will give notice at least 7 days in advance.</li>
                <li>If a user does not agree to the amended Terms, the user may stop using the Service and withdraw.</li>
            </ul>

            <h2>Article 4 (Provision and Change of Service)</h2>
            <ul>
                <li>The Company provides the following services:
                    <ul>
                        <li>Creating and managing location-based travel records (pin map) on the map</li>
                        <li>Uploading and storing photos, videos, and text</li>
                        <li>Viewing public records (Our RE:MAP)</li>
                        <li>AI travel diary generation (subscription required)</li>
                        <li>AI place recommendations (subscription required)</li>
                        <li>Memory slideshow video creation</li>
                        <li>PDF travel book creation</li>
                        <li>Travel statistics analysis</li>
                    </ul>
                </li>
                <li>The Company may change the content of the Service, and important changes will be announced in advance.</li>
            </ul>

            <h2>Article 5 (Sign-up and Account)</h2>
            <ul>
                <li>Users sign up for the Service by logging in with a Google account.</li>
                <li>Users must provide accurate information and must not misappropriate others' information.</li>
                <li>Users are responsible for managing their accounts, and the Company is not liable for damages arising from unauthorized use of an account.</li>
            </ul>

            <h2>Article 6 (Subscription and Payment)</h2>
            <ul>
                <li>AI features (AI travel diary, AI place recommendations) are available through a monthly subscription.</li>
                <li>Payments are processed through the in-app purchase systems of the Google Play Store or Apple App Store.</li>
                <li>Subscriptions renew automatically each month unless cancelled.</li>
                <li>Subscriptions can be cancelled on each app store's subscription management page; after cancellation, the Service remains available until the end of the current billing cycle.</li>
                <li>Refunds follow each app store's refund policy.</li>
            </ul>

            <h2>Article 7 (User Obligations)</h2>
            <p>Users must not engage in the following:</p>
            <ul>
                <li>Misappropriating or fraudulently using others' personal information</li>
                <li>Illegal acts using the Service, or acts against public order and morals</li>
                <li>Defamation, insult, harassment, or threats against others</li>
                <li>Posting inappropriate content such as obscene material, violent content, or hate speech</li>
                <li>Acts that interfere with the stable operation of the Service</li>
                <li>Using the Service for commercial purposes without authorization</li>
            </ul>

            <h2>Article 8 (Management of Content)</h2>
            <ul>
                <li>Copyright of content registered by a user belongs to that user.</li>
                <li>Content a user sets to "public" may be exposed to other users.</li>
                <li>The Company may delete or restrict access to content that violates laws or these Terms, or is inappropriate, without prior notice.</li>
            </ul>

            <h2>Article 9 (Restriction and Termination of Use)</h2>
            <ul>
                <li>The Company may restrict use of the Service or suspend an account if the user violates these Terms.</li>
                <li>Users may delete their account at any time in the app settings; upon withdrawal, all data is deleted immediately and cannot be recovered.</li>
            </ul>

            <h2>Article 10 (Disclaimer)</h2>
            <ul>
                <li>The Company is not liable when it cannot provide the Service due to force majeure such as natural disasters, war, or service interruption by telecommunications carriers.</li>
                <li>The Company is not liable for service disruptions caused by the user's fault.</li>
                <li>Content generated through AI features is for reference only, and its accuracy is not guaranteed.</li>
            </ul>

            <h2>Article 11 (Protection of Personal Information)</h2>
            <p>
                The Company protects users' personal information in accordance with applicable laws and the Company's
                Privacy Policy. For details, please refer to the Privacy Policy in the app or on the website.
            </p>

            <h2>Article 12 (Dispute Resolution)</h2>
            <ul>
                <li>Disputes regarding these Terms are governed by the laws of the Republic of Korea.</li>
                <li>Disputes related to use of the Service are subject to the exclusive jurisdiction of the court having jurisdiction over the Company's location.</li>
            </ul>

            <h2>Addendum</h2>
            <p>These Terms take effect on April 20, 2026.</p>
        </div>
    );
};

export default RemapTermsEn;
