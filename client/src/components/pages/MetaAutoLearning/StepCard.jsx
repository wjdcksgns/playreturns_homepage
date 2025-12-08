import styles from './StepCard.module.css';

const StepCard = ({children, number}) => {
  return (
    <div className={styles.step}>
      <h4 className={styles.stepNumber}>step {number}.</h4>
      {children}
    </div>
  )
}

export default StepCard;