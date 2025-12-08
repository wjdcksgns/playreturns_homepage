import InfoInputBox from './InfoInputBox';
import Input from './Input';

const InfoInput = ({type, id, label, value, placeholder, maxLength, onChange, info}) => {
  return (
    <InfoInputBox id={id} label={label} info={info}>
      <Input type={type} id={id} value={value} placeholder={placeholder} maxLength={maxLength} onChange={onChange} />
    </InfoInputBox>
  )
}

export default InfoInput;