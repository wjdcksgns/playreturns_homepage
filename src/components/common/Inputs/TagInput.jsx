import Tags from "../Tags/Tags";
import InfoInputBox from "./InfoInputBox";

const TagInput = ({type, id, label, value, placeholder, maxLength, onChange, onKeyDown, info, tags}) => {
  return (
    <InfoInputBox id={id} label={label} info={info}>
      <Tags type={type} id={id} value={value} placeholder={placeholder} maxLength={maxLength} onChange={onChange} onKeyDown={onKeyDown} tags={tags} />
    </InfoInputBox>
  )
}

export default TagInput;