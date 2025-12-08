import MinHeightSection from '../../common/Sections/MinHeightSection';
import Container from '../../common/Layout/Container';
import JoinContents from './JoinContents';

const JoinSection = () => {
  return (
    <MinHeightSection>
      <Container>
        <JoinContents />
      </Container>
    </MinHeightSection>
  )
}

export default JoinSection;