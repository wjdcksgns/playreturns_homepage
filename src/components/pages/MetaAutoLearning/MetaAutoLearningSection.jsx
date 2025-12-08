import MinHeightSection from '../../common/Sections/MinHeightSection';
import Container from '../../common/Layout/Container';
import MetaAutoLearningContents from './MetaAutoLearningContents';

const MetaAutoLearningSection = () => {
  return (
    <MinHeightSection>
      <Container isWide={true}>
        <MetaAutoLearningContents />
      </Container>
    </MinHeightSection>
  )
}

export default MetaAutoLearningSection;