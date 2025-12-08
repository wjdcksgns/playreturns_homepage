import MinHeightSection from '../../common/Sections/MinHeightSection';
import Container from '../../common/Layout/Container';
import MyListContents from './MyListContents';

const MyListSection = () => {
  return (
    <MinHeightSection>
      <Container>
        <MyListContents />
      </Container>
    </MinHeightSection>
  )
}

export default MyListSection;