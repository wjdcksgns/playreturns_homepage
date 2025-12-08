import MinHeightSection from '../../common/Sections/MinHeightSection';
import Container from '../../common/Layout/Container';
import LoginContents from './LoginContents';

const LoginSection = () => {
  return (
    <MinHeightSection>
      <Container>
        <LoginContents />
      </Container>
    </MinHeightSection>
  )
}

export default LoginSection;