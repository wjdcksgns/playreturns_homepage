import IntroSection from "./IntroSection";
import MinHeightSection from "./MinHeightSection";

const SubWrap = ({children, name, title, isWide}) => {
  return (
    <main>
      <IntroSection section={name} title={title} />
      <MinHeightSection containerClassName={isWide ? 'container wide' : 'container'}>{children}</MinHeightSection>
    </main>
  )
}

export default SubWrap;