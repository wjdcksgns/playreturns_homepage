import MainSection from '../../components/pages/Main/MainSection';
import AppsSection from '../../components/pages/Main/AppsSection';
import StatsSection from '../../components/pages/Main/StatsSection';
import WhyUsSection from '../../components/pages/Main/WhyUsSection';
import CtaSection from '../../components/pages/Main/CtaSection';
import { usePageTitle } from '../../common/hooks/usePageTitle';

const Main = () => {
  usePageTitle();

  return (
    <>
      <MainSection />
      <AppsSection />
      <StatsSection />
      <WhyUsSection />
      <CtaSection />
    </>
  )
}

export default Main;