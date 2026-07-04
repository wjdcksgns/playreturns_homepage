import MainSection from '../../components/pages/Main/MainSection';
import SaveFlySpotlight from '../../components/pages/Main/SaveFlySpotlight';
import AppsSection from '../../components/pages/Main/AppsSection';
import SnuSection from '../../components/pages/Main/SnuSection';
import StatsSection from '../../components/pages/Main/StatsSection';
import WhyUsSection from '../../components/pages/Main/WhyUsSection';
import CtaSection from '../../components/pages/Main/CtaSection';
import { usePageTitle } from '../../common/hooks/usePageTitle';

const Main = () => {
  usePageTitle();

  return (
    <>
      <MainSection />
      <SaveFlySpotlight />
      <AppsSection />
      <SnuSection />
      <StatsSection />
      <WhyUsSection />
      <CtaSection />
    </>
  )
}

export default Main;