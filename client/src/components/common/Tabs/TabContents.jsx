import ArticleBox from '../Boxes/ArticleBox';

const TabContents = ({id, name, children}) => {
  return (
    <ArticleBox size="md" key={id} data-tab-name={name}>
      {children}
    </ArticleBox>
  )
}

export default TabContents;