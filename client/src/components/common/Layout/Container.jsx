const Container = ({children, isWide}) => {
  return (
    <div className={isWide ? 'container wide' : 'container'}>
      {children}
    </div>
  )
}

export default Container;