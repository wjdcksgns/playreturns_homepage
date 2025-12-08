import { useEffect } from "react";
import { useLocation } from "react-router-dom";

function ScrollToTop() {
    const { pathname } = useLocation();

    useEffect(() => {
        window.scrollTo(0, 0); // 페이지 이동 시 맨 위로
    }, [pathname]);

    return null;
}

export default ScrollToTop;
