// breakpoint max-width
export const breakpoint = {
  xSmall: 375,  /* 0 ~ 375 */
  small: 768,   /* 376 ~ 768 */
  medium: 1024, /* 769 ~ 1024 */
  large: 1440,  /* 1025 ~ 1440 */
  xLarge: 1920  /* 1441 ~ 1920 */
}

// 반응형 너비 확인
export const responsive = {
	xSmall: (width) => { /* 0 ~ 375 */
		return width <= breakpoint.xSmall;
	},
	small: (width) => { /* 376 ~ 768 */
		return width > breakpoint.xSmall && width <= breakpoint.small;
	},
	medium: (width) => { /* 769 ~ 1024 */
		return width > breakpoint.small && width <= breakpoint.medium;
	},
	large: (width) => { /* 1025 ~ 1440 */
		return width > breakpoint.medium && width <= breakpoint.large;
	},
	xLarge: (width) => { /* 1441 ~ 1920 */
		return width > breakpoint.large && width <= breakpoint.xLarge;
	},
	xxLarge: (width) => { /* 1921 ~ */
		return width > breakpoint.xLarge;
	},
	mobile: (width) => { /* 0 ~ 1024 */
		return responsive.xSmall(width) || responsive.small(width) || responsive.medium(width);
	},
	pc: (width) => { /* 1025 ~ */
		return responsive.large(width) || responsive.xLarge(width) || responsive.xxLarge(width);
	},
}

// vh css 변수
export const setVh = (height) => {
  let vh = height * 0.01;
  document.documentElement.style.setProperty("--vh", `${vh}px`);
}