// MathJax config for pymdownx.arithmatex (generic mode). Re-typesets on
// mkdocs-material's instant-navigation page swaps so math survives client-side
// nav. See https://squidfunk.github.io/mkdocs-material/reference/math/
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

document$.subscribe(() => {
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
