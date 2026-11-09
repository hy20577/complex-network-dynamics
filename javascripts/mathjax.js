// MathJax settings for the whole site. Notebook pages are typeset by
// mkdocs-jupyter with these settings, including $...$ written inside raw HTML
// in text cells; code cells are <pre> blocks, which MathJax always skips.
// The script below typesets the maths of the Markdown pages (arithmatex).
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true
  },
  startup: {
    typeset: false, // Typeset below, so the two passes never overlap
    pageReady: () => MathJax.startup.defaultPageReady().then(typesetMarkdownMath)
  }
};

function typesetMarkdownMath() {
  const areas = [...document.querySelectorAll(".arithmatex")]
    .filter(el => !el.closest(".jupyter-wrapper"))
  if (!areas.length) return Promise.resolve()
  MathJax.typesetClear(areas)
  MathJax.texReset()
  return MathJax.typesetPromise(areas)
}

// Pages loaded later without a full reload; before MathJax has loaded,
// pageReady above does the first typesetting
document$.subscribe(() => {
  if (typeof MathJax.typesetPromise === "function") typesetMarkdownMath()
})
