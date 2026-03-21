/* Load Mermaid from CDN and initialize after DOM is ready */
(function () {
  function runMermaid() {
    if (typeof mermaid === "undefined") return;
    mermaid.initialize({
      startOnLoad: false,
      theme: document.body.getAttribute("data-md-color-scheme") === "slate"
        ? "dark"
        : "default",
      flowchart: { useMaxWidth: true, htmlLabels: true },
      sequence:  { useMaxWidth: true },
    });
    mermaid.run({ querySelector: ".mermaid" });
  }

  var script = document.createElement("script");
  script.src = "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js";
  script.onload = runMermaid;
  document.head.appendChild(script);

  /* Re-run on Material instant navigation (SPA page transitions) */
  document.addEventListener("DOMContentSwitch", runMermaid);
})();
