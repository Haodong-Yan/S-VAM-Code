function fallbackCopy(text) {
  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "fixed";
  textarea.style.top = "-9999px";
  document.body.appendChild(textarea);
  textarea.select();

  try {
    return document.execCommand("copy");
  } finally {
    document.body.removeChild(textarea);
  }
}

const copyButton = document.querySelector("[data-copy-target]");

if (copyButton) {
  const defaultLabel = copyButton.textContent;

  copyButton.addEventListener("click", async () => {
    const targetId = copyButton.getAttribute("data-copy-target");
    const target = targetId ? document.getElementById(targetId) : null;

    if (!target) {
      return;
    }

    const text = target.innerText.trim();

    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
      } else if (!fallbackCopy(text)) {
        throw new Error("Fallback copy failed");
      }

      copyButton.textContent = "Copied";
      copyButton.classList.add("is-copied");

      window.setTimeout(() => {
        copyButton.textContent = defaultLabel;
        copyButton.classList.remove("is-copied");
      }, 1600);
    } catch (error) {
      copyButton.textContent = "Copy failed";

      window.setTimeout(() => {
        copyButton.textContent = defaultLabel;
      }, 1600);
    }
  });
}
