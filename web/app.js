const $ = (selector) => document.querySelector(selector);
const fileInput = $("#fileInput");
const chooseButton = $("#chooseButton");
const demoButton = $("#demoButton");
const replaceButton = $("#replaceButton");
const dropZone = $("#dropZone");
const resultView = $("#resultView");
const sourceImage = $("#sourceImage");
const maskImage = $("#maskImage");
const lineCanvas = $("#lineCanvas");
const canvasWrap = $("#canvasWrap");
const analyzeButton = $("#analyzeButton");
const processing = $("#processing");
const errorMessage = $("#errorMessage");

let selectedFile = null;
let contours = [];
let lineColor = "#00ff88";

async function checkStatus() {
  try {
    const response = await fetch("/api/status");
    const data = await response.json();
    if (!data.ready) errorMessage.textContent = "Checkpoint модели не найден.";
  } catch {
    errorMessage.textContent = "Сервер недоступен.";
  }
}

function selectFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    errorMessage.textContent = "Выберите файл изображения.";
    return;
  }
  selectedFile = file;
  sourceImage.src = URL.createObjectURL(file);
  $("#fileName").textContent = file.name;
  dropZone.hidden = true;
  resultView.hidden = false;
  maskImage.removeAttribute("src");
  contours = [];
  analyzeButton.disabled = false;
  errorMessage.textContent = "";
  sourceImage.onload = syncOverlaySize;
}

function syncOverlaySize() {
  if (!sourceImage.naturalWidth || !sourceImage.naturalHeight) return;
  const wrapWidth = canvasWrap.clientWidth;
  const wrapHeight = canvasWrap.clientHeight;
  const scale = Math.min(
    wrapWidth / sourceImage.naturalWidth,
    wrapHeight / sourceImage.naturalHeight,
  );
  const width = Math.round(sourceImage.naturalWidth * scale);
  const height = Math.round(sourceImage.naturalHeight * scale);
  const left = Math.round((wrapWidth - width) / 2);
  const top = Math.round((wrapHeight - height) / 2);

  sourceImage.style.width = `${width}px`;
  sourceImage.style.height = `${height}px`;
  [maskImage, lineCanvas].forEach((layer) => {
    layer.style.left = `${left}px`;
    layer.style.top = `${top}px`;
    layer.style.width = `${width}px`;
    layer.style.height = `${height}px`;
  });
  lineCanvas.width = Math.max(1, Math.round(width * devicePixelRatio));
  lineCanvas.height = Math.max(1, Math.round(height * devicePixelRatio));
  drawContours();
}

function drawContours() {
  const context = lineCanvas.getContext("2d");
  context.clearRect(0, 0, lineCanvas.width, lineCanvas.height);
  if (!contours.length) return;
  context.strokeStyle = lineColor;
  context.lineWidth = 3 * devicePixelRatio;
  context.lineJoin = "round";
  context.lineCap = "round";
  context.shadowColor = lineColor;
  context.shadowBlur = 3 * devicePixelRatio;
  contours.forEach((points) => {
    if (points.length < 2) return;
    context.beginPath();
    context.moveTo(points[0][0] * lineCanvas.width, points[0][1] * lineCanvas.height);
    for (let index = 1; index < points.length; index += 1) {
      context.lineTo(points[index][0] * lineCanvas.width, points[index][1] * lineCanvas.height);
    }
    context.closePath();
    context.stroke();
  });
}

async function analyze() {
  if (!selectedFile) return;
  processing.hidden = false;
  analyzeButton.disabled = true;
  errorMessage.textContent = "";
  const body = new FormData();
  body.append("image", selectedFile);
  body.append("threshold", "0.5");
  try {
    const response = await fetch("/api/predict", { method: "POST", body });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Не удалось обработать изображение.");
    contours = data.contours;
    maskImage.src = data.mask_overlay;
    maskImage.style.opacity = 0.35;
    maskImage.style.display = "block";
    syncOverlaySize();
  } catch (error) {
    errorMessage.textContent = error.message;
  } finally {
    processing.hidden = true;
    analyzeButton.disabled = false;
  }
}

chooseButton.addEventListener("click", () => fileInput.click());
demoButton.addEventListener("click", async () => {
  demoButton.disabled = true;
  errorMessage.textContent = "";
  try {
    const response = await fetch("/api/demo");
    if (!response.ok) {
      const data = await response.json();
      throw new Error(data.detail || "Демо-снимок недоступен.");
    }
    const blob = await response.blob();
    const filename = response.headers.get("X-Demo-Filename") || "deepglobe_demo.jpg";
    selectFile(new File([blob], filename, { type: blob.type || "image/jpeg" }));
  } catch (error) {
    errorMessage.textContent = error.message;
  } finally {
    demoButton.disabled = false;
  }
});
replaceButton.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => selectFile(fileInput.files[0]));
analyzeButton.addEventListener("click", analyze);

["dragenter", "dragover"].forEach((eventName) => dropZone.addEventListener(eventName, (event) => {
  event.preventDefault();
  dropZone.classList.add("dragging");
}));
["dragleave", "drop"].forEach((eventName) => dropZone.addEventListener(eventName, (event) => {
  event.preventDefault();
  dropZone.classList.remove("dragging");
}));
dropZone.addEventListener("drop", (event) => selectFile(event.dataTransfer.files[0]));

window.addEventListener("resize", syncOverlaySize);
checkStatus();
