// --- Element references -----------------------------------------------
const tabUpload = document.getElementById('tab-upload');
const tabCamera = document.getElementById('tab-camera');
const panelUpload = document.getElementById('panel-upload');
const panelCamera = document.getElementById('panel-camera');

const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const emptyState = document.getElementById('empty-state');
const previewImg = document.getElementById('preview-img');
const removeBtn = document.getElementById('remove-btn');

const cameraVideo = document.getElementById('camera-video');
const cameraCanvas = document.getElementById('camera-canvas');
const cameraPreview = document.getElementById('camera-preview');
const cameraEmpty = document.getElementById('camera-empty');
const cameraStartBtn = document.getElementById('camera-start-btn');
const cameraCaptureBtn = document.getElementById('camera-capture-btn');
const cameraRetakeBtn = document.getElementById('camera-retake-btn');

const predictBtn = document.getElementById('predict-btn');
const btnSpinner = document.getElementById('btn-spinner');
const btnLabel = document.getElementById('btn-label');

const resultBox = document.getElementById('result');
const warningBanner = document.getElementById('warning-banner');
const warningText = document.getElementById('warning-text');
const topLabel = document.getElementById('top-label');
const topPct = document.getElementById('top-pct');
const probList = document.getElementById('prob-list');
const explainBtn = document.getElementById('explain-btn');
const gradcamBox = document.getElementById('gradcam-box');
const gradcamImg = document.getElementById('gradcam-img');
const resetBtn = document.getElementById('reset-btn');
const errorBox = document.getElementById('error-box');

let activeMode = 'upload'; // 'upload' | 'camera'
let selectedBlob = null;
let mediaStream = null;

// --- Tabs ----------------------------------------------------------------
function setMode(mode) {
  activeMode = mode;
  const isUpload = mode === 'upload';

  tabUpload.classList.toggle('is-active', isUpload);
  tabCamera.classList.toggle('is-active', !isUpload);
  tabUpload.setAttribute('aria-selected', isUpload);
  tabCamera.setAttribute('aria-selected', !isUpload);

  panelUpload.hidden = !isUpload;
  panelCamera.hidden = isUpload;

  clearSelection();
  hideResult();
  hideError();

  if (!isUpload === false) {
    stopCamera();
  }
}

tabUpload.addEventListener('click', () => setMode('upload'));
tabCamera.addEventListener('click', () => setMode('camera'));

// --- Upload panel ----------------------------------------------------------
dropzone.setAttribute('tabindex', '0');
dropzone.setAttribute('role', 'button');
dropzone.setAttribute('aria-label', 'Upload an image');

dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    fileInput.click();
  }
});

['dragenter', 'dragover'].forEach((evt) => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add('is-dragover');
  });
});

['dragleave', 'drop'].forEach((evt) => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove('is-dragover');
  });
});

dropzone.addEventListener('drop', (e) => {
  const file = e.dataTransfer.files[0];
  if (file) handleBlob(file);
});

fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (file) handleBlob(file);
});

removeBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  clearSelection();
});

function handleBlob(blob) {
  selectedBlob = blob;
  hideError();
  hideResult();

  const url = URL.createObjectURL(blob);
  previewImg.src = url;
  previewImg.hidden = false;
  emptyState.hidden = true;
  removeBtn.hidden = false;
  predictBtn.disabled = false;
}

function clearSelection() {
  selectedBlob = null;
  fileInput.value = '';
  previewImg.hidden = true;
  previewImg.src = '';
  removeBtn.hidden = true;
  emptyState.hidden = false;
  predictBtn.disabled = true;
}

// --- Camera panel ----------------------------------------------------------
cameraStartBtn.addEventListener('click', startCamera);
cameraCaptureBtn.addEventListener('click', captureFrame);
cameraRetakeBtn.addEventListener('click', retake);

async function startCamera() {
  hideError();
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment' },
    });
    cameraVideo.srcObject = mediaStream;
    cameraVideo.hidden = false;
    cameraEmpty.hidden = true;
    cameraStartBtn.hidden = true;
    cameraCaptureBtn.hidden = false;
  } catch (err) {
    showError('Could not access the camera. Check your browser permissions.');
  }
}

function captureFrame() {
  const w = cameraVideo.videoWidth;
  const h = cameraVideo.videoHeight;
  cameraCanvas.width = w;
  cameraCanvas.height = h;
  cameraCanvas.getContext('2d').drawImage(cameraVideo, 0, 0, w, h);

  cameraCanvas.toBlob((blob) => {
    selectedBlob = blob;
    cameraPreview.src = URL.createObjectURL(blob);
    cameraPreview.hidden = false;
    cameraVideo.hidden = true;
    cameraCaptureBtn.hidden = true;
    cameraRetakeBtn.hidden = false;
    predictBtn.disabled = false;
    hideResult();
  }, 'image/jpeg', 0.92);

  stopCamera();
}

function retake() {
  selectedBlob = null;
  cameraPreview.hidden = true;
  cameraRetakeBtn.hidden = true;
  cameraEmpty.hidden = false;
  cameraStartBtn.hidden = false;
  predictBtn.disabled = true;
  hideResult();
}

function stopCamera() {
  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }
}

// --- Predict ---------------------------------------------------------------
predictBtn.addEventListener('click', async () => {
  if (!selectedBlob) return;

  setLoading(true);
  hideError();
  hideResult();
  gradcamBox.hidden = true;

  try {
    const formData = new FormData();
    formData.append('file', selectedBlob, 'capture.jpg');

    const res = await fetch('/api/predict', { method: 'POST', body: formData });
    const data = await res.json();

    if (!res.ok) throw new Error(data.detail || 'Something went wrong.');

    showResult(data);
  } catch (err) {
    showError(err.message);
  } finally {
    setLoading(false);
  }
});

explainBtn.addEventListener('click', async () => {
  if (!selectedBlob) return;

  explainBtn.disabled = true;
  explainBtn.textContent = 'Loading...';

  try {
    const formData = new FormData();
    formData.append('file', selectedBlob, 'capture.jpg');

    const res = await fetch('/api/explain', { method: 'POST', body: formData });
    if (!res.ok) throw new Error('Could not generate explanation.');

    const blob = await res.blob();
    gradcamImg.src = URL.createObjectURL(blob);
    gradcamBox.hidden = false;
  } catch (err) {
    showError(err.message);
  } finally {
    explainBtn.disabled = false;
    explainBtn.textContent = 'Show what the model saw';
  }
});

resetBtn.addEventListener('click', () => {
  clearSelection();
  if (activeMode === 'camera') retake();
  hideResult();
});

function setLoading(isLoading) {
  predictBtn.disabled = isLoading;
  btnSpinner.hidden = !isLoading;
  btnLabel.textContent = isLoading ? 'Classifying' : 'Classify';
}

function showResult(data) {
  if (data.warning) {
    warningText.textContent = data.warning;
    warningBanner.hidden = false;
  } else {
    warningBanner.hidden = true;
  }

  const topPctValue = (data.top.probability * 100).toFixed(1);
  topLabel.textContent = data.top.display_name;
  topPct.textContent = topPctValue + '% confidence';

  probList.innerHTML = '';
  data.all.forEach((item) => {
    const pct = (item.probability * 100).toFixed(1);
    const isTop = item.class_name === data.top.class_name;

    const row = document.createElement('div');
    row.className = 'prob-row' + (isTop ? ' is-top' : '');
    row.innerHTML = `
      <span class="prob-row__name">${item.display_name}</span>
      <span class="prob-row__track"><span class="prob-row__fill" style="width:0%"></span></span>
      <span class="prob-row__pct">${pct}%</span>
    `;
    probList.appendChild(row);
  });

  resultBox.hidden = false;

  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      document.querySelectorAll('.prob-row__fill').forEach((el, i) => {
        el.style.width = (data.all[i].probability * 100) + '%';
      });
    });
  });
}

function hideResult() {
  resultBox.hidden = true;
  warningBanner.hidden = true;
  gradcamBox.hidden = true;
  probList.innerHTML = '';
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}

function hideError() {
  errorBox.hidden = true;
}
