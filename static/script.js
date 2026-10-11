// Where the API lives. Empty = same server (Render). The Vercel copy sets the Render URL.
const API_BASE = '';

// --- Elements ----------------------------------------------------------------
const $ = (id) => document.getElementById(id);

const tabUpload = $('tab-upload');
const tabCamera = $('tab-camera');
const panelUpload = $('panel-upload');
const panelCamera = $('panel-camera');

const dropzone = $('dropzone');
const fileInput = $('file-input');
const emptyState = $('empty-state');
const previewImg = $('preview-img');
const removeBtn = $('remove-btn');

const cameraVideo = $('camera-video');
const cameraCanvas = $('camera-canvas');
const cameraPreview = $('camera-preview');
const cameraEmpty = $('camera-empty');
const cameraStartBtn = $('camera-start-btn');
const cameraCaptureBtn = $('camera-capture-btn');
const cameraRetakeBtn = $('camera-retake-btn');

const predictBtn = $('predict-btn');
const btnSpinner = $('btn-spinner');
const btnLabel = $('btn-label');

const resultBox = $('result');
const warningBanner = $('warning-banner');
const warningText = $('warning-text');
const verdict = $('verdict');
const topLabel = $('top-label');
const topPct = $('top-pct');
const probList = $('prob-list');
const moreBtn = $('more-btn');
const explainBtn = $('explain-btn');
const gradcamBox = $('gradcam-box');
const gradcamImg = $('gradcam-img');
const resetBtn = $('reset-btn');
const errorBox = $('error-box');

const VISIBLE_ROWS = 3;

let activeMode = 'upload';
let selectedBlob = null;
let mediaStream = null;

// --- Tabs ----------------------------------------------------------------------
function setMode(mode) {
  activeMode = mode;
  const isUpload = mode === 'upload';

  tabUpload.classList.toggle('is-active', isUpload);
  tabCamera.classList.toggle('is-active', !isUpload);
  tabUpload.setAttribute('aria-selected', String(isUpload));
  tabCamera.setAttribute('aria-selected', String(!isUpload));

  panelUpload.hidden = !isUpload;
  panelCamera.hidden = isUpload;

  clearSelection();
  resetCameraUI();
  hideResult();
  hideError();
}

tabUpload.addEventListener('click', () => setMode('upload'));
tabCamera.addEventListener('click', () => setMode('camera'));

// --- Upload --------------------------------------------------------------------
dropzone.setAttribute('tabindex', '0');
dropzone.setAttribute('role', 'button');
dropzone.setAttribute('aria-label', 'Choose a photo');

dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    fileInput.click();
  }
});

['dragenter', 'dragover'].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add('is-dragover');
  })
);

['dragleave', 'drop'].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove('is-dragover');
  })
);

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
  hideResult();
});

function handleBlob(blob) {
  selectedBlob = blob;
  hideError();
  hideResult();

  previewImg.src = URL.createObjectURL(blob);
  previewImg.hidden = false;
  emptyState.hidden = true;
  removeBtn.hidden = false;
  predictBtn.disabled = false;
}

function clearSelection() {
  selectedBlob = null;
  fileInput.value = '';
  previewImg.hidden = true;
  previewImg.removeAttribute('src');
  removeBtn.hidden = true;
  emptyState.hidden = false;
  predictBtn.disabled = true;
}

// --- Camera ----------------------------------------------------------------------
cameraStartBtn.addEventListener('click', startCamera);
cameraCaptureBtn.addEventListener('click', captureFrame);
cameraRetakeBtn.addEventListener('click', () => {
  resetCameraUI();
  hideResult();
});

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
    showError('Camera blocked. Allow camera access and try again.');
  }
}

function captureFrame() {
  const w = cameraVideo.videoWidth;
  const h = cameraVideo.videoHeight;
  cameraCanvas.width = w;
  cameraCanvas.height = h;
  cameraCanvas.getContext('2d').drawImage(cameraVideo, 0, 0, w, h);

  cameraCanvas.toBlob(
    (blob) => {
      selectedBlob = blob;
      cameraPreview.src = URL.createObjectURL(blob);
      cameraPreview.hidden = false;
      cameraVideo.hidden = true;
      cameraCaptureBtn.hidden = true;
      cameraRetakeBtn.hidden = false;
      predictBtn.disabled = false;
      hideResult();
    },
    'image/jpeg',
    0.92
  );

  stopCamera();
}

function resetCameraUI() {
  stopCamera();
  selectedBlob = activeMode === 'camera' ? null : selectedBlob;
  cameraVideo.hidden = true;
  cameraPreview.hidden = true;
  cameraPreview.removeAttribute('src');
  cameraCaptureBtn.hidden = true;
  cameraRetakeBtn.hidden = true;
  cameraStartBtn.hidden = false;
  cameraEmpty.hidden = false;
  if (activeMode === 'camera') predictBtn.disabled = true;
}

function stopCamera() {
  if (mediaStream) {
    mediaStream.getTracks().forEach((t) => t.stop());
    mediaStream = null;
  }
}

// --- Talking to the server ---------------------------------------------------------
async function send(path) {
  const formData = new FormData();
  formData.append('file', selectedBlob, 'photo.jpg');

  try {
    return await fetch(API_BASE + path, { method: 'POST', body: formData });
  } catch (err) {
    throw new Error("Can't reach the server. Try again.");
  }
}

async function readError(res) {
  try {
    const data = await res.json();
    if (data && typeof data.detail === 'string') return data.detail;
  } catch (err) { /* not JSON */ }
  return 'Something went wrong. Try again.';
}

predictBtn.addEventListener('click', async () => {
  if (!selectedBlob) return;

  setLoading(true);
  hideError();
  hideResult();

  try {
    const res = await send('/api/predict');
    if (!res.ok) throw new Error(await readError(res));
    showResult(await res.json());
  } catch (err) {
    showError(err.message);
  } finally {
    setLoading(false);
  }
});

explainBtn.addEventListener('click', async () => {
  if (!selectedBlob) return;

  explainBtn.disabled = true;
  explainBtn.textContent = 'Loading…';
  hideError();

  try {
    const res = await send('/api/explain');
    if (!res.ok) throw new Error(await readError(res));

    gradcamImg.src = URL.createObjectURL(await res.blob());
    gradcamBox.hidden = false;
  } catch (err) {
    showError(err.message);
  } finally {
    explainBtn.disabled = false;
    explainBtn.textContent = 'See why';
  }
});

resetBtn.addEventListener('click', () => {
  clearSelection();
  resetCameraUI();
  hideResult();
  hideError();
  window.scrollTo({ top: 0, behavior: 'smooth' });
});

moreBtn.addEventListener('click', () => {
  const open = probList.classList.toggle('is-open');
  moreBtn.textContent = open ? 'Show less' : 'Show all';
});

// --- Result ------------------------------------------------------------------------------
function setLoading(isLoading) {
  predictBtn.disabled = isLoading;
  btnSpinner.hidden = !isLoading;
  btnLabel.textContent = isLoading ? 'Sorting…' : 'Sort';
}

function showResult(data) {
  warningBanner.hidden = !data.warning;
  if (data.warning) warningText.textContent = 'Hard to see. Try better light.';

  verdict.dataset.bin = data.top.class_name;
  topLabel.textContent = data.top.display_name;
  topPct.textContent = Math.round(data.top.probability * 100) + '% sure';

  probList.innerHTML = '';
  probList.classList.remove('is-open');

  data.all.forEach((item, i) => {
    const pct = (item.probability * 100).toFixed(1);

    const row = document.createElement('div');
    row.className = 'prob-row' + (i === 0 ? ' is-top' : '') + (i >= VISIBLE_ROWS ? ' is-extra' : '');
    row.dataset.bin = item.class_name;

    const name = document.createElement('span');
    name.className = 'prob-row__name';
    name.textContent = item.display_name;

    const track = document.createElement('span');
    track.className = 'prob-row__track';
    const fill = document.createElement('span');
    fill.className = 'prob-row__fill';
    track.appendChild(fill);

    const value = document.createElement('span');
    value.className = 'prob-row__pct';
    value.textContent = pct + '%';

    row.append(name, track, value);
    probList.appendChild(row);
  });

  moreBtn.hidden = data.all.length <= VISIBLE_ROWS;
  moreBtn.textContent = 'Show all';

  resultBox.hidden = false;

  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      probList.querySelectorAll('.prob-row__fill').forEach((el, i) => {
        el.style.width = data.all[i].probability * 100 + '%';
      });
    });
  });

  resultBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
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
