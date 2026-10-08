/* ═══════════════════════════════════════════════════════════════
   AudioScript — Frontend Application Logic
   ═══════════════════════════════════════════════════════════════ */

'use strict';

// ── DOM refs ──────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

const dropZone         = $('drop-zone');
const fileInput        = $('file-input');
const browseBtn        = $('browse-btn');
const filePreview      = $('file-preview');
const fileName         = $('file-name');
const fileSize         = $('file-size');
const removeFileBtn    = $('remove-file-btn');
const languageSelect   = $('language-select');
const taskSelect       = $('task-select');
const transcribeBtn    = $('transcribe-btn');
const uploadCard       = $('upload-card');
const progressCard     = $('progress-card');
const progressSub      = $('progress-sub');
const progressBar      = $('progress-bar');
const resultsCard      = $('results-card');
const resultsMeta      = $('results-meta');
const errorCard        = $('error-card');
const errorMessage     = $('error-message');
const retryBtn         = $('retry-btn');
const newTranscriptionBtn = $('new-transcription-btn');
const modelBadge       = $('model-badge');

// Tabs
const tabTranscript = $('tab-transcript');
const tabSegments   = $('tab-segments');
const tabJson       = $('tab-json');
const panelTranscript = $('panel-transcript');
const panelSegments   = $('panel-segments');
const panelJson       = $('panel-json');

// Actions
const copyTextBtn    = $('copy-text-btn');
const downloadTxtBtn = $('download-txt-btn');
const downloadJsonBtn = $('download-json-btn');
const segmentSearch  = $('segment-search');
const segmentsList   = $('segments-list');
const transcriptText = $('transcript-text');
const jsonView       = $('json-view');

// ── State ─────────────────────────────────────────────────────────────────────
let selectedFile = null;
let lastResult   = null;

// ── Health check on load ──────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const res = await fetch('/health');
    if (!res.ok) throw new Error('not ok');
    const data = await res.json();
    if (data.model_loaded) {
      modelBadge.textContent = `Whisper ${data.whisper_model} ✓`;
      modelBadge.classList.add('ready');
    } else {
      modelBadge.textContent = 'Model loading…';
      // Retry after 3 seconds
      setTimeout(checkHealth, 3000);
    }
  } catch {
    modelBadge.textContent = 'Service unavailable';
  }
}
checkHealth();

// ── File selection ────────────────────────────────────────────────────────────
function formatBytes(bytes) {
  if (bytes < 1024)          return `${bytes} B`;
  if (bytes < 1024 * 1024)   return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function setFile(file) {
  if (!file) return;
  selectedFile = file;
  fileName.textContent = file.name;
  fileSize.textContent = formatBytes(file.size);
  filePreview.classList.remove('hidden');
  transcribeBtn.disabled = false;
  transcribeBtn.setAttribute('aria-disabled', 'false');
  dropZone.classList.add('has-file');
}

function clearFile() {
  selectedFile = null;
  fileInput.value = '';
  filePreview.classList.add('hidden');
  transcribeBtn.disabled = true;
  transcribeBtn.setAttribute('aria-disabled', 'true');
  dropZone.classList.remove('has-file');
}

browseBtn.addEventListener('click', e => { e.stopPropagation(); fileInput.click(); });
dropZone.addEventListener('click', () => { if (!selectedFile) fileInput.click(); });
dropZone.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') fileInput.click(); });
fileInput.addEventListener('change', () => setFile(fileInput.files[0]));
removeFileBtn.addEventListener('click', clearFile);

// ── Drag & Drop ───────────────────────────────────────────────────────────────
['dragenter', 'dragover'].forEach(evt =>
  dropZone.addEventListener(evt, e => { e.preventDefault(); dropZone.classList.add('drag-over'); })
);
['dragleave', 'drop'].forEach(evt =>
  dropZone.addEventListener(evt, e => { e.preventDefault(); dropZone.classList.remove('drag-over'); })
);
dropZone.addEventListener('drop', e => {
  const file = e.dataTransfer.files[0];
  if (file) setFile(file);
});

// ── Tab switching ─────────────────────────────────────────────────────────────
function switchTab(activeTab, activePanel) {
  [tabTranscript, tabSegments, tabJson].forEach(t => {
    t.classList.remove('active');
    t.setAttribute('aria-selected', 'false');
  });
  [panelTranscript, panelSegments, panelJson].forEach(p => p.classList.add('hidden'));
  activeTab.classList.add('active');
  activeTab.setAttribute('aria-selected', 'true');
  activePanel.classList.remove('hidden');
}
tabTranscript.addEventListener('click', () => switchTab(tabTranscript, panelTranscript));
tabSegments.addEventListener('click',   () => switchTab(tabSegments,   panelSegments));
tabJson.addEventListener('click',       () => switchTab(tabJson,       panelJson));

// ── Segment search ────────────────────────────────────────────────────────────
segmentSearch.addEventListener('input', () => {
  const q = segmentSearch.value.toLowerCase().trim();
  document.querySelectorAll('.segment-item').forEach(item => {
    const text = item.querySelector('.segment-text').textContent.toLowerCase();
    item.classList.toggle('hidden-search', q !== '' && !text.includes(q));
  });
});

// ── Progress simulation ───────────────────────────────────────────────────────
const PROGRESS_MESSAGES = [
  'Validating audio file…',
  'Converting to 16 kHz mono PCM…',
  'Detecting language…',
  'Running Whisper transcription…',
  'Processing timestamps…',
  'Merging segments…',
  'Finalising output…',
];
let progressTimer = null;
let progressIdx   = 0;

function startProgress() {
  progressIdx = 0;
  progressSub.textContent = PROGRESS_MESSAGES[0];
  progressTimer = setInterval(() => {
    progressIdx = (progressIdx + 1) % PROGRESS_MESSAGES.length;
    progressSub.textContent = PROGRESS_MESSAGES[progressIdx];
  }, 2500);
}
function stopProgress() { clearInterval(progressTimer); }

// ── Render results ────────────────────────────────────────────────────────────
function formatTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = (seconds % 60).toFixed(2).padStart(5, '0');
  return `${String(m).padStart(2, '0')}:${s}`;
}

function renderMeta(data) {
  const chips = [
    { label: 'Language', val: data.language.toUpperCase() },
    { label: 'Duration',  val: `${formatTime(data.duration_s)}` },
    { label: 'Segments',  val: data.segments.length },
    { label: 'Model',     val: `whisper-${data.model}` },
    { label: 'Processed', val: `${data.processing_time_s}s` },
    { label: 'Chunks',    val: data.chunks_processed },
    { label: 'File size', val: `${data.audio_meta.size_mb} MB` },
  ];
  resultsMeta.innerHTML = chips.map(c =>
    `<div class="meta-chip"><span class="label">${c.label}</span><span class="val">${c.val}</span></div>`
  ).join('');
}

function renderSegments(segments) {
  segmentsList.innerHTML = '';
  segments.forEach((seg, i) => {
    const conf   = Math.min(1, Math.max(0, seg.confidence));
    const confPct = Math.round(conf * 100);
    const el = document.createElement('div');
    el.className = 'segment-item';
    el.setAttribute('aria-label', `Segment ${i + 1}: ${seg.text}`);
    el.innerHTML = `
      <div class="segment-time">${formatTime(seg.start)} → ${formatTime(seg.end)}</div>
      <div>
        <div class="segment-text">${escapeHtml(seg.text)}</div>
        <div class="segment-confidence">
          <div class="conf-bar-wrap"><div class="conf-bar" style="width:${confPct}%"></div></div>
          <span class="conf-label">${confPct}%</span>
        </div>
      </div>`;
    el.style.animationDelay = `${i * 30}ms`;
    segmentsList.appendChild(el);
  });
}

function escapeHtml(str) {
  return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function showResults(data) {
  lastResult = data;
  renderMeta(data);
  transcriptText.textContent = data.text;
  renderSegments(data.segments);
  jsonView.textContent = JSON.stringify(data, null, 2);
  // Reset tabs
  switchTab(tabTranscript, panelTranscript);
  progressCard.classList.add('hidden');
  uploadCard.classList.add('hidden');
  resultsCard.classList.remove('hidden');
}

function showError(msg) {
  errorMessage.textContent = msg;
  progressCard.classList.add('hidden');
  errorCard.classList.remove('hidden');
}

// ── Transcription request ─────────────────────────────────────────────────────
transcribeBtn.addEventListener('click', async () => {
  if (!selectedFile) return;

  // Show progress
  uploadCard.classList.add('hidden');
  errorCard.classList.add('hidden');
  resultsCard.classList.add('hidden');
  progressCard.classList.remove('hidden');
  startProgress();

  const formData = new FormData();
  formData.append('file', selectedFile);
  formData.append('language', languageSelect.value);
  formData.append('task', taskSelect.value);

  try {
    const res = await fetch('/transcribe', {
      method: 'POST',
      body: formData,
    });

    stopProgress();

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      showError(err.detail || `Server error ${res.status}`);
      return;
    }

    const data = await res.json();
    showResults(data);
  } catch (err) {
    stopProgress();
    showError(`Network error: ${err.message}. Is the server running?`);
  }
});

// ── Reset / retry ─────────────────────────────────────────────────────────────
function resetToUpload() {
  resultsCard.classList.add('hidden');
  errorCard.classList.add('hidden');
  progressCard.classList.add('hidden');
  uploadCard.classList.remove('hidden');
  clearFile();
  lastResult = null;
}
newTranscriptionBtn.addEventListener('click', resetToUpload);
retryBtn.addEventListener('click', resetToUpload);

// ── Copy ──────────────────────────────────────────────────────────────────────
copyTextBtn.addEventListener('click', async () => {
  if (!lastResult) return;
  try {
    await navigator.clipboard.writeText(lastResult.text);
    copyTextBtn.classList.add('copied');
    copyTextBtn.querySelector('svg').style.display = 'none';
    const origText = copyTextBtn.childNodes[copyTextBtn.childNodes.length - 1].textContent;
    copyTextBtn.childNodes[copyTextBtn.childNodes.length - 1].textContent = ' Copied!';
    setTimeout(() => {
      copyTextBtn.classList.remove('copied');
      copyTextBtn.querySelector('svg').style.display = '';
      copyTextBtn.childNodes[copyTextBtn.childNodes.length - 1].textContent = origText;
    }, 2000);
  } catch {
    alert('Copy failed. Please select and copy the text manually.');
  }
});

// ── Downloads ─────────────────────────────────────────────────────────────────
function downloadFile(content, filename, mimeType) {
  const blob = new Blob([content], { type: mimeType });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

downloadTxtBtn.addEventListener('click', () => {
  if (!lastResult) return;
  const baseName = lastResult.file.replace(/\.[^.]+$/, '');
  const lines = lastResult.segments.map(
    s => `[${formatTime(s.start)} → ${formatTime(s.end)}]\n${s.text}\n`
  ).join('\n');
  const content = `Transcription: ${lastResult.file}\nLanguage: ${lastResult.language}\nDate: ${new Date().toISOString()}\n${'─'.repeat(60)}\n\n${lines}\n${'─'.repeat(60)}\n\nFull Transcript:\n${lastResult.text}`;
  downloadFile(content, `${baseName}_transcript.txt`, 'text/plain');
});

downloadJsonBtn.addEventListener('click', () => {
  if (!lastResult) return;
  const baseName = lastResult.file.replace(/\.[^.]+$/, '');
  downloadFile(JSON.stringify(lastResult, null, 2), `${baseName}_transcript.json`, 'application/json');
});
