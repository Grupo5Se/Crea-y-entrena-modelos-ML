const API_BASE = 'http://127.0.0.1:8000';

const apiStatus = document.getElementById('apiStatus');
const statBache = document.getElementById('statBache');
const statFisura = document.getElementById('statFisura');
const statModel = document.getElementById('statModel');
const fileInput = document.getElementById('fileInput');
const dropzone = document.getElementById('dropzone');
const preview = document.getElementById('preview');
const analyzeBtn = document.getElementById('analyzeBtn');
const clearBtn = document.getElementById('clearBtn');
const resultLabel = document.getElementById('resultLabel');
const confidence = document.getElementById('confidence');
const scoreList = document.getElementById('scoreList');

let selectedFile = null;

async function loadStats() {
  try {
    const health = await fetch(`${API_BASE}/api/health`).then(r => r.json());
    apiStatus.innerHTML = `<span class="status-dot status-ok"></span>API activa · ${health.time}`;
  } catch (error) {
    apiStatus.innerHTML = `<span class="status-dot status-bad"></span>API no disponible`;
  }

  try {
    const stats = await fetch(`${API_BASE}/api/stats`).then(r => r.json());
    statBache.textContent = stats.dataset_counts?.bache ?? 0;
    statFisura.textContent = stats.dataset_counts?.fisura ?? 0;
    statModel.textContent = stats.has_model ? 'Sí' : 'No';
  } catch (error) {
    statBache.textContent = '-';
    statFisura.textContent = '-';
    statModel.textContent = '-';
  }
}

function setPreview(file) {
  if (!file) return;
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    preview.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

function bindDropzone() {
  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, e => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, e => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
    });
  });

  dropzone.addEventListener('drop', e => {
    const file = e.dataTransfer.files?.[0];
    if (file) {
      fileInput.files = e.dataTransfer.files;
      setPreview(file);
    }
  });

  fileInput.addEventListener('change', e => {
    const file = e.target.files?.[0];
    if (file) setPreview(file);
  });
}

async function analyzeImage() {
  if (!selectedFile) {
    alert('Selecciona una imagen primero.');
    return;
  }

  const formData = new FormData();
  formData.append('file', selectedFile);

  analyzeBtn.disabled = true;
  analyzeBtn.textContent = 'Analizando...';
  resultLabel.textContent = 'Procesando';
  confidence.textContent = 'Esperando respuesta del modelo';
  scoreList.innerHTML = '';

  try {
    const response = await fetch(`${API_BASE}/api/predict`, {
      method: 'POST',
      body: formData,
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'No se pudo analizar la imagen.');

    resultLabel.textContent = data.label.toUpperCase();
    confidence.textContent = `Confianza: ${(data.confidence * 100).toFixed(2)}%`;
    scoreList.innerHTML = Object.entries(data.scores)
      .map(([label, value]) => `<li><span>${label}</span><strong>${(value * 100).toFixed(2)}%</strong></li>`)
      .join('');
  } catch (error) {
    resultLabel.textContent = 'Error';
    confidence.textContent = error.message;
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = 'Analizar imagen';
  }
}

function clearAll() {
  selectedFile = null;
  fileInput.value = '';
  preview.src = 'https://placehold.co/900x520/0f172a/e2e8f0?text=Vista+previa+de+la+imagen';
  resultLabel.textContent = 'Sin análisis';
  confidence.textContent = 'Carga una imagen y pulsa Analizar';
  scoreList.innerHTML = '';
}

analyzeBtn.addEventListener('click', analyzeImage);
clearBtn.addEventListener('click', clearAll);

bindDropzone();
loadStats();
setInterval(loadStats, 12000);
