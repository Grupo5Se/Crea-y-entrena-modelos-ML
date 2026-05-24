const API_BASE = 'http://127.0.0.1:8000';

const datasetBache = document.getElementById('datasetBache');
const datasetFisura = document.getElementById('datasetFisura');
const modelReady = document.getElementById('modelReady');
const accuracyText = document.getElementById('accuracyText');
const labelSelect = document.getElementById('labelSelect');
const trainFiles = document.getElementById('trainFiles');
const uploadBtn = document.getElementById('uploadBtn');
const trainBtn = document.getElementById('trainBtn');
const refreshBtn = document.getElementById('refreshBtn');
const logsBox = document.getElementById('logsBox');
const trainState = document.getElementById('trainState');

async function fetchJSON(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || 'Error de comunicación');
  return data;
}

async function loadDashboard() {
  try {
    const [stats, status, logs] = await Promise.all([
      fetchJSON(`${API_BASE}/api/stats`),
      fetchJSON(`${API_BASE}/api/train/status`),
      fetchJSON(`${API_BASE}/api/train/logs`),
    ]);

    datasetBache.textContent = stats.dataset_counts?.bache ?? 0;
    datasetFisura.textContent = stats.dataset_counts?.fisura ?? 0;
    modelReady.textContent = stats.has_model ? 'Sí' : 'No';
    accuracyText.textContent = stats.training_summary?.accuracy
      ? `${(stats.training_summary.accuracy * 100).toFixed(2)}%`
      : 'Sin entrenamiento';

    trainState.innerHTML = status.running
      ? '<span class="status-dot"></span>Entrenamiento en proceso'
      : '<span class="status-dot status-ok"></span>Sin procesos activos';
    logsBox.value = logs.logs || 'Aún no hay logs.';
  } catch (error) {
    trainState.innerHTML = `<span class="status-dot status-bad"></span>${error.message}`;
  }
}

async function uploadTrainingImages() {
  if (!trainFiles.files.length) {
    alert('Selecciona una o varias imágenes.');
    return;
  }

  const formData = new FormData();
  formData.append('label', labelSelect.value);
  Array.from(trainFiles.files).forEach(file => formData.append('files', file));

  uploadBtn.disabled = true;
  uploadBtn.textContent = 'Subiendo...';
  try {
    const data = await fetchJSON(`${API_BASE}/api/upload/training`, {
      method: 'POST',
      body: formData,
    });
    alert(`${data.saved_files.length} archivo(s) guardado(s) en la clase ${data.label}.`);
    trainFiles.value = '';
    await loadDashboard();
  } catch (error) {
    alert(error.message);
  } finally {
    uploadBtn.disabled = false;
    uploadBtn.textContent = 'Subir al dataset';
  }
}

async function startTraining() {
  trainBtn.disabled = true;
  trainBtn.textContent = 'Iniciando...';
  try {
    const data = await fetchJSON(`${API_BASE}/api/train/start`, { method: 'POST' });
    alert(data.message);
    await loadDashboard();
  } catch (error) {
    alert(error.message);
  } finally {
    trainBtn.disabled = false;
    trainBtn.textContent = 'Entrenar modelo';
  }
}

uploadBtn.addEventListener('click', uploadTrainingImages);
trainBtn.addEventListener('click', startTraining);
refreshBtn.addEventListener('click', loadDashboard);

loadDashboard();
setInterval(loadDashboard, 6000);
