/* ═══════════════════════════════════════════════════════════
   app.js — EduPredict AI  SPA Controller
   ═══════════════════════════════════════════════════════════ */

const API = '';   // same origin — Flask serves everything

/* ─── Chart colour palette (matches CSS) ─────────────────── */
const COLORS = [
  '#6C63FF', '#43D8C9', '#FFD166', '#FF6584', '#06D6A0', '#EF476F',
];

/* ─── Chart.js global defaults ───────────────────────────── */
Chart.defaults.color          = '#9A9ABF';
Chart.defaults.borderColor    = 'rgba(255,255,255,0.06)';
Chart.defaults.font.family    = "'Inter', sans-serif";
Chart.defaults.plugins.legend.labels.color = '#9A9ABF';

/* ───────────────────────────────────────────────────────────
   NAVIGATION
   ─────────────────────────────────────────────────────────── */
function navigateTo(page) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
  document.getElementById(`page-${page}`).classList.add('active');
  document.getElementById(`nav-${page}`).classList.add('active');
  window.scrollTo({ top: 0, behavior: 'smooth' });

  if (page === 'analytics') loadAnalytics();
}

document.querySelectorAll('.nav-link').forEach(link => {
  link.addEventListener('click', e => {
    e.preventDefault();
    navigateTo(link.dataset.page);
  });
});

/* ───────────────────────────────────────────────────────────
   TOAST
   ─────────────────────────────────────────────────────────── */
function showToast(msg, type = 'info', duration = 4000) {
  const icons = { success: '✅', error: '❌', info: 'ℹ️' };
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.innerHTML = `<span>${icons[type]}</span><span>${msg}</span>`;
  document.getElementById('toast-container').appendChild(el);
  setTimeout(() => el.remove(), duration);
}

/* ───────────────────────────────────────────────────────────
   HEALTH CHECK & HOME STATS
   ─────────────────────────────────────────────────────────── */
async function checkHealth() {
  try {
    const res  = await fetch(`${API}/api/health`);
    const data = await res.json();
    const dot  = document.querySelector('.status-dot');
    const txt  = document.querySelector('.status-text');

    if (data.models_ready) {
      dot.className  = 'status-dot ok';
      txt.textContent = 'Models Ready';
    } else {
      dot.className  = 'status-dot error';
      txt.textContent = 'Run train.py';
      showToast('Models not trained yet. Please run python train.py first.', 'error', 8000);
    }
  } catch {
    document.querySelector('.status-dot').className = 'status-dot error';
    document.querySelector('.status-text').textContent = 'Server offline';
    showToast('Cannot connect to Flask server. Is it running?', 'error', 8000);
  }
}

async function loadHomeStats() {
  try {
    const [statsRes, modelsRes] = await Promise.all([
      fetch(`${API}/api/stats`),
      fetch(`${API}/api/models`),
    ]);
    const stats  = await statsRes.json();
    const models = await modelsRes.json();

    if (!stats.error) {
      document.getElementById('stat-total').textContent = stats.total_students?.toLocaleString() ?? '—';
      document.getElementById('stat-avg').textContent   = `${stats.avg_overall ?? '—'}`;
      document.getElementById('stat-pass').textContent  = `${stats.pass_rate ?? '—'}%`;
      document.getElementById('home-students').textContent = stats.total_students?.toLocaleString() ?? '—';
    }

    if (!models.error) {
      const bestReg = models.best_regressor;
      const bestClf = models.best_classifier;
      document.getElementById('home-best-reg').textContent = shortName(bestReg);
      document.getElementById('home-best-clf').textContent = shortName(bestClf);
      const r2  = models.regression?.[bestReg]?.R2;
      const acc = models.classification?.[bestClf]?.Accuracy;
      document.getElementById('home-reg-r2').textContent = r2  != null ? `R² = ${r2}` : '';
      document.getElementById('home-clf-acc').textContent = acc != null ? `Acc = ${(acc*100).toFixed(1)}%` : '';
    }
  } catch { /* silent */ }
}

/* ───────────────────────────────────────────────────────────
   PREDICTION FORM
   ─────────────────────────────────────────────────────────── */
async function handlePredict(e) {
  e.preventDefault();
  const form = e.target;
  const btn  = document.getElementById('predict-btn');
  const btnTxt = document.getElementById('predict-btn-text');

  const payload = {
    gender:                     form.gender.value,
    parental_education:         form.parental_education.value,
    study_hours:                form.study_hours.value,
    attendance_rate:            form.attendance_rate.value,
    past_exam_score:            form.past_exam_score.value,
    internet_access:            form.internet_access.value,
    extracurricular_activities: form.extracurricular_activities.value,
  };

  // Validate
  for (const [k,v] of Object.entries(payload)) {
    if (!v) { showToast(`Please fill in: ${k.replace('_',' ')}`, 'error'); return; }
  }

  // Loading state
  btn.disabled = true;
  btnTxt.textContent = 'Predicting…';

  try {
    const res  = await fetch(`${API}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    if (!data.success) throw new Error(data.error || 'Prediction failed');

    renderPredictionResult(data.prediction);
    showToast('Prediction complete! ✨', 'success');

    // Fetch AI insights
    fetchInsights(payload, data.prediction);

  } catch (err) {
    showToast(err.message, 'error');
  } finally {
    btn.disabled = false;
    btnTxt.textContent = 'Run Prediction';
  }
}

/* ─── Render prediction result ───────────────────────────── */
function renderPredictionResult(pred) {
  document.getElementById('predict-results').style.display = 'flex';
  document.getElementById('predict-results').style.flexDirection = 'column';

  // Grade
  const grade = pred.grade_prediction;
  document.getElementById('result-grade').textContent = grade.toFixed(1);
  document.getElementById('letter-badge').textContent = pred.letter_grade;

  // Pass/Fail badge
  const pb = document.getElementById('pass-badge');
  pb.textContent  = pred.pass_fail === 1 ? '✅ PASS' : '❌ FAIL';
  pb.className    = `pass-badge ${pred.pass_fail === 1 ? 'pass' : 'fail'}`;

  // Pass probability bar
  const pp = pred.pass_probability ?? 0;
  document.getElementById('pass-prob-val').textContent = `${pp}%`;
  setTimeout(() => {
    document.getElementById('pass-prob-bar').style.width = `${pp}%`;
  }, 100);

  // Best model names
  document.getElementById('result-best-reg').textContent = pred.best_regressor || '—';
  document.getElementById('result-best-clf').textContent = pred.best_classifier || '—';

  // All model predictions table
  const tbody = document.getElementById('models-tbody');
  tbody.innerHTML = '';
  const regPreds = pred.all_reg_predictions || {};
  const clfPreds = pred.all_clf_predictions || {};
  const allModels = new Set([...Object.keys(regPreds), ...Object.keys(clfPreds)]);

  for (const name of allModels) {
    const regVal = regPreds[name] != null ? `${regPreds[name].toFixed(1)}/100` : '—';
    const clfVal = clfPreds[name] != null
      ? `<span class="${clfPreds[name] === 1 ? 'td-pass' : 'td-fail'}">${clfPreds[name] === 1 ? 'Pass' : 'Fail'}</span>`
      : '—';
    tbody.innerHTML += `
      <tr>
        <td class="td-model">${cleanName(name)}</td>
        <td>${regVal}</td>
        <td>${clfVal}</td>
      </tr>`;
  }

  // Scroll to results
  document.getElementById('predict-results').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/* ─── Fetch AI insights ──────────────────────────────────── */
async function fetchInsights(studentData, prediction) {
  const loadEl    = document.getElementById('insights-loading');
  const contentEl = document.getElementById('insights-content');

  loadEl.style.display    = 'flex';
  contentEl.style.display = 'none';

  try {
    const res  = await fetch(`${API}/api/insights`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ student_data: studentData, prediction }),
    });
    const data = await res.json();

    if (data.success) {
      contentEl.innerHTML = marked.parse(data.insights);
      loadEl.style.display    = 'none';
      contentEl.style.display = 'block';
    } else {
      throw new Error(data.error);
    }
  } catch (err) {
    loadEl.innerHTML = `<span style="color:var(--red)">⚠️ ${err.message}</span>`;
  }
}

/* ───────────────────────────────────────────────────────────
   ANALYTICS — METRICS TABLES + CHARTS
   ─────────────────────────────────────────────────────────── */
let analyticsLoaded = false;
const chartInstances = {};

async function loadAnalytics() {
  if (analyticsLoaded) return;
  analyticsLoaded = true;

  try {
    const res    = await fetch(`${API}/api/models`);
    const models = await res.json();

    if (models.error) {
      showToast(models.error, 'error');
      return;
    }

    buildRegTable(models.regression,     models.best_regressor);
    buildClfTable(models.classification, models.best_classifier);
    buildRegCharts(models.regression);
    buildClfCharts(models.classification);

  } catch (err) {
    showToast('Failed to load model metrics: ' + err.message, 'error');
  }

  // Load plot images
  loadPlots();
}

/* ── Regression table ─────────────────────────────────────── */
function buildRegTable(data, best) {
  const tbody = document.getElementById('reg-tbody');
  const sorted = Object.entries(data).sort((a,b) => b[1].R2 - a[1].R2);
  tbody.innerHTML = '';
  sorted.forEach(([name, m], i) => {
    const isBest = name === best;
    const rankEl = i === 0 ? '🥇' : i === 1 ? '🥈' : i === 2 ? '🥉' : `#${i+1}`;
    tbody.innerHTML += `
      <tr class="${isBest ? 'best-row' : ''}">
        <td class="td-model">${cleanName(name)} ${isBest ? '⭐' : ''}</td>
        <td class="metric-val rank-${i+1}">${m.R2}</td>
        <td class="metric-val">${m.RMSE}</td>
        <td class="metric-val">${m.MAE}</td>
        <td class="metric-val">${m.CV_R2}</td>
        <td class="rank-${i+1}">${rankEl}</td>
      </tr>`;
  });
}

/* ── Classification table ─────────────────────────────────── */
function buildClfTable(data, best) {
  const tbody = document.getElementById('clf-tbody');
  const sorted = Object.entries(data).sort((a,b) => b[1].Accuracy - a[1].Accuracy);
  tbody.innerHTML = '';
  sorted.forEach(([name, m], i) => {
    const isBest = name === best;
    const rankEl = i === 0 ? '🥇' : i === 1 ? '🥈' : i === 2 ? '🥉' : `#${i+1}`;
    tbody.innerHTML += `
      <tr class="${isBest ? 'best-row' : ''}">
        <td class="td-model">${cleanName(name)} ${isBest ? '⭐' : ''}</td>
        <td class="metric-val rank-${i+1}">${(m.Accuracy*100).toFixed(2)}%</td>
        <td class="metric-val">${m.F1.toFixed(4)}</td>
        <td class="metric-val">${(m.CV_Acc*100).toFixed(2)}%</td>
        <td class="rank-${i+1}">${rankEl}</td>
      </tr>`;
  });
}

/* ── Regression charts ────────────────────────────────────── */
function buildRegCharts(data) {
  const names  = Object.keys(data);
  const short  = names.map(shortName);
  const r2s    = names.map(n => data[n].R2);
  const rmses  = names.map(n => data[n].RMSE);

  destroyChart('chart-reg-r2');
  destroyChart('chart-reg-rmse');

  chartInstances['chart-reg-r2'] = new Chart(
    document.getElementById('chart-reg-r2'),
    barConfig(short, r2s, 'R² Score', 'R²')
  );
  chartInstances['chart-reg-rmse'] = new Chart(
    document.getElementById('chart-reg-rmse'),
    barConfig(short, rmses, 'RMSE', 'RMSE', true)
  );
}

/* ── Classification charts ────────────────────────────────── */
function buildClfCharts(data) {
  const names = Object.keys(data);
  const short = names.map(shortName);
  const accs  = names.map(n => data[n].Accuracy);
  const f1s   = names.map(n => data[n].F1);

  destroyChart('chart-clf-acc');
  destroyChart('chart-clf-f1');

  chartInstances['chart-clf-acc'] = new Chart(
    document.getElementById('chart-clf-acc'),
    barConfig(short, accs, 'Accuracy', 'Accuracy')
  );
  chartInstances['chart-clf-f1'] = new Chart(
    document.getElementById('chart-clf-f1'),
    barConfig(short, f1s, 'F1 Score', 'F1')
  );
}

function destroyChart(id) {
  if (chartInstances[id]) { chartInstances[id].destroy(); delete chartInstances[id]; }
}

function barConfig(labels, values, label, yLabel, lower = false) {
  return {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label,
        data: values,
        backgroundColor: COLORS.map(c => c + 'CC'),
        borderColor:     COLORS,
        borderWidth: 1.5,
        borderRadius: 6,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 800, easing: 'easeOutQuart' },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(18,18,30,0.95)',
          borderColor: 'rgba(108,99,255,0.4)',
          borderWidth: 1,
          titleColor: '#F0F0FF',
          bodyColor: '#9A9ABF',
          padding: 12,
          callbacks: {
            label: ctx => ` ${ctx.parsed.y.toFixed(4)}`,
          }
        }
      },
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { font: { size: 11 } } },
        y: {
          grid: { color: 'rgba(255,255,255,0.04)' },
          ticks: { font: { size: 11 } },
          title: { display: true, text: yLabel, color: '#9A9ABF', font: { size: 11 } },
          beginAtZero: !lower,
        }
      }
    }
  };
}

/* ── Load EDA plots ───────────────────────────────────────── */
async function loadPlots() {
  try {
    const res   = await fetch(`${API}/api/plots`);
    const data  = await res.json();
    const grid  = document.getElementById('plots-grid');

    if (!data.plots || data.plots.length === 0) {
      grid.innerHTML = '<p style="color:var(--text-muted);padding:40px;text-align:center;">No plots found. Run train.py to generate charts.</p>';
      return;
    }

    grid.innerHTML = '';
    data.plots.forEach(plotPath => {
      const name = plotPath.split('/').pop().replace(/_/g,' ').replace('.png','');
      grid.innerHTML += `
        <div class="glass-card plot-card">
          <img src="${plotPath}" alt="${name}" loading="lazy" />
          <div class="plot-title">${name}</div>
        </div>`;
    });
  } catch (err) {
    document.getElementById('plots-grid').innerHTML =
      `<p style="color:var(--text-muted);padding:40px;text-align:center;">Failed to load plots: ${err.message}</p>`;
  }
}

/* ───────────────────────────────────────────────────────────
   TAB SWITCHING
   ─────────────────────────────────────────────────────────── */
function switchTab(tab) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  document.getElementById(`tab-${tab}`).classList.add('active');
  document.getElementById(`tab-content-${tab}`).classList.add('active');
}

/* ───────────────────────────────────────────────────────────
   HELPERS
   ─────────────────────────────────────────────────────────── */
function shortName(name) {
  const map = {
    'Linear Regression':  'Lin.Reg',
    'Logistic Regression':'Log.Reg',
    'SVM SVR':            'SVR',
    'SVM SVC':            'SVC',
    'Random Forest':      'R.Forest',
    'Gradient Boosting':  'Grad.Boost',
    'XGBoost':            'XGBoost',
  };
  return map[name] || name?.split(' ').map(w => w[0]).join('') || '—';
}

function cleanName(name) {
  // Remove file-path artifacts from joblib-loaded names
  return name.replace(/[_]+/g, ' ').replace(/\b\w/g, l => l.toUpperCase()).trim();
}

/* ───────────────────────────────────────────────────────────
   INIT
   ─────────────────────────────────────────────────────────── */
(async function init() {
  await checkHealth();
  await loadHomeStats();
})();
