/**
 * FLUX — Infectious Disease Intelligence Command Center
 * Client Application Logic & Chart Controllers
 */

const API_BASE = window.location.origin;

// State management
const state = {
  activeView: 'landing',
  activeRegion: 'Global',
  activeDisease: 'all',
  activeSearch: '',
  forecastHorizon: 14,
  mapInitialized: false,
  mapInstance: null,
  charts: {},
  sampleTweets: [],
  currentReportMarkdown: ''
};

// ----------------- Routing & Navigation -----------------
function navigate(viewName) {
  state.activeView = viewName;
  
  // Update nav link active states
  document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
  const activeNav = document.getElementById(`nav-${viewName}`);
  if (activeNav) activeNav.classList.add('active');

  // Update panels
  document.querySelectorAll('.view-panel').forEach(panel => panel.classList.remove('active'));
  const targetPanel = document.getElementById(`view-${viewName}`);
  if (targetPanel) targetPanel.classList.add('active');

  // Trigger view specific initialization
  if (viewName === 'dashboard') {
    loadDashboardMetrics();
    loadDailySurveillanceChart();
  } else if (viewName === 'explore') {
    loadExploreData();
  } else if (viewName === 'nlp-lab') {
    runNlpInference();
  } else if (viewName === 'trends') {
    loadForecastData(state.forecastHorizon);
  } else if (viewName === 'map') {
    setTimeout(initGeospatialMap, 150);
  } else if (viewName === 'signals') {
    loadAnomalySignals();
  } else if (viewName === 'research') {
    renderProjectList();
  }

  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ----------------- Toast Notifications -----------------
function showToast(message, type = 'info') {
  const container = document.getElementById('toastContainer');
  if (!container) return;

  const toast = document.createElement('div');
  toast.className = 'toast';
  toast.textContent = message;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(10px)';
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}

// ----------------- Global Filter Handler -----------------
function handleFilterChange() {
  const diseaseEl = document.getElementById('filterDisease');
  const regionEl = document.getElementById('filterRegion');
  
  if (diseaseEl) state.activeDisease = diseaseEl.value;
  if (regionEl) state.activeRegion = regionEl.value;

  showToast(`Applied filters: ${state.activeRegion} · ${state.activeDisease}`);
  
  // Reload current view
  if (state.activeView === 'dashboard') {
    loadDashboardMetrics();
    loadDailySurveillanceChart();
  } else if (state.activeView === 'explore') {
    loadExploreData();
  } else {
    loadLandingMetrics();
  }
}

function handleSearchInput(e) {
  state.activeSearch = e.target.value;
  if (state.activeView === 'explore') {
    loadExploreData();
  }
}

// ----------------- KPI Loaders -----------------
async function loadLandingMetrics() {
  try {
    const res = await fetch(`${API_BASE}/api/v1/overview?region=${encodeURIComponent(state.activeRegion)}`);
    if (!res.ok) return;
    const data = await res.json();

    const totalEl = document.getElementById('kpi-total');
    if (totalEl) totalEl.textContent = Number(data.total_analyzed_posts).toLocaleString();

    const anomEl = document.getElementById('kpi-anomalies');
    if (anomEl) anomEl.textContent = data.anomaly_alerts_count;

    const fcastEl = document.getElementById('kpi-forecast');
    if (fcastEl) fcastEl.textContent = Number(data.forecasted_7d_volume).toLocaleString();
  } catch (err) {
    console.error('Failed to load landing metrics:', err);
  }
}

async function loadDashboardMetrics() {
  try {
    const res = await fetch(`${API_BASE}/api/v1/overview?region=${encodeURIComponent(state.activeRegion)}`);
    if (!res.ok) return;
    const data = await res.json();

    document.getElementById('dash-tweets').textContent = Number(data.total_analyzed_posts).toLocaleString();
    document.getElementById('dash-region-label').textContent = `Region: ${state.activeRegion}`;

    const sent = data.sentiment_proportions;
    document.getElementById('dash-sent-ratio').textContent = `${sent.Positive}% / ${sent.Neutral}% / ${sent.Negative}%`;

    if (data.top_symptoms && data.top_symptoms.length > 0) {
      const topSym = data.top_symptoms[0];
      document.getElementById('dash-top-symptom').textContent = topSym.symptom.toUpperCase();
      document.getElementById('dash-symptom-rate').textContent = `${topSym.rate_pct}% mention frequency`;
    }

    renderSentimentDoughnut(data.sentiment_proportions);
    renderSymptomsBar(data.top_symptoms);
  } catch (err) {
    console.error('Failed to load dashboard metrics:', err);
  }
}

// ----------------- Chart Controllers -----------------
async function loadDailySurveillanceChart() {
  try {
    const res = await fetch(`${API_BASE}/api/v1/surveillance/daily`);
    if (!res.ok) return;
    const payload = await res.json();
    const series = payload.series;

    const dates = series.map(d => d.date);
    const tweets = series.map(d => {
      if (state.activeRegion === 'India') return d.india_tweets;
      if (state.activeRegion === 'Maharashtra') return d.maha_tweets;
      if (state.activeRegion === 'Mumbai') return d.mumbai_tweets;
      return d.tweets_total;
    });

    const cases = series.map(d => {
      if (state.activeRegion === 'India') return d.india_cases;
      if (state.activeRegion === 'Maharashtra') return d.maha_cases;
      if (state.activeRegion === 'Mumbai') return d.mumbai_cases;
      return d.world_cases_owid;
    });

    const canvas = document.getElementById('dailyTrendChart');
    if (!canvas) return;

    if (state.charts.dailyTrend) {
      state.charts.dailyTrend.destroy();
    }

    const ctx = canvas.getContext('2d');
    state.charts.dailyTrend = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: dates,
        datasets: [
          {
            label: `Daily Social Volume (${state.activeRegion})`,
            data: tweets,
            backgroundColor: 'rgba(6, 182, 212, 0.65)',
            borderColor: '#06b6d4',
            borderWidth: 1,
            yAxisID: 'y',
            order: 2
          },
          {
            label: `Confirmed New Cases (${state.activeRegion})`,
            data: cases,
            type: 'line',
            borderColor: '#f43f5e',
            backgroundColor: 'rgba(244, 63, 94, 0.1)',
            borderWidth: 2.5,
            pointRadius: 3,
            tension: 0.25,
            yAxisID: 'y1',
            order: 1
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { labels: { color: '#94a3b8', font: { family: 'Plus Jakarta Sans', size: 12 } } }
        },
        scales: {
          x: {
            ticks: { color: '#64748b', maxTicksLimit: 12 },
            grid: { color: 'rgba(255, 255, 255, 0.05)' }
          },
          y: {
            type: 'linear',
            position: 'left',
            ticks: { color: '#06b6d4' },
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            title: { display: true, text: 'Tweets / Day', color: '#06b6d4' }
          },
          y1: {
            type: 'linear',
            position: 'right',
            ticks: { color: '#f43f5e' },
            grid: { drawOnChartArea: false },
            title: { display: true, text: 'Confirmed Cases / Day', color: '#f43f5e' }
          }
        }
      }
    });
  } catch (err) {
    console.error('Failed to load daily trend chart:', err);
  }
}

function renderSentimentDoughnut(sentDist) {
  const canvas = document.getElementById('sentimentDoughnutChart');
  if (!canvas) return;

  if (state.charts.sentiment) {
    state.charts.sentiment.destroy();
  }

  const ctx = canvas.getContext('2d');
  state.charts.sentiment = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Positive', 'Neutral', 'Negative'],
      datasets: [{
        data: [sentDist.Positive, sentDist.Neutral, sentDist.Negative],
        backgroundColor: ['#10b981', '#64748b', '#f43f5e'],
        borderWidth: 0,
        hoverOffset: 6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: 'bottom', labels: { color: '#94a3b8', padding: 14 } }
      },
      cutout: '68%'
    }
  });
}

function renderSymptomsBar(symptoms) {
  const canvas = document.getElementById('symptomsChart');
  if (!canvas || !symptoms) return;

  if (state.charts.symptoms) {
    state.charts.symptoms.destroy();
  }

  const labels = symptoms.map(s => s.symptom);
  const values = symptoms.map(s => s.rate_pct);

  const ctx = canvas.getContext('2d');
  state.charts.symptoms = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: '% Frequency in Selection',
        data: values,
        backgroundColor: 'rgba(20, 184, 166, 0.7)',
        borderColor: '#14b8a6',
        borderWidth: 1,
        borderRadius: 4
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false }
      },
      scales: {
        x: {
          ticks: { color: '#64748b' },
          grid: { color: 'rgba(255, 255, 255, 0.05)' }
        },
        y: {
          ticks: { color: '#cbd5e1' },
          grid: { display: false }
        }
      }
    }
  });
}

// ----------------- Explorer Data Loader -----------------
async function loadExploreData() {
  try {
    const kwParam = state.activeSearch ? `&keyword=${encodeURIComponent(state.activeSearch)}` : '';
    const res = await fetch(`${API_BASE}/api/v1/tweets/sample?limit=50&region=${encodeURIComponent(state.activeRegion)}${kwParam}`);
    if (!res.ok) return;
    const payload = await res.json();
    const rows = payload.sample;
    state.sampleTweets = rows;

    const tbody = document.getElementById('exploreTableBody');
    if (!tbody) return;

    if (!rows || rows.length === 0) {
      tbody.innerHTML = `<tr><td colspan="7" style="text-align:center; padding:2rem; color:var(--text-dim);">No tweets matched your search filter.</td></tr>`;
      return;
    }

    tbody.innerHTML = rows.map(r => {
      const sentBadge = r.sentiment === 'Positive' ? 'badge-success' : (r.sentiment === 'Negative' ? 'badge-danger' : 'badge-neutral');
      return `
        <tr>
          <td style="white-space:nowrap; font-family:var(--font-mono); font-size:0.75rem;">${r.date.split('T')[0]}</td>
          <td><span class="badge-neutral">${r.region}</span></td>
          <td style="max-width:140px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; color:var(--text-dim);">${r.user_location || '—'}</td>
          <td><span class="${sentBadge}">${r.sentiment}</span></td>
          <td style="font-family:var(--font-mono); font-size:0.78rem;">${r.compound > 0 ? '+' : ''}${r.compound.toFixed(3)}</td>
          <td style="font-size:0.78rem; color:var(--cyan-bright);">${r.topic_label ? r.topic_label.split(':')[0] : 'T0'}</td>
          <td style="max-width:400px; font-size:0.82rem;">${escapeHtml(r.text)}</td>
        </tr>
      `;
    }).join('');
  } catch (err) {
    console.error('Failed to load explore data:', err);
  }
}

function escapeHtml(str) {
  if (!str) return '';
  return str.replace(/[&<>"']/g, m => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' })[m]);
}

function downloadSampleCSV() {
  if (!state.sampleTweets || state.sampleTweets.length === 0) {
    showToast('No filtered tweets to download');
    return;
  }
  const headers = ['date', 'region', 'user_location', 'sentiment', 'compound', 'topic_label', 'text'];
  const csvContent = 'data:text/csv;charset=utf-8,' + [
    headers.join(','),
    ...state.sampleTweets.map(row => headers.map(field => `"${String(row[field] || '').replace(/"/g, '""')}"`).join(','))
  ].join('\n');

  const link = document.createElement('a');
  link.setAttribute('href', encodeURI(csvContent));
  link.setAttribute('download', `flux_tweets_sample_${state.activeRegion.toLowerCase()}.csv`);
  document.body.appendChild(link);
  link.click();
  link.remove();
  showToast('Downloaded filtered CSV extract');
}

// ----------------- NLP Lab & Explainable AI (XAI) -----------------
function setNlpSample(text) {
  const el = document.getElementById('nlpInputText');
  if (el) el.value = text;
  runNlpInference();
}

async function runNlpInference() {
  const inputEl = document.getElementById('nlpInputText');
  if (!inputEl) return;
  const text = inputEl.value.trim();
  if (!text) return;

  try {
    const res = await fetch(`${API_BASE}/api/v1/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: text, explain: true })
    });
    if (!res.ok) return;
    const data = await res.json();

    const predLabel = document.getElementById('nlpPredLabel');
    predLabel.textContent = data.sentiment;
    predLabel.style.color = data.sentiment === 'Positive' ? 'var(--emerald-success)' : (data.sentiment === 'Negative' ? 'var(--rose-alert)' : '#fff');

    document.getElementById('nlpPredConf').textContent = `${(data.confidence * 100).toFixed(1)}%`;
    document.getElementById('nlpVaderVerdict').textContent = 
      `VADER Compound: ${data.vader_compound > 0 ? '+' : ''}${data.vader_compound} (${data.vader_sentiment}) · Consensus: ${data.consensus ? 'Agreed' : 'Divergent'}`;

    renderXaiBars(data.top_explanations);
  } catch (err) {
    console.error('NLP Inference failed:', err);
  }
}

function renderXaiBars(explanations) {
  const container = document.getElementById('xaiBarsContainer');
  if (!container) return;

  if (!explanations || explanations.length === 0) {
    container.innerHTML = `<p style="font-size:0.8rem; color:var(--text-dim);">No specific token weights above threshold.</p>`;
    return;
  }

  const maxAbsWeight = Math.max(...explanations.map(x => Math.abs(x.weight)), 0.01);

  container.innerHTML = explanations.map(exp => {
    const pct = Math.min(100, Math.round((Math.abs(exp.weight) / maxAbsWeight) * 100));
    const fillClass = exp.weight >= 0 ? 'xai-fill-pos' : 'xai-fill-neg';
    return `
      <div class="xai-bar-item">
        <span class="xai-token" title="${exp.token}">${exp.token}</span>
        <div class="xai-track">
          <div class="${fillClass}" style="width: ${pct}%;"></div>
        </div>
        <span class="xai-weight">${exp.weight > 0 ? '+' : ''}${exp.weight.toFixed(3)}</span>
      </div>
    `;
  }).join('');
}

// ----------------- Forecasting Chart -----------------
function setForecastHorizon(days) {
  state.forecastHorizon = days;
  loadForecastData(days);
}

async function loadForecastData(horizonDays) {
  try {
    const res = await fetch(`${API_BASE}/api/v1/surveillance/forecast?horizon_days=${horizonDays}&target=india`);
    if (!res.ok) return;
    const data = await res.json();

    document.getElementById('forecastMetricsBadge').textContent = 
      `MAE: ${data.evaluation_metrics.mae} · RMSE: ${data.evaluation_metrics.rmse} · Residual σ: ${data.evaluation_metrics.residual_std}`;

    const hist = data.historical;
    const fcast = data.forecast;

    const allLabels = [...hist.map(h => h.date), ...fcast.map(f => f.date)];
    const histData = [...hist.map(h => h.value), ...fcast.map(() => null)];
    const fcastMean = [...hist.map((h, i) => i === hist.length - 1 ? h.value : null), ...fcast.map(f => f.forecast_mean)];
    const fcastUpper = [...hist.map((h, i) => i === hist.length - 1 ? h.value : null), ...fcast.map(f => f.upper_95)];
    const fcastLower = [...hist.map((h, i) => i === hist.length - 1 ? h.value : null), ...fcast.map(f => f.lower_95)];

    const canvas = document.getElementById('forecastChart');
    if (!canvas) return;

    if (state.charts.forecast) {
      state.charts.forecast.destroy();
    }

    const ctx = canvas.getContext('2d');
    state.charts.forecast = new Chart(ctx, {
      type: 'line',
      data: {
        labels: allLabels,
        datasets: [
          {
            label: 'Observed Historical Volume',
            data: histData,
            borderColor: '#06b6d4',
            backgroundColor: 'rgba(6, 182, 212, 0.1)',
            borderWidth: 2.5,
            pointRadius: 3
          },
          {
            label: `Forecast Projected Mean (${horizonDays}D)`,
            data: fcastMean,
            borderColor: '#f59e0b',
            borderWidth: 2.5,
            borderDash: [5, 5],
            pointRadius: 3
          },
          {
            label: 'Upper 95% Confidence Band',
            data: fcastUpper,
            borderColor: 'rgba(245, 158, 11, 0.4)',
            borderWidth: 1,
            pointRadius: 0,
            fill: '+1',
            backgroundColor: 'rgba(245, 158, 11, 0.1)'
          },
          {
            label: 'Lower 95% Confidence Band',
            data: fcastLower,
            borderColor: 'rgba(245, 158, 11, 0.4)',
            borderWidth: 1,
            pointRadius: 0,
            fill: false
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { labels: { color: '#94a3b8' } }
        },
        scales: {
          x: { ticks: { color: '#64748b' }, grid: { color: 'rgba(255, 255, 255, 0.05)' } },
          y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255, 255, 255, 0.05)' } }
        }
      }
    });
  } catch (err) {
    console.error('Failed to load forecast data:', err);
  }
}

// ----------------- Geospatial Map (Leaflet) -----------------
async function initGeospatialMap() {
  const container = document.getElementById('mapContainer');
  if (!container) return;

  if (state.mapInstance) {
    state.mapInstance.invalidateSize();
    return;
  }

  // Centered over India / Western Eurasia
  state.mapInstance = L.map('mapContainer').setView([20.5937, 78.9629], 4);

  // Dark Tiles from CartoDB
  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
    subdomains: 'abcd',
    maxZoom: 19
  }).addTo(state.mapInstance);

  try {
    const res = await fetch(`${API_BASE}/api/v1/surveillance/geo`);
    if (!res.ok) return;
    const data = await res.json();

    const hubsGrid = document.getElementById('geoHubsGrid');
    if (hubsGrid) {
      hubsGrid.innerHTML = data.hubs.map(h => `
        <div class="card" style="padding:1rem;">
          <div style="font-weight:700; color:#fff; font-size:0.95rem;">${h.name}</div>
          <div style="font-size:0.75rem; color:var(--text-dim); margin-bottom:0.5rem;">${h.state_country}</div>
          <div style="display:flex; justify-content:space-between; font-size:0.8rem; margin-bottom:0.3rem;">
            <span style="color:var(--text-dim);">Discussion Volume:</span>
            <span style="font-weight:700; color:var(--cyan-bright);">${h.discussion_volume.toLocaleString()}</span>
          </div>
          <div style="display:flex; justify-content:space-between; font-size:0.8rem;">
            <span style="color:var(--text-dim);">Top Clinical Signal:</span>
            <span style="color:var(--amber-warning); font-weight:600;">${h.top_symptom} (${h.symptom_rate_pct}%)</span>
          </div>
        </div>
      `).join('');
    }

    data.hubs.forEach(hub => {
      const circleColor = hub.intensity_level === 'Severe' ? '#f43f5e' : (hub.intensity_level === 'High' ? '#f59e0b' : '#06b6d4');
      const circle = L.circleMarker([hub.lat, hub.lng], {
        radius: hub.radius / 2.5,
        fillColor: circleColor,
        color: '#fff',
        weight: 1.5,
        opacity: 0.9,
        fillOpacity: 0.6
      }).addTo(state.mapInstance);

      circle.bindPopup(`
        <div style="font-family:'Plus Jakarta Sans',sans-serif; color:#0f172a; min-width:180px;">
          <h4 style="margin:0 0 4px 0; font-size:14px;">${hub.name}</h4>
          <p style="margin:0 0 6px 0; font-size:11px; color:#64748b;">${hub.state_country}</p>
          <div style="font-size:12px; margin-bottom:3px;"><strong>Volume:</strong> ${hub.discussion_volume.toLocaleString()} posts</div>
          <div style="font-size:12px; margin-bottom:3px;"><strong>Top Symptom:</strong> ${hub.top_symptom}</div>
          <div style="font-size:12px;"><strong>Mean Compound:</strong> ${hub.mean_compound > 0 ? '+' : ''}${hub.mean_compound}</div>
        </div>
      `);
    });
  } catch (err) {
    console.error('Failed to load geospatial data:', err);
  }
}

// ----------------- Anomaly Signals Monitor -----------------
async function loadAnomalySignals() {
  const threshEl = document.getElementById('signalThresholdSelect');
  const thresh = threshEl ? threshEl.value : '2.0';

  try {
    const res = await fetch(`${API_BASE}/api/v1/surveillance/anomalies?threshold=${thresh}&metric=india`);
    if (!res.ok) return;
    const data = await res.json();

    const tbody = document.getElementById('signalsTableBody');
    if (!tbody) return;

    if (!data.signals || data.signals.length === 0) {
      tbody.innerHTML = `<tr><td colspan="9" style="text-align:center; padding:2rem; color:var(--emerald-success);">✅ No abnormal signals exceeding ${thresh}σ threshold. Baseline stable.</td></tr>`;
      return;
    }

    tbody.innerHTML = data.signals.map(s => {
      const sevBadge = s.severity.includes('High') ? 'badge-danger' : (s.severity.includes('Elevated') ? 'badge-warning' : 'badge-neutral');
      return `
        <tr>
          <td style="font-family:var(--font-mono); font-size:0.78rem;">${s.detection_id}</td>
          <td style="white-space:nowrap; font-family:var(--font-mono);">${s.timestamp}</td>
          <td>${s.geographic_level}</td>
          <td style="font-weight:700;">${s.observed_count}</td>
          <td style="color:var(--text-dim);">${s.baseline_mean}</td>
          <td style="color:var(--rose-alert); font-weight:700;">+${s.magnitude_deviation_pct}%</td>
          <td><span class="${sevBadge}">${s.severity}</span></td>
          <td style="font-size:0.8rem; color:var(--text-muted);">${s.recommended_action}</td>
          <td><span class="badge-neutral">${s.analyst_status}</span></td>
        </tr>
      `;
    }).join('');
  } catch (err) {
    console.error('Failed to load anomaly signals:', err);
  }
}

// ----------------- Research Workspace -----------------
function renderProjectList() {
  const container = document.getElementById('projectListContainer');
  if (!container) return;

  const defaultProjects = [
    { id: 'PROJ-1', title: 'Mumbai Hospital Bed Strain & Oxygen Signals', date: '2026-10-27', notes: 'Evaluating social signals as leading indicators during early containment' },
    { id: 'PROJ-2', title: 'Vaccine Hesitancy & Efficacy Sentiment Evolution', date: '2026-10-26', notes: 'Cross-comparison of Phase 3 trial announcement reactions across India' }
  ];

  const saved = JSON.parse(localStorage.getItem('flux_projects') || JSON.stringify(defaultProjects));

  container.innerHTML = saved.map(p => `
    <div style="padding:1rem; background:rgba(255,255,255,0.03); border:1px solid var(--border-subtle); border-radius:8px;">
      <div style="font-weight:700; color:#fff; font-size:0.92rem; margin-bottom:0.25rem;">${p.title}</div>
      <div style="font-size:0.75rem; color:var(--text-dim); margin-bottom:0.4rem;">Project ID: ${p.id} · Created: ${p.date}</div>
      <div style="font-size:0.82rem; color:var(--text-muted);">${p.notes}</div>
    </div>
  `).join('');

  const notesEl = document.getElementById('analystNotesText');
  if (notesEl) {
    notesEl.value = localStorage.getItem('flux_analyst_notes') || '';
  }
}

function createNewProject() {
  const title = prompt('Enter New Research Project Title:', 'COVID-19 Urban Surveillance Analysis');
  if (!title) return;

  const defaultProjects = [
    { id: 'PROJ-1', title: 'Mumbai Hospital Bed Strain & Oxygen Signals', date: '2026-10-27', notes: 'Evaluating social signals as leading indicators during early containment' }
  ];
  const saved = JSON.parse(localStorage.getItem('flux_projects') || JSON.stringify(defaultProjects));
  saved.unshift({
    id: `PROJ-${saved.length + 1}`,
    title: title,
    date: new Date().toISOString().split('T')[0],
    notes: 'Initialized research project workspace.'
  });
  localStorage.setItem('flux_projects', JSON.stringify(saved));
  renderProjectList();
  showToast(`Created research project: ${title}`);
}

function saveAnalystNotes() {
  const text = document.getElementById('analystNotesText').value;
  localStorage.setItem('flux_analyst_notes', text);
  showToast('Analyst notes saved to workspace.');
}

// ----------------- Executive Report Generator -----------------
async function compileExecutiveReport() {
  const notes = localStorage.getItem('flux_analyst_notes') || '';
  try {
    const res = await fetch(`${API_BASE}/api/v1/reports/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        title: 'FLUX Epidemiological Intelligence Briefing',
        disease: state.activeDisease === 'all' ? 'COVID-19 & Influenza' : state.activeDisease.toUpperCase(),
        region: state.activeRegion,
        analyst_notes: notes
      })
    });
    if (!res.ok) return;
    const data = await res.json();
    state.currentReportMarkdown = data.markdown;

    const preview = document.getElementById('reportPreviewContainer');
    preview.innerHTML = `<pre style="white-space:pre-wrap; font-family:var(--font-mono); font-size:0.85rem; color:#cbd5e1; background:rgba(0,0,0,0.3); padding:1.25rem; border-radius:8px;">${escapeHtml(data.markdown)}</pre>`;

    document.getElementById('btnDownloadReport').style.display = 'inline-flex';
    showToast('Compiled fresh epidemiological briefing');
  } catch (err) {
    console.error('Report compilation failed:', err);
  }
}

function downloadCompiledReport() {
  if (!state.currentReportMarkdown) return;
  const blob = new Blob([state.currentReportMarkdown], { type: 'text/markdown' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `FLUX_Intelligence_Briefing_${new Date().toISOString().split('T')[0]}.md`;
  a.click();
  URL.revokeObjectURL(url);
  showToast('Downloaded Markdown briefing report');
}

// ----------------- Initialization -----------------
window.addEventListener('DOMContentLoaded', () => {
  loadLandingMetrics();
  
  // Hash routing
  const hash = window.location.hash.replace('#', '');
  if (hash) {
    navigate(hash);
  }
});
