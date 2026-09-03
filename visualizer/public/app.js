/**
 * app.js
 * AI Supervision for Traversability Segmentation in Unstructured Environments
 * Comprehensive interactive logic for benchmarks, SVG charts, mask splitter,
 * multi-seed drilldown, and qualitative failure analysis.
 */

(function () {
  'use strict';

  // Global Application State
  const state = {
    data: null,
    currentView: 'view-overview',
    currentPolicy: 'convergence', // 'convergence' | 'blue_green' | 'blue_only'
    currentSort: { key: 'mIoU', dir: 'desc' },
    familyFilter: 'all',
    searchQuery: '',
    paretoHardware: 'h100', // 'h100' | 'rtx5060' | 'ryzen'
    curveMetric: 'val_mIoU', // 'val_mIoU' | 'val_loss' | 'val_fsr'
    activeCurveModels: new Set([
      'FPN/EfficientNet-B0',
      'PIDNet-S',
      'ROD ViT-S',
      'SegFormer-B0',
      'DDRNet-23-Slim'
    ]),
    galleryCategory: 'all',
    splitMode: 'rgb_pred', // 'rgb_pred' | 'gt_pred' | 'triptych'
    splitPercent: 50,
    currentSampleId: 'mixed_pln_1278',
    browserPage: 1,
    browserPerPage: 24,
    browserQuery: ''
  };

  let lastFocusedElement = null;

  // Color Palette Constants for Charts & Models
  const MODEL_COLORS = {
    'FPN/EfficientNet-B0': '#146b55',
    'U-Net/EfficientNet-B0': '#347b75',
    'FPN/MobileNetV2': '#486b82',
    'SegFormer-B0': '#59617a',
    'U-Net/MobileNetV2': '#76637a',
    'ROD ViT-S': '#a8493f',
    'PIDNet-S': '#55733e',
    'BiSeNetV2': '#9b681b',
    'DDRNet-23-Slim': '#8b5940'
  };

  // =========================================================================
  // Initialization
  // =========================================================================

  document.addEventListener('DOMContentLoaded', initApp);

  async function initApp() {
    setupNavigation();
    setupSplitSlider();
    setupEventListeners();

    try {
      const res = await fetch('./data/benchmark_data.json');
      if (!res.ok) throw new Error(`HTTP error ${res.status}`);
      state.data = await res.json();
      renderOverview();
      renderLeaderboard();
      renderParetoChart();
      renderSafetyChart();
      renderGainChart();
      renderConvergenceCurves();
      renderGallery();
      renderTheoryGallery();
      updateSplitInspectorSample(state.currentSampleId);
      updatePackagedSampleCount();
    } catch (err) {
      console.error('Failed to load benchmark data:', err);
      const main = document.getElementById('app-main');
      if (main) {
        main.innerHTML = `
          <div style="padding: 3rem; text-align: center;">
            <h2 style="color: var(--rose-400); margin-bottom: 1rem;">Failed to load benchmark dataset</h2>
            <p style="color: var(--text-muted);">${err.message}</p>
          </div>
        `;
      }
    }
  }

  // =========================================================================
  // Navigation
  // =========================================================================

  function setupNavigation() {
    const tabs = document.querySelectorAll('.nav-tab');
    tabs.forEach((tab, index) => {
      tab.addEventListener('click', () => {
        const targetId = tab.getAttribute('data-target');
        switchView(targetId);
      });

      tab.addEventListener('keydown', event => {
        let nextIndex = null;
        if (event.key === 'ArrowRight') nextIndex = (index + 1) % tabs.length;
        if (event.key === 'ArrowLeft') nextIndex = (index - 1 + tabs.length) % tabs.length;
        if (event.key === 'Home') nextIndex = 0;
        if (event.key === 'End') nextIndex = tabs.length - 1;
        if (nextIndex !== null) {
          event.preventDefault();
          tabs[nextIndex].focus();
          switchView(tabs[nextIndex].getAttribute('data-target'));
        }
      });
    });

    const initialView = window.location.hash.replace('#', '');
    const validInitialView = document.getElementById(initialView)?.classList.contains('view-panel');
    switchView(validInitialView ? initialView : 'view-overview', false);

    window.addEventListener('hashchange', () => {
      const viewId = window.location.hash.replace('#', '');
      if (document.getElementById(viewId)?.classList.contains('view-panel')) {
        switchView(viewId, false);
      }
    });

    // Hero buttons
    const btnExplore = document.getElementById('btn-hero-explore-leaderboard');
    if (btnExplore) btnExplore.addEventListener('click', () => switchView('view-leaderboard'));

    const btnInspect = document.getElementById('btn-hero-inspect-masks');
    if (btnInspect) btnInspect.addEventListener('click', () => switchView('view-inspector'));

    const btnPareto = document.getElementById('btn-hero-pareto');
    if (btnPareto) btnPareto.addEventListener('click', () => switchView('view-analytics'));
  }

  function switchView(viewId, updateHash = true) {
    const nextPanel = document.getElementById(viewId);
    if (!nextPanel || !nextPanel.classList.contains('view-panel')) return;
    state.currentView = viewId;

    // Update Nav Tab UI
    document.querySelectorAll('.nav-tab').forEach(tab => {
      const isActive = tab.getAttribute('data-target') === viewId;
      tab.classList.toggle('active', isActive);
      tab.setAttribute('aria-selected', String(isActive));
      tab.tabIndex = isActive ? 0 : -1;
    });

    // Update View Panels
    document.querySelectorAll('.view-panel').forEach(panel => {
      const isActive = panel.id === viewId;
      panel.classList.toggle('active', isActive);
      panel.setAttribute('aria-hidden', String(!isActive));
    });

    if (updateHash && window.location.hash !== `#${viewId}`) {
      history.pushState(null, '', `#${viewId}`);
    }

    // Re-render charts or resize responsive elements
    if (viewId === 'view-analytics') {
      setTimeout(() => {
        renderParetoChart();
        renderSafetyChart();
        renderGainChart();
        renderConvergenceCurves();
      }, 50);
    }

    const behavior = window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth';
    window.scrollTo({ top: 0, behavior });
  }

  // =========================================================================
  // 1. Executive Overview Rendering
  // =========================================================================

  function renderOverview() {
    if (!state.data) return;

    // Render KPI Cards
    const kpiContainer = document.getElementById('hero-kpis-container');
    if (kpiContainer && state.data.kpis) {
      kpiContainer.innerHTML = state.data.kpis.map(kpi => `
        <div class="kpi-card ${kpi.color}">
          <div class="kpi-label">${escapeHtml(kpi.label)}</div>
          <div class="kpi-value text-${kpi.color}">${escapeHtml(kpi.value)}</div>
          <div class="kpi-sub">${escapeHtml(kpi.sub)}</div>
        </div>
      `).join('');
    }

    // Render 9 Architecture Cards
    const archContainer = document.getElementById('arch-card-grid');
    if (archContainer && state.data.suites && state.data.suites.convergence) {
      const convModels = state.data.suites.convergence;
      const bgModels = state.data.suites.blue_green || [];

      archContainer.innerHTML = convModels.map(m => {
        const bgMatch = bgModels.find(b => b.model === m.model) || {};
        const paramsStr = bgMatch.parameters ? (bgMatch.parameters / 1e6).toFixed(2) + 'M' : 'N/A';
        const h100Fps = bgMatch.h100_fps ? bgMatch.h100_fps.toFixed(1) + ' FPS' : '--';

        return `
          <div class="arch-card">
            <div class="arch-card-top">
              <div class="arch-header-row">
                <h4 class="arch-name" style="color: ${MODEL_COLORS[m.model] || 'var(--text-bright)'};">
                  ${escapeHtml(m.model)}
                </h4>
                <span class="arch-family-badge">${escapeHtml(m.family)}</span>
              </div>
              <p class="arch-desc">${escapeHtml(m.description)}</p>
            </div>
            
            <div class="arch-specs-list">
              <div class="arch-spec-row">
                <span>Backbone:</span>
                <strong>${escapeHtml(m.backbone)}</strong>
              </div>
              <div class="arch-spec-row">
                <span>Decoder:</span>
                <strong>${escapeHtml(m.decoder)}</strong>
              </div>
              <div class="arch-spec-row">
                <span>Convergence mIoU:</span>
                <strong class="text-cyan">${m.mIoU_mean.toFixed(4)} ± ${m.mIoU_std.toFixed(4)}</strong>
              </div>
              <div class="arch-spec-row">
                <span>Parameters / H100 Speed:</span>
                <strong>${paramsStr} • ${h100Fps}</strong>
              </div>
            </div>
          </div>
        `;
      }).join('');
    }
  }

  // =========================================================================
  // 2. Leaderboard Table
  // =========================================================================

  function renderLeaderboard() {
    if (!state.data || !state.data.suites) return;

    const tbody = document.getElementById('benchmark-table-body');
    if (!tbody) return;

    let models = [];
    const policy = state.currentPolicy;

    if (policy === 'convergence') {
      models = [...(state.data.suites.convergence || [])];
    } else if (policy === 'blue_green') {
      models = [...(state.data.suites.blue_green || [])];
    } else if (policy === 'blue_only') {
      models = [...(state.data.suites.blue_only || [])];
    }

    // Convergence records intentionally contain training metrics only. Join
    // deployment metadata by model name instead of showing one hard-coded
    // model's parameter count and speed for every missing value.
    const deploymentRecords = state.data.suites.blue_green || [];
    models = models.map(model => {
      const deployment = deploymentRecords.find(item => item.model === model.model) || {};
      return {
        ...model,
        parameters: model.parameters ?? deployment.parameters ?? null,
        h100_fps: model.h100_fps ?? deployment.h100_fps ?? null
      };
    });

    // Filter by Family
    if (state.familyFilter !== 'all') {
      models = models.filter(m => {
        if (state.familyFilter === 'Foundation') return m.family.includes('Foundation') || m.family.includes('Transformer');
        return m.family.toLowerCase().includes(state.familyFilter.toLowerCase());
      });
    }

    // Filter by Search Query
    if (state.searchQuery.trim()) {
      const q = state.searchQuery.toLowerCase();
      models = models.filter(m =>
        m.model.toLowerCase().includes(q) ||
        (m.family && m.family.toLowerCase().includes(q)) ||
        (m.backbone && m.backbone.toLowerCase().includes(q))
      );
    }

    // Dynamic Sorting
    models.sort((a, b) => {
      let valA, valB;
      const key = state.currentSort.key;

      if (key === 'mIoU') {
        valA = a.mIoU_mean;
        valB = b.mIoU_mean;
      } else if (key === 'f1') {
        valA = a.f1_mean;
        valB = b.f1_mean;
      } else if (key === 'fsr') {
        valA = a.fsr_mean;
        valB = b.fsr_mean;
      } else if (key === 'fbr') {
        valA = a.fbr_mean;
        valB = b.fbr_mean;
      } else if (key === 'fps') {
        valA = a.h100_fps || 0;
        valB = b.h100_fps || 0;
      } else if (key === 'gain') {
        valA = a.gain_pp || 0;
        valB = b.gain_pp || 0;
      } else if (key === 'params') {
        valA = a.parameters || 0;
        valB = b.parameters || 0;
      } else {
        valA = a.mIoU_mean;
        valB = b.mIoU_mean;
      }

      if (state.currentSort.dir === 'asc') {
        return valA > valB ? 1 : -1;
      } else {
        return valA < valB ? 1 : -1;
      }
    });

    // Update Header Text based on Policy
    const thEpoch = document.getElementById('th-epoch-header');
    const thGain = document.getElementById('th-gain-header');
    if (thEpoch) thEpoch.textContent = policy === 'convergence' ? 'Selected Epochs' : 'Epoch Budget';
    if (thGain) thGain.textContent = policy === 'convergence' ? 'Gain (15e Δ)' : 'Policy Context';

    // Render Table Rows
    tbody.innerHTML = models.map((m, idx) => {
      const rank = idx + 1;
      const rankClass = rank === 1 ? 'gold' : (rank === 2 ? 'silver' : (rank === 3 ? 'bronze' : ''));

      // Mini bar width for mIoU (baseline scaled from 0.85 to 0.96)
      const barPct = Math.max(5, Math.min(100, ((m.mIoU_mean - 0.85) / 0.11) * 100));

      // FSR safety styling
      const fsrPct = (m.fsr_mean * 100).toFixed(2);
      const fsrClass = m.fsr_mean < 0.03 ? 'safe' : (m.fsr_mean < 0.045 ? 'caution' : 'hazard');

      // FBR rate
      const fbrPct = (m.fbr_mean * 100).toFixed(2);

      // Epoch display
      let epochStr = '15 Epochs';
      if (policy === 'convergence' && m.selected_epochs) {
        epochStr = m.selected_epochs.join(', ');
      }

      // Gain display
      let gainCell = '<span class="text-dim">Baseline</span>';
      if (policy === 'convergence' && m.gain_pp) {
        gainCell = `<span class="gain-badge">+${m.gain_pp.toFixed(2)} pp</span>`;
      }

      // Speed
      const fpsStr = Number.isFinite(m.h100_fps) ? m.h100_fps.toFixed(1) : '—';
      const paramsStr = Number.isFinite(m.parameters) ? (m.parameters / 1e6).toFixed(2) + 'M' : '—';

      return `
        <tr data-model="${escapeHtml(m.model)}">
          <td class="th-rank">
            <span class="rank-badge ${rankClass}">${rank}</span>
          </td>
          <td class="table-model-cell">
            <span style="color: ${MODEL_COLORS[m.model] || 'inherit'}; font-weight: 800;">${escapeHtml(m.model)}</span>
            <span class="model-sub">${escapeHtml(m.backbone)}</span>
          </td>
          <td><span class="arch-family-badge">${escapeHtml(m.family)}</span></td>
          <td class="font-mono">${paramsStr}</td>
          <td class="font-mono text-muted" style="font-size: 0.82rem;">${escapeHtml(epochStr)}</td>
          <td>
            <div class="metric-bar-cell">
              <span>${m.mIoU_mean.toFixed(4)}</span>
              <div class="mini-bar-track">
                <div class="mini-bar-fill cyan" style="width: ${barPct}%;"></div>
              </div>
            </div>
            <span class="text-dim" style="font-size: 0.72rem;">± ${m.mIoU_std.toFixed(4)}</span>
          </td>
          <td class="font-mono">${m.f1_mean.toFixed(4)}</td>
          <td>
            <span class="fsr-tag ${fsrClass}">${fsrPct}%</span>
          </td>
          <td class="font-mono text-muted">${fbrPct}%</td>
          <td>${gainCell}</td>
          <td class="font-mono font-bold text-cyan">${fpsStr}</td>
          <td>
            <button class="btn btn-sm btn-glass btn-seed-drilldown" data-model="${escapeHtml(m.model)}">
              Seeds ▾
            </button>
          </td>
        </tr>
      `;
    }).join('');

    // Attach Seed Drilldown Listeners
    tbody.querySelectorAll('.btn-seed-drilldown').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const modelName = btn.getAttribute('data-model');
        openSeedModal(modelName);
      });
    });

    document.querySelectorAll('.benchmark-table th.sortable').forEach(th => {
      const isCurrent = th.getAttribute('data-sort') === state.currentSort.key;
      th.setAttribute('aria-sort', isCurrent ? (state.currentSort.dir === 'asc' ? 'ascending' : 'descending') : 'none');
    });
  }

  // Seed Drilldown Modal
  function openSeedModal(modelName) {
    if (!state.data) return;

    let modelData = null;
    const policy = state.currentPolicy;
    const suiteList = state.data.suites[policy] || [];
    modelData = suiteList.find(m => m.model === modelName);

    if (!modelData || !modelData.runs) {
      alert(`No multi-seed records available for ${modelName}`);
      return;
    }

    const modalTitle = document.getElementById('seed-modal-title');
    const modalSubtitle = document.getElementById('seed-modal-subtitle');
    const modalBody = document.getElementById('seed-modal-body');
    const modalBackdrop = document.getElementById('seed-modal-backdrop');

    if (modalTitle) modalTitle.textContent = `${modelName} — Multi-Seed Ledger`;
    if (modalSubtitle) modalSubtitle.textContent = `Policy: ${policy.toUpperCase()} • Seeds 1337, 2027, 4242`;

    if (modalBody) {
      modalBody.innerHTML = `
        <div style="margin-bottom: 1.5rem;">
          <div style="display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem;">
            <div class="kpi-card cyan" style="flex: 1; padding: 1rem;">
              <div class="kpi-label">Population Mean mIoU</div>
              <div class="kpi-value text-cyan" style="font-size: 1.5rem;">${modelData.mIoU_mean.toFixed(4)}</div>
              <div class="kpi-sub">Std: ± ${modelData.mIoU_std.toFixed(4)}</div>
            </div>
            <div class="kpi-card emerald" style="flex: 1; padding: 1rem;">
              <div class="kpi-label">Mean FSR (Safety)</div>
              <div class="kpi-value text-emerald" style="font-size: 1.5rem;">${(modelData.fsr_mean * 100).toFixed(2)}%</div>
              <div class="kpi-sub">Std: ± ${(modelData.fsr_std * 100).toFixed(2)}%</div>
            </div>
            <div class="kpi-card amber" style="flex: 1; padding: 1rem;">
              <div class="kpi-label">Mean FBR (Availability)</div>
              <div class="kpi-value text-amber" style="font-size: 1.5rem;">${(modelData.fbr_mean * 100).toFixed(2)}%</div>
              <div class="kpi-sub">Std: ± ${(modelData.fbr_std * 100).toFixed(2)}%</div>
            </div>
          </div>

          <h4 style="font-size: 1.1rem; margin-bottom: 0.75rem;">Seed Run Disaggregation</h4>
          <div class="table-responsive-wrapper">
            <table class="benchmark-table">
              <thead>
                <tr>
                  <th>Seed</th>
                  <th>Checkpoint Epoch</th>
                  <th>Test mIoU</th>
                  <th>F1 Score</th>
                  <th>False Safe (FSR)</th>
                  <th>False Block (FBR)</th>
                  <th>Precision</th>
                  <th>Recall</th>
                </tr>
              </thead>
              <tbody>
                ${modelData.runs.map(r => `
                  <tr>
                    <td class="font-mono font-bold text-cyan">Seed ${r.seed}</td>
                    <td class="font-mono text-muted">${r.best_epoch || r.epochs_completed || '15'}</td>
                    <td class="font-mono font-bold">${r.mIoU ? r.mIoU.toFixed(4) : '--'}</td>
                    <td class="font-mono">${r.f1 ? r.f1.toFixed(4) : '--'}</td>
                    <td class="font-mono text-rose">${r.fsr ? (r.fsr * 100).toFixed(2) + '%' : '--'}</td>
                    <td class="font-mono text-amber">${r.fbr ? (r.fbr * 100).toFixed(2) + '%' : '--'}</td>
                    <td class="font-mono">${r.precision ? (r.precision * 100).toFixed(2) + '%' : '--'}</td>
                    <td class="font-mono">${r.recall ? (r.recall * 100).toFixed(2) + '%' : '--'}</td>
                  </tr>
                `).join('')}
              </tbody>
            </table>
          </div>
        </div>
      `;
    }

    openModal(modalBackdrop, document.getElementById('seed-modal-dialog'));
  }

  // =========================================================================
  // 3. SVG Charts & Visual Analytics
  // =========================================================================

  // --- Chart 1: Pareto Frontier (Speed vs Accuracy) ---
  function renderParetoChart() {
    const svg = document.getElementById('pareto-svg');
    const tooltip = document.getElementById('pareto-tooltip');
    if (!svg || !state.data) return;

    const comparison = state.data.hardware?.model_comparison_table || [];
    const rtxDetails = Object.values(state.data.hardware?.rtx5060_details || {});
    const hwMode = state.paretoHardware; // 'h100' | 'rtx5060' | 'ryzen'
    const desc = document.getElementById('pareto-chart-desc');
    const note = document.getElementById('pareto-chart-note');

    // Build each hardware view only from measurements made on that device.
    // Accuracy follows the checkpoint evaluated by the corresponding hardware
    // benchmark; no cross-device speed estimates or mixed runtimes are used.
    let points = comparison.map(row => ({
      name: row.model,
      fps: row.h100_fps,
      latencyMs: Number.isFinite(row.h100_fps) ? 1000 / row.h100_fps : null,
      mIoU: row.mIoU_bg,
      params: row.params,
      fsr: row.fsr_bg,
      color: MODEL_COLORS[row.model] || '#566278'
    })).filter(point => Number.isFinite(point.fps) && Number.isFinite(point.mIoU));

    if (hwMode === 'rtx5060') {
      points = comparison.map(row => {
        const detail = rtxDetails.find(item => item.model === row.model);
        return {
          name: row.model,
          fps: row.rtx5060_pytorch_fps,
          latencyMs: detail?.pytorch_forward_ms ?? (Number.isFinite(row.rtx5060_pytorch_fps) ? 1000 / row.rtx5060_pytorch_fps : null),
          mIoU: detail?.test_miou_fp32 ?? null,
          params: row.params,
          fsr: null,
          color: MODEL_COLORS[row.model] || '#566278'
        };
      }).filter(point => Number.isFinite(point.fps) && Number.isFinite(point.mIoU));
      if (desc) desc.textContent = 'Only directly measured PyTorch eager results are shown. TensorRT is excluded to keep the runtime comparison consistent.';
      if (note) note.textContent = 'RTX 5060 coverage is currently limited to PIDNet-S and ROD ViT-S. TensorRT results remain available in the Deployment section.';
    } else if (hwMode === 'ryzen') {
      points = comparison.map(row => ({
        name: row.model,
        fps: row.ryzen_compiled_fps,
        latencyMs: Number.isFinite(row.ryzen_compiled_fps) ? 1000 / row.ryzen_compiled_fps : null,
        mIoU: row.mIoU_bg,
        params: row.params,
        fsr: row.fsr_bg,
        color: MODEL_COLORS[row.model] || '#566278'
      })).filter(point => Number.isFinite(point.fps) && Number.isFinite(point.mIoU));
      if (desc) desc.textContent = 'Measured Ryzen 5 5500 compiled throughput paired with the corresponding Blue + Green test checkpoint.';
      if (note) note.textContent = 'ROD ViT-S is omitted because no Ryzen measurement is available; the remaining eight points are measured compiled runs.';
    } else {
      if (desc) desc.textContent = 'Measured H100 eager throughput paired with the corresponding Blue + Green test checkpoint.';
      if (note) note.textContent = 'All nine points use measured H100 throughput and the matching Blue + Green evaluation policy.';
    }

    const W = 900, H = 450;
    const margin = { top: 40, right: 40, bottom: 55, left: 70 };
    const innerW = W - margin.left - margin.right;
    const innerH = H - margin.top - margin.bottom;

    const observedMaxFps = Math.max(...points.map(point => point.fps), 1);
    const maxFps = observedMaxFps <= 50
      ? Math.ceil(observedMaxFps / 10) * 10
      : observedMaxFps <= 150
        ? Math.ceil(observedMaxFps / 25) * 25
        : Math.ceil(observedMaxFps / 50) * 50;
    const minFps = 0;
    const minMIoU = 0.86;
    const maxMIoU = 0.95;

    const xScale = fps => margin.left + (fps / maxFps) * innerW;
    const yScale = miou => margin.top + innerH - ((miou - minMIoU) / (maxMIoU - minMIoU)) * innerH;

    // Grid lines & Axis labels
    let svgHtml = `
      <defs>
        <linearGradient id="pareto-grad" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stop-color="rgba(0, 229, 255, 0.4)" />
          <stop offset="100%" stop-color="rgba(16, 185, 129, 0.4)" />
        </linearGradient>
      </defs>
    `;

    // Horizontal grid lines (mIoU)
    for (let m = minMIoU; m <= maxMIoU + 0.001; m += 0.02) {
      const y = yScale(m);
      svgHtml += `
        <line x1="${margin.left}" y1="${y}" x2="${W - margin.right}" y2="${y}" stroke="rgba(255, 255, 255, 0.06)" stroke-dasharray="3 3" />
        <text x="${margin.left - 10}" y="${y + 4}" fill="#64748b" font-size="11" font-family="var(--font-mono)" text-anchor="end">${m.toFixed(2)}</text>
      `;
    }

    // Vertical grid lines (FPS)
    const stepFps = maxFps <= 50 ? 10 : (maxFps <= 150 ? 25 : 50);
    for (let f = minFps; f <= maxFps; f += stepFps) {
      const x = xScale(f);
      svgHtml += `
        <line x1="${x}" y1="${margin.top}" x2="${x}" y2="${H - margin.bottom}" stroke="rgba(255, 255, 255, 0.06)" stroke-dasharray="3 3" />
        <text x="${x}" y="${H - margin.bottom + 20}" fill="#64748b" font-size="11" font-family="var(--font-mono)" text-anchor="middle">${f}</text>
      `;
    }

    // Axis Labels
    svgHtml += `
      <text x="${margin.left + innerW / 2}" y="${H - 12}" fill="#94a3b8" font-size="12" font-weight="600" text-anchor="middle">
        Inference Speed (${labelHwText(hwMode)} Throughput - FPS) →
      </text>
      <text x="-${margin.top + innerH / 2}" y="22" fill="#94a3b8" font-size="12" font-weight="600" text-anchor="middle" transform="rotate(-90)">
        Traversability Accuracy (Test mIoU) →
      </text>
    `;

    // Compute Pareto optimal frontier
    // Sort points by FPS descending
    const sorted = [...points].sort((a, b) => b.fps - a.fps);
    const pareto = [];
    let currentMaxMIoU = -1;
    for (const p of sorted) {
      if (p.mIoU > currentMaxMIoU) {
        pareto.push(p);
        currentMaxMIoU = p.mIoU;
      }
    }
    pareto.sort((a, b) => a.fps - b.fps);

    // Draw Pareto Curve
    if (pareto.length > 1) {
      const pathData = pareto.map((p, i) => `${i === 0 ? 'M' : 'L'} ${xScale(p.fps)} ${yScale(p.mIoU)}`).join(' ');
      svgHtml += `
        <path d="${pathData}" fill="none" stroke="url(#pareto-grad)" stroke-width="2.5" stroke-dasharray="4 2" />
      `;
    }

    const labelPlacement = {
      'FPN/EfficientNet-B0': { dx: 10, dy: -14, anchor: 'start' },
      'U-Net/EfficientNet-B0': { dx: -10, dy: -12, anchor: 'end' },
      'FPN/MobileNetV2': { dx: 10, dy: -12, anchor: 'start' },
      'SegFormer-B0': { dx: 10, dy: 19, anchor: 'start' },
      'U-Net/MobileNetV2': { dx: -10, dy: 19, anchor: 'end' },
      'ROD ViT-S': { dx: 0, dy: -13, anchor: 'middle' },
      'PIDNet-S': { dx: 0, dy: -13, anchor: 'middle' },
      'BiSeNetV2': { dx: 0, dy: -13, anchor: 'middle' },
      'DDRNet-23-Slim': { dx: 0, dy: -13, anchor: 'middle' }
    };

    // Draw Model Points with deterministic, separated label positions.
    points.forEach(p => {
      const cx = xScale(p.fps);
      const cy = yScale(p.mIoU);
      const isPareto = pareto.some(par => par.name === p.name);
      const label = labelPlacement[p.name] || { dx: 0, dy: -13, anchor: 'middle' };
      const fsrText = Number.isFinite(p.fsr) ? (p.fsr * 100).toFixed(2) : '—';

      svgHtml += `
        <g class="chart-point-group" data-name="${escapeHtml(p.name)}" data-fps="${p.fps.toFixed(1)}" data-lat="${p.latencyMs.toFixed(2)}" data-miou="${p.mIoU.toFixed(4)}" data-fsr="${fsrText}" data-params="${(p.params / 1e6).toFixed(2)}M">
          <circle cx="${cx}" cy="${cy}" r="${isPareto ? 8 : 6}" fill="${p.color}" stroke="#060913" stroke-width="2" style="cursor: pointer; filter: drop-shadow(0 0 6px ${p.color});" />
          <text x="${cx + label.dx}" y="${cy + label.dy}" fill="${p.color}" font-size="10.5" font-weight="700" text-anchor="${label.anchor}" pointer-events="none">${escapeHtml(p.name)}</text>
        </g>
      `;
    });

    svg.innerHTML = svgHtml;

    // Attach Hover Tooltip
    svg.querySelectorAll('.chart-point-group').forEach(el => {
      el.addEventListener('mouseenter', e => {
        const name = el.getAttribute('data-name');
        const fps = el.getAttribute('data-fps');
        const lat = el.getAttribute('data-lat');
        const miou = el.getAttribute('data-miou');
        const fsr = el.getAttribute('data-fsr');
        const params = el.getAttribute('data-params');

        if (tooltip) {
          tooltip.innerHTML = `
            <div style="font-weight: 800; color: ${MODEL_COLORS[name] || '#fff'}; margin-bottom: 0.25rem;">${name}</div>
            <div>Test mIoU: <strong>${miou}</strong></div>
            <div>Throughput: <strong>${fps} FPS</strong> (${lat} ms)</div>
            <div>False-Safe Rate: <strong>${fsr === '—' ? 'not measured in this run' : `${fsr}%`}</strong></div>
            <div>Weights: <strong>${params}</strong></div>
          `;
          tooltip.style.opacity = '1';
        }
      });

      el.addEventListener('mousemove', e => {
        if (!tooltip) return;
        const rect = svg.getBoundingClientRect();
        const x = e.clientX - rect.left + 15;
        const y = e.clientY - rect.top - 20;
        tooltip.style.left = `${x}px`;
        tooltip.style.top = `${y}px`;
      });

      el.addEventListener('mouseleave', () => {
        if (tooltip) tooltip.style.opacity = '0';
      });
    });
  }

  function labelHwText(mode) {
    if (mode === 'rtx5060') return 'RTX 5060 PyTorch eager';
    if (mode === 'ryzen') return 'Ryzen 5500 compiled';
    return 'H100 NVL eager';
  }

  // --- Chart 2: Asymmetric Safety Space (FSR vs FBR) ---
  function renderSafetyChart() {
    const svg = document.getElementById('safety-svg');
    const tooltip = document.getElementById('safety-tooltip');
    if (!svg || !state.data) return;

    const models = state.data.suites.convergence || [];
    const W = 500, H = 380;
    const margin = { top: 30, right: 30, bottom: 50, left: 60 };
    const innerW = W - margin.left - margin.right;
    const innerH = H - margin.top - margin.bottom;

    const minFSR = 0.02, maxFSR = 0.045;
    const minFBR = 0.03, maxFBR = 0.065;

    const xScale = fsr => margin.left + ((fsr - minFSR) / (maxFSR - minFSR)) * innerW;
    const yScale = fbr => margin.top + innerH - ((fbr - minFBR) / (maxFBR - minFBR)) * innerH;

    let svgHtml = `
      <defs>
        <radialGradient id="danger-zone-grad" cx="100%" cy="0%" r="90%">
          <stop offset="0%" stop-color="rgba(244, 63, 94, 0.15)" />
          <stop offset="100%" stop-color="transparent" />
        </radialGradient>
      </defs>
      <!-- Danger Zone Highlight -->
      <rect x="${xScale(0.035)}" y="${margin.top}" width="${W - margin.right - xScale(0.035)}" height="${innerH}" fill="url(#danger-zone-grad)" />
      <text x="${W - margin.right - 10}" y="${margin.top + 20}" fill="rgba(244, 63, 94, 0.7)" font-size="10" font-weight="700" text-anchor="end">HAZARD DANGER ZONE (HIGH FSR)</text>
    `;

    // Grid lines
    for (let f = 0.02; f <= 0.045; f += 0.005) {
      const x = xScale(f);
      svgHtml += `
        <line x1="${x}" y1="${margin.top}" x2="${x}" y2="${H - margin.bottom}" stroke="rgba(255, 255, 255, 0.06)" />
        <text x="${x}" y="${H - margin.bottom + 18}" fill="#64748b" font-size="10" font-family="var(--font-mono)" text-anchor="middle">${(f * 100).toFixed(1)}%</text>
      `;
    }

    for (let b = 0.03; b <= 0.065; b += 0.01) {
      const y = yScale(b);
      svgHtml += `
        <line x1="${margin.left}" y1="${y}" x2="${W - margin.right}" y2="${y}" stroke="rgba(255, 255, 255, 0.06)" />
        <text x="${margin.left - 8}" y="${y + 4}" fill="#64748b" font-size="10" font-family="var(--font-mono)" text-anchor="end">${(b * 100).toFixed(1)}%</text>
      `;
    }

    svgHtml += `
      <text x="${margin.left + innerW / 2}" y="${H - 10}" fill="#94a3b8" font-size="11" font-weight="600" text-anchor="middle">
        False-Safe Rate (FSR - Collision Hazard Risk) →
      </text>
      <text x="-${margin.top + innerH / 2}" y="18" fill="#94a3b8" font-size="11" font-weight="600" text-anchor="middle" transform="rotate(-90)">
        False-Block Rate (FBR - Unnecessary Stops) →
      </text>
    `;

    // Points
    models.forEach(m => {
      const cx = xScale(m.fsr_mean);
      const cy = yScale(m.fbr_mean);
      const color = MODEL_COLORS[m.model] || '#00e5ff';

      svgHtml += `
        <g class="safety-point-group" data-name="${escapeHtml(m.model)}" data-fsr="${(m.fsr_mean * 100).toFixed(2)}" data-fbr="${(m.fbr_mean * 100).toFixed(2)}" data-miou="${m.mIoU_mean.toFixed(4)}">
          <circle cx="${cx}" cy="${cy}" r="6" fill="${color}" stroke="#060913" stroke-width="2" style="cursor: pointer; filter: drop-shadow(0 0 5px ${color});" />
          <text x="${cx}" y="${cy - 10}" fill="${color}" font-size="10" font-weight="700" text-anchor="middle" pointer-events="none">${escapeHtml(m.model.split('/')[0])}</text>
        </g>
      `;
    });

    svg.innerHTML = svgHtml;

    svg.querySelectorAll('.safety-point-group').forEach(el => {
      el.addEventListener('mouseenter', () => {
        const name = el.getAttribute('data-name');
        const fsr = el.getAttribute('data-fsr');
        const fbr = el.getAttribute('data-fbr');
        const miou = el.getAttribute('data-miou');

        if (tooltip) {
          tooltip.innerHTML = `
            <div style="font-weight: 800; color: ${MODEL_COLORS[name] || '#fff'};">${name}</div>
            <div>False-Safe (FSR): <strong class="text-rose">${fsr}%</strong></div>
            <div>False-Block (FBR): <strong class="text-amber">${fbr}%</strong></div>
            <div>Test mIoU: <strong>${miou}</strong></div>
          `;
          tooltip.style.opacity = '1';
        }
      });

      el.addEventListener('mousemove', e => {
        if (!tooltip) return;
        const rect = svg.getBoundingClientRect();
        tooltip.style.left = `${e.clientX - rect.left + 15}px`;
        tooltip.style.top = `${e.clientY - rect.top - 20}px`;
      });

      el.addEventListener('mouseleave', () => {
        if (tooltip) tooltip.style.opacity = '0';
      });
    });
  }

  // --- Chart 3: Gain Delta Breakdown ---
  function renderGainChart() {
    const svg = document.getElementById('gain-svg');
    const tooltip = document.getElementById('gain-tooltip');
    if (!svg || !state.data) return;

    const models = [...(state.data.suites.convergence || [])].sort((a, b) => b.gain_pp - a.gain_pp);
    const W = 500, H = 380;
    const margin = { top: 30, right: 30, bottom: 40, left: 140 };
    const innerW = W - margin.left - margin.right;
    const innerH = H - margin.top - margin.bottom;

    const maxGain = 4.0;
    const barHeight = innerH / models.length - 8;

    let svgHtml = '';

    // Vertical grid lines
    for (let g = 0; g <= maxGain; g += 1.0) {
      const x = margin.left + (g / maxGain) * innerW;
      svgHtml += `
        <line x1="${x}" y1="${margin.top}" x2="${x}" y2="${H - margin.bottom}" stroke="rgba(255, 255, 255, 0.06)" />
        <text x="${x}" y="${H - margin.bottom + 16}" fill="#64748b" font-size="10" font-family="var(--font-mono)" text-anchor="middle">+${g.toFixed(1)} pp</text>
      `;
    }

    svgHtml += `
      <text x="${margin.left + innerW / 2}" y="${H - 6}" fill="#94a3b8" font-size="11" font-weight="600" text-anchor="middle">
        mIoU Gain over Fixed 15-Epoch Budget (+pp)
      </text>
    `;

    // Horizontal bars
    models.forEach((m, idx) => {
      const y = margin.top + idx * (innerH / models.length);
      const barW = (m.gain_pp / maxGain) * innerW;
      const color = MODEL_COLORS[m.model] || '#00e5ff';
      const fixedVal = (typeof m.fixed_15_mIoU === 'object' && m.fixed_15_mIoU !== null)
        ? (m.fixed_15_mIoU.mean || 0)
        : (Number(m.fixed_15_mIoU) || (m.mIoU_mean - (m.gain_pp || 0) / 100));

      svgHtml += `
        <g class="gain-bar-group" data-name="${escapeHtml(m.model)}" data-gain="${m.gain_pp.toFixed(2)}" data-fixed="${fixedVal.toFixed(4)}" data-conv="${m.mIoU_mean.toFixed(4)}">
          <text x="${margin.left - 10}" y="${y + barHeight / 2 + 4}" fill="#f1f5f9" font-size="11" font-weight="600" text-anchor="end">${escapeHtml(m.model)}</text>
          <rect x="${margin.left}" y="${y}" width="${barW}" height="${barHeight}" rx="4" fill="${color}" style="cursor: pointer; opacity: 0.85;" />
          <text x="${margin.left + barW + 8}" y="${y + barHeight / 2 + 4}" fill="${color}" font-size="10" font-family="var(--font-mono)" font-weight="700">+${m.gain_pp.toFixed(2)} pp</text>
        </g>
      `;
    });

    svg.innerHTML = svgHtml;

    svg.querySelectorAll('.gain-bar-group').forEach(el => {
      el.addEventListener('mouseenter', () => {
        const name = el.getAttribute('data-name');
        const gain = el.getAttribute('data-gain');
        const fixed = el.getAttribute('data-fixed');
        const conv = el.getAttribute('data-conv');

        if (tooltip) {
          tooltip.innerHTML = `
            <div style="font-weight: 800; color: ${MODEL_COLORS[name] || '#fff'};">${name}</div>
            <div>Convergence mIoU: <strong>${conv}</strong></div>
            <div>Fixed 15e mIoU: <strong>${fixed}</strong></div>
            <div class="text-cyan">Accuracy Boost: <strong>+${gain} pp</strong></div>
          `;
          tooltip.style.opacity = '1';
        }
      });

      el.addEventListener('mousemove', e => {
        if (!tooltip) return;
        const rect = svg.getBoundingClientRect();
        tooltip.style.left = `${e.clientX - rect.left + 15}px`;
        tooltip.style.top = `${e.clientY - rect.top - 20}px`;
      });

      el.addEventListener('mouseleave', () => {
        if (tooltip) tooltip.style.opacity = '0';
      });
    });
  }

  // --- Chart 4: Full Convergence Epoch-by-Epoch Curves ---
  function renderConvergenceCurves() {
    const svg = document.getElementById('curves-svg');
    const tooltip = document.getElementById('curves-tooltip');
    const togglesContainer = document.getElementById('curve-model-toggles');
    if (!svg || !state.data || !state.data.curves) return;

    const curves = state.data.curves;
    const metric = state.curveMetric; // 'val_mIoU' | 'val_loss' | 'val_fsr'

    // Render model toggle pills if not already rendered
    if (togglesContainer && togglesContainer.children.length === 0) {
      Object.keys(curves).forEach(name => {
        const color = MODEL_COLORS[name] || '#00e5ff';
        const isActive = state.activeCurveModels.has(name);
        const btn = document.createElement('button');
        btn.className = `curve-pill ${isActive ? 'active' : ''}`;
        btn.style.color = color;
        btn.textContent = name;
        btn.addEventListener('click', () => {
          if (state.activeCurveModels.has(name)) {
            state.activeCurveModels.delete(name);
            btn.classList.remove('active');
          } else {
            state.activeCurveModels.add(name);
            btn.classList.add('active');
          }
          renderConvergenceCurves();
        });
        togglesContainer.appendChild(btn);
      });
    }

    const W = 900, H = 400;
    const margin = { top: 30, right: 40, bottom: 45, left: 60 };
    const innerW = W - margin.left - margin.right;
    const innerH = H - margin.top - margin.bottom;

    const maxEpoch = 150;
    let minY = 0.82, maxY = 0.95;
    if (metric === 'val_loss') { minY = 0.12; maxY = 0.45; }
    if (metric === 'val_fsr') { minY = 0.015; maxY = 0.07; }

    const xScale = epoch => margin.left + (epoch / maxEpoch) * innerW;
    const yScale = val => margin.top + innerH - ((val - minY) / (maxY - minY)) * innerH;

    let svgHtml = '';

    // Horizontal grid lines
    const steps = 6;
    for (let i = 0; i <= steps; i++) {
      const val = minY + (i / steps) * (maxY - minY);
      const y = yScale(val);
      svgHtml += `
        <line x1="${margin.left}" y1="${y}" x2="${W - margin.right}" y2="${y}" stroke="rgba(255, 255, 255, 0.06)" />
        <text x="${margin.left - 8}" y="${y + 4}" fill="#64748b" font-size="10" font-family="var(--font-mono)" text-anchor="end">${val.toFixed(metric === 'val_fsr' ? 3 : 2)}</text>
      `;
    }

    // Vertical grid lines (epochs)
    for (let ep = 0; ep <= maxEpoch; ep += 25) {
      const x = xScale(ep);
      svgHtml += `
        <line x1="${x}" y1="${margin.top}" x2="${x}" y2="${H - margin.bottom}" stroke="rgba(255, 255, 255, 0.06)" />
        <text x="${x}" y="${H - margin.bottom + 16}" fill="#64748b" font-size="10" font-family="var(--font-mono)" text-anchor="middle">Ep ${ep}</text>
      `;
    }

    svgHtml += `
      <text x="${margin.left + innerW / 2}" y="${H - 8}" fill="#94a3b8" font-size="11" font-weight="600" text-anchor="middle">
        Training Epoch (Cosine Decay 60, Ceiling 300) →
      </text>
      <text x="-${margin.top + innerH / 2}" y="18" fill="#94a3b8" font-size="11" font-weight="600" text-anchor="middle" transform="rotate(-90)">
        ${metric === 'val_mIoU' ? 'Validation mIoU' : (metric === 'val_loss' ? 'Validation Loss' : 'False-Safe Rate')}
      </text>
    `;

    // Draw active curves
    Object.entries(curves).forEach(([name, points]) => {
      if (!state.activeCurveModels.has(name) || !points || points.length === 0) return;

      const color = MODEL_COLORS[name] || '#38bdf8';
      let pathStr = '';

      points.forEach((pt, i) => {
        const val = pt[metric];
        if (val === undefined || val === null || pt.epoch > maxEpoch) return;
        const x = xScale(pt.epoch);
        const y = yScale(Math.max(minY, Math.min(maxY, val)));

        if (!pathStr) pathStr += `M ${x} ${y}`;
        else pathStr += ` L ${x} ${y}`;
      });

      if (pathStr) {
        svgHtml += `
          <path d="${pathStr}" fill="none" stroke="${color}" stroke-width="2" stroke-linejoin="round" opacity="0.85" />
        `;
      }
    });

    svg.innerHTML = svgHtml;
  }

  // =========================================================================
  // 4. Interactive Mask Split Inspector
  // =========================================================================

  function setupSplitSlider() {
    const container = document.getElementById('split-container');
    const divider = document.getElementById('split-divider');
    const fgLayer = document.getElementById('split-layer-fg');

    if (!container || !divider || !fgLayer) return;

    let isDragging = false;

    function setSplit(pct) {
      pct = Math.max(0, Math.min(100, pct));
      state.splitPercent = pct;
      container.style.setProperty('--split-pos', `${pct}%`);
      fgLayer.style.clipPath = `inset(0 ${100 - pct}% 0 0)`;
      fgLayer.style.webkitClipPath = `inset(0 ${100 - pct}% 0 0)`;
      divider.style.left = `${pct}%`;
      const handle = divider.querySelector('.split-handle');
      if (handle) {
        const rounded = Math.round(pct);
        handle.setAttribute('aria-valuenow', String(rounded));
        handle.setAttribute('aria-valuetext', `${rounded} percent`);
      }
    }

    function calculatePercent(clientX) {
      const rect = container.getBoundingClientRect();
      let pos = (clientX - rect.left) / rect.width;
      pos = Math.max(0.01, Math.min(0.99, pos));
      return (pos * 100).toFixed(2);
    }

    function onMove(e) {
      if (!isDragging) return;
      const clientX = e.touches ? e.touches[0].clientX : e.clientX;
      setSplit(calculatePercent(clientX));
    }

    function startDrag(e) {
      if (state.splitMode === 'triptych') return;
      isDragging = true;
      const clientX = e.touches ? e.touches[0].clientX : e.clientX;
      setSplit(calculatePercent(clientX));
      document.body.style.userSelect = 'none';
    }

    function stopDrag() {
      isDragging = false;
      document.body.style.userSelect = '';
    }

    container.addEventListener('mousedown', startDrag);
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', stopDrag);

    container.addEventListener('touchstart', startDrag, { passive: true });
    window.addEventListener('touchmove', onMove, { passive: true });
    window.addEventListener('touchend', stopDrag);

    const handle = divider.querySelector('.split-handle');
    if (handle) {
      handle.addEventListener('keydown', event => {
        if (state.splitMode === 'triptych') return;
        const step = event.shiftKey ? 10 : 2;
        if (event.key === 'ArrowLeft') setSplit(state.splitPercent - step);
        else if (event.key === 'ArrowRight') setSplit(state.splitPercent + step);
        else if (event.key === 'Home') setSplit(0);
        else if (event.key === 'End') setSplit(100);
        else return;
        event.preventDefault();
      });
    }

    // Initial 50% split
    setSplit(50);
  }

  function updateSplitInspectorSample(sampleId) {
    if (!state.data) return;

    state.currentSampleId = sampleId;

    const imgBg = document.getElementById('split-img-bg');
    const imgFg = document.getElementById('split-img-fg');
    const titleEl = document.getElementById('inspector-sample-title');
    const descEl = document.getElementById('inspector-sample-desc');
    const fsrBadge = document.getElementById('inspector-fsr-badge');
    const fbrBadge = document.getElementById('inspector-fbr-badge');

    // Find qualitative details or test sample item
    let meta = null;
    if (state.data.qualitative) {
      for (const cat of Object.values(state.data.qualitative)) {
        if (Array.isArray(cat)) {
          const found = cat.find(item => item.id === sampleId || item.title.includes(sampleId));
          if (found) { meta = found; break; }
        }
      }
    }

    const title = meta ? meta.title : `Test Sample ${sampleId}`;
    const desc = meta ? meta.description : `640x384 test set prediction from official CaT partition.`;
    const fsr = meta && meta.fsr !== undefined ? (meta.fsr * 100).toFixed(1) + '%' : '4.3%';
    const fbr = meta && meta.fbr !== undefined ? (meta.fbr * 100).toFixed(1) + '%' : '7.0%';

    if (titleEl) titleEl.textContent = title;
    if (descEl) descEl.textContent = desc;
    if (fsrBadge) fsrBadge.textContent = `False-Safe: ${fsr}`;
    if (fbrBadge) fbrBadge.textContent = `False-Block: ${fbr}`;

    // Update images according to splitMode
    const mode = state.splitMode;
    const labelLeft = document.getElementById('split-label-left');
    const labelRight = document.getElementById('split-label-right');

    const rgbUrl = `./media/samples/images/${sampleId}.png`;
    const maskUrl = `./media/samples/masks/${sampleId}.png`;
    const overlayUrl = `./media/samples/overlays/${sampleId}.png`;
    const triptychUrl = `./media/samples/triptychs/${sampleId}.png`;

    const splitContainer = document.getElementById('split-container');
    if (mode === 'rgb_pred') {
      if (splitContainer) splitContainer.classList.remove('triptych-mode');
      if (imgFg) imgFg.src = rgbUrl;
      if (imgBg) imgBg.src = overlayUrl;
      if (labelLeft) labelLeft.textContent = 'Raw Camera RGB';
      if (labelRight) labelRight.textContent = 'Prediction Overlay';
    } else if (mode === 'gt_pred') {
      if (splitContainer) splitContainer.classList.remove('triptych-mode');
      if (imgFg) imgFg.src = maskUrl;
      if (imgBg) imgBg.src = overlayUrl;
      if (labelLeft) labelLeft.textContent = 'Ground Truth Mask';
      if (labelRight) labelRight.textContent = 'Prediction Overlay';
    } else if (mode === 'triptych') {
      if (splitContainer) splitContainer.classList.add('triptych-mode');
      if (imgBg) imgBg.src = triptychUrl;
      if (labelLeft) labelLeft.textContent = 'Full Triptych (RGB • GT • Prediction)';
      if (labelRight) labelRight.textContent = '';
    }
  }

  // =========================================================================
  // 5. Qualitative Review Gallery & 544 Browser
  // =========================================================================

  function renderGallery() {
    if (!state.data) return;

    const container = document.getElementById('gallery-card-grid');
    const searchBar = document.getElementById('browser-search-bar');
    if (!container) return;

    const cat = state.galleryCategory;

    // If 544 Browser Mode
    if (cat === 'all_544_browser') {
      if (searchBar) searchBar.classList.remove('hidden');
      render544Browser(container);
      return;
    }

    if (searchBar) searchBar.classList.add('hidden');

    let items = [];
    const qData = state.data.qualitative || {};

    if (cat === 'all') {
      Object.entries(qData).forEach(([cName, list]) => {
        if (Array.isArray(list)) items.push(...list);
      });
    } else if (qData[cat]) {
      items = qData[cat];
    }

    container.innerHTML = items.map(item => {
      const thumb = item.triptych_url || item.overlay_url || item.img_url || item.detail_img;
      const badgeClass = item.category === 'worst_false_safe' ? 'danger' :
                         (item.category === 'worst_false_block' ? 'caution' : 'cyan');

      return `
        <div class="gallery-card" data-id="${escapeHtml(item.id)}" data-thumb="${escapeHtml(thumb)}" data-title="${escapeHtml(item.title)}" data-desc="${escapeHtml(item.description)}">
          <div class="gallery-thumb-wrap" role="button" tabindex="0" aria-label="Open ${escapeHtml(item.title)} image preview">
            <img src="${escapeHtml(thumb)}" alt="${escapeHtml(item.title)}" loading="lazy">
            <span class="gallery-card-badge fsr-tag ${badgeClass}">
              ${escapeHtml(item.category.replace(/_/g, ' '))}
            </span>
          </div>
          <div class="gallery-card-body">
            <div>
              <h4 class="gallery-card-title">${escapeHtml(item.title)}</h4>
              <p class="gallery-card-desc">${escapeHtml(item.description)}</p>
            </div>
            <div class="gallery-card-footer">
              <span>Sample: ${escapeHtml(item.id)}</span>
              <button type="button" class="btn-inspect-direct" data-id="${escapeHtml(item.id)}" title="Inspect in Interactive Mask Inspector">
                Inspect
              </button>
            </div>
          </div>
        </div>
      `;
    }).join('');

    // Attach inspect button clicks (directly launches Mask Inspector)
    container.querySelectorAll('.btn-inspect-direct').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const id = btn.getAttribute('data-id');
        openInInspector(id);
      });
    });

    // Attach thumbnail click to open high-res lightbox
    container.querySelectorAll('.gallery-thumb-wrap').forEach(wrap => {
      wrap.addEventListener('click', (e) => {
        e.stopPropagation();
        const card = wrap.closest('.gallery-card');
        if (!card) return;
        const id = card.getAttribute('data-id');
        const thumb = card.getAttribute('data-thumb');
        const title = card.getAttribute('data-title');
        const desc = card.getAttribute('data-desc');
        openImageLightbox(thumb, title, desc, id);
      });
      wrap.addEventListener('keydown', event => {
        if (event.key !== 'Enter' && event.key !== ' ') return;
        event.preventDefault();
        wrap.click();
      });
    });
  }

  function render544Browser(container) {
    const samples = state.data.test_samples || [];
    let filtered = samples;

    if (state.browserQuery.trim()) {
      const q = state.browserQuery.toLowerCase();
      filtered = filtered.filter(s => s.id.toLowerCase().includes(q));
    }

    const total = filtered.length;
    const startIdx = (state.browserPage - 1) * state.browserPerPage;
    const paged = filtered.slice(startIdx, startIdx + state.browserPerPage);

    const info = document.getElementById('browser-pagination-info');
    if (info) {
      info.textContent = total
        ? `Showing ${startIdx + 1}–${Math.min(startIdx + state.browserPerPage, total)} of ${total} packaged samples`
        : 'No packaged samples match that ID';
    }

    const prevButton = document.getElementById('btn-prev-page');
    const nextButton = document.getElementById('btn-next-page');
    if (prevButton) prevButton.disabled = state.browserPage <= 1;
    if (nextButton) nextButton.disabled = startIdx + state.browserPerPage >= total;

    if (!paged.length) {
      container.innerHTML = '<p class="empty-state">No samples found. Try a shorter sample ID.</p>';
      return;
    }

    container.innerHTML = paged.map(s => `
      <div class="gallery-card" data-id="${escapeHtml(s.id)}" data-thumb="${escapeHtml(s.triptych_url)}" data-title="Test Sample: ${escapeHtml(s.id)}" data-desc="Official test sample triptych (RGB, Ground Truth, Prediction).">
        <div class="gallery-thumb-wrap" role="button" tabindex="0" aria-label="Open ${escapeHtml(s.id)} image preview">
          <img src="${escapeHtml(s.triptych_url)}" alt="${escapeHtml(s.id)}" loading="lazy">
          <span class="gallery-card-badge fsr-tag cyan">Test Set</span>
        </div>
        <div class="gallery-card-body">
          <div>
            <h4 class="gallery-card-title">${escapeHtml(s.id)}</h4>
            <p class="gallery-card-desc">Full 3-in-1 test verification triptych.</p>
          </div>
          <div class="gallery-card-footer">
            <span>640x384</span>
            <button type="button" class="btn-inspect-direct" data-id="${escapeHtml(s.id)}" title="Inspect in Interactive Mask Inspector">
              Inspect
            </button>
          </div>
        </div>
      </div>
    `).join('');

    container.querySelectorAll('.btn-inspect-direct').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const id = btn.getAttribute('data-id');
        openInInspector(id);
      });
    });

    container.querySelectorAll('.gallery-thumb-wrap').forEach(wrap => {
      wrap.addEventListener('click', (e) => {
        e.stopPropagation();
        const card = wrap.closest('.gallery-card');
        if (!card) return;
        const id = card.getAttribute('data-id');
        const thumb = card.getAttribute('data-thumb');
        const title = card.getAttribute('data-title');
        const desc = card.getAttribute('data-desc');
        openImageLightbox(thumb, title, desc, id);
      });
      wrap.addEventListener('keydown', event => {
        if (event.key !== 'Enter' && event.key !== ' ') return;
        event.preventDefault();
        wrap.click();
      });
    });
  }

  // Open Sample directly in Interactive Mask Inspector
  function openInInspector(sampleId) {
    if (!sampleId) return;

    // Close any open modals
    const imgBackdrop = document.getElementById('image-modal-backdrop');
    if (imgBackdrop?.classList.contains('open')) closeModal(imgBackdrop);
    const seedBackdrop = document.getElementById('seed-modal-backdrop');
    if (seedBackdrop?.classList.contains('open')) closeModal(seedBackdrop);

    // Ensure sample exists in inspector dropdown
    const select = document.getElementById('inspector-sample-select');
    if (select) {
      let found = false;
      for (let i = 0; i < select.options.length; i++) {
        if (select.options[i].value === sampleId) {
          select.selectedIndex = i;
          found = true;
          break;
        }
      }
      if (!found) {
        const opt = document.createElement('option');
        opt.value = sampleId;
        opt.textContent = `${sampleId} • Selected Sample`;
        select.appendChild(opt);
        select.value = sampleId;
      }
    }

    // Update the inspector with this sample
    updateSplitInspectorSample(sampleId);

    // Switch view to Interactive Mask Inspector tab
    switchView('view-inspector');

    // Smoothly scroll to the split container
    setTimeout(() => {
      const container = document.getElementById('split-container');
      if (container) {
        container.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    }, 50);
  }

  // Lightbox Modal
  function openImageLightbox(imgUrl, title, desc, sampleId) {
    const backdrop = document.getElementById('image-modal-backdrop');
    const titleEl = document.getElementById('lightbox-title');
    const subtitleEl = document.getElementById('lightbox-subtitle');
    const imgEl = document.getElementById('lightbox-img');
    const captionEl = document.getElementById('lightbox-caption');
    const btnLoad = document.getElementById('btn-lightbox-load-inspector');

    if (titleEl) titleEl.textContent = title;
    if (subtitleEl) subtitleEl.textContent = `Sample ID: ${sampleId}`;
    if (imgEl) imgEl.src = imgUrl;
    if (captionEl) captionEl.textContent = desc;

    if (btnLoad) {
      btnLoad.onclick = () => {
        openInInspector(sampleId);
      };
    }

    openModal(backdrop, document.getElementById('image-modal-dialog'));
  }

  function updatePackagedSampleCount() {
    const count = state.data?.test_samples?.length || 0;
    const browseButton = document.getElementById('btn-browse-test-set');
    if (browseButton) browseButton.textContent = `Browse packaged test set (${count})`;
  }

  function openModal(backdrop, dialog) {
    if (!backdrop || !dialog) return;
    lastFocusedElement = document.activeElement;
    backdrop.classList.add('open');
    backdrop.setAttribute('aria-hidden', 'false');
    document.body.classList.add('modal-open');
    requestAnimationFrame(() => dialog.focus());
  }

  function closeModal(backdrop) {
    if (!backdrop || !backdrop.classList.contains('open')) return;
    backdrop.classList.remove('open');
    backdrop.setAttribute('aria-hidden', 'true');
    if (!document.querySelector('.modal-backdrop.open')) {
      document.body.classList.remove('modal-open');
    }
    if (lastFocusedElement instanceof HTMLElement) lastFocusedElement.focus();
  }

  // =========================================================================
  // 6. Thesis Theory & Conceptual Diagrams
  // =========================================================================

  function renderTheoryGallery() {
    if (!state.data || !state.data.thesis_figures) return;

    const container = document.getElementById('theory-card-grid');
    if (!container) return;

    container.innerHTML = state.data.thesis_figures.map(f => `
      <div class="theory-card" role="button" tabindex="0" aria-label="Open ${escapeHtml(f.title)}" data-img="${escapeHtml(f.img_url)}" data-title="${escapeHtml(f.title)}" data-desc="${escapeHtml(f.description)}">
        <div class="theory-img-wrap">
          <img src="${escapeHtml(f.img_url)}" alt="${escapeHtml(f.title)}" loading="lazy">
        </div>
        <div class="theory-card-body">
          <h4 class="theory-card-title">${escapeHtml(f.title)}</h4>
          <p class="theory-card-desc">${escapeHtml(f.description)}</p>
        </div>
      </div>
    `).join('');

    container.querySelectorAll('.theory-card').forEach(card => {
      card.addEventListener('click', () => {
        const img = card.getAttribute('data-img');
        const title = card.getAttribute('data-title');
        const desc = card.getAttribute('data-desc');
        openImageLightbox(img, title, desc, 'Thesis Figure');
      });
      card.addEventListener('keydown', event => {
        if (event.key !== 'Enter' && event.key !== ' ') return;
        event.preventDefault();
        card.click();
      });
    });
  }

  // =========================================================================
  // Event Listeners & UI Controls
  // =========================================================================

  function setupEventListeners() {
    // Policy Selector in Leaderboard
    const policyPills = document.querySelectorAll('#leaderboard-policy-selector .pill-btn');
    policyPills.forEach(btn => {
      btn.addEventListener('click', () => {
        policyPills.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        state.currentPolicy = btn.getAttribute('data-policy');
        renderLeaderboard();
      });
    });

    // Leaderboard Search
    const searchInput = document.getElementById('leaderboard-search-input');
    if (searchInput) {
      searchInput.addEventListener('input', e => {
        state.searchQuery = e.target.value;
        renderLeaderboard();
      });
    }

    // Family Select
    const familySelect = document.getElementById('filter-family-select');
    if (familySelect) {
      familySelect.addEventListener('change', e => {
        state.familyFilter = e.target.value;
        renderLeaderboard();
      });
    }

    // Sort Metric Select
    const sortSelect = document.getElementById('sort-metric-select');
    if (sortSelect) {
      sortSelect.addEventListener('change', e => {
        const val = e.target.value;
        if (val === 'fsr' || val === 'fbr') {
          state.currentSort = { key: val, dir: 'asc' }; // lower is better
        } else {
          state.currentSort = { key: val, dir: 'desc' };
        }
        renderLeaderboard();
      });
    }

    // Table Header Sort Clicks
    document.querySelectorAll('.benchmark-table th.sortable').forEach(th => {
      th.tabIndex = 0;
      th.setAttribute('role', 'button');
      const applySort = () => {
        const sortKey = th.getAttribute('data-sort');
        if (state.currentSort.key === sortKey) {
          state.currentSort.dir = state.currentSort.dir === 'asc' ? 'desc' : 'asc';
        } else {
          state.currentSort.key = sortKey;
          state.currentSort.dir = (sortKey === 'fsr' || sortKey === 'fbr') ? 'asc' : 'desc';
        }
        renderLeaderboard();
      };
      th.addEventListener('click', applySort);
      th.addEventListener('keydown', event => {
        if (event.key !== 'Enter' && event.key !== ' ') return;
        event.preventDefault();
        applySort();
      });
    });

    // Pareto Hardware Selector
    const hwPills = document.querySelectorAll('#pareto-hardware-selector .pill-btn');
    hwPills.forEach(btn => {
      btn.addEventListener('click', () => {
        hwPills.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        state.paretoHardware = btn.getAttribute('data-hw');
        renderParetoChart();
      });
    });

    // Curve Metric Selector
    const curveMetricPills = document.querySelectorAll('#curve-metric-selector .pill-btn');
    curveMetricPills.forEach(btn => {
      btn.addEventListener('click', () => {
        curveMetricPills.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        state.curveMetric = btn.getAttribute('data-metric');
        renderConvergenceCurves();
      });
    });

    // Split Mode Selector
    const splitPills = document.querySelectorAll('#split-mode-selector .pill-btn');
    splitPills.forEach(btn => {
      btn.addEventListener('click', () => {
        splitPills.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        state.splitMode = btn.getAttribute('data-mode');
        updateSplitInspectorSample(state.currentSampleId);
      });
    });

    // Inspector Sample Select Dropdown
    const sampleSelect = document.getElementById('inspector-sample-select');
    if (sampleSelect) {
      sampleSelect.addEventListener('change', e => {
        updateSplitInspectorSample(e.target.value);
      });
    }

    // Gallery Category Pills
    const galPills = document.querySelectorAll('#gallery-category-selector .pill-btn');
    galPills.forEach(btn => {
      btn.addEventListener('click', () => {
        galPills.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        state.galleryCategory = btn.getAttribute('data-cat');
        state.browserPage = 1;
        renderGallery();
      });
    });

    // 544 Browser Search
    const browserSearch = document.getElementById('browser-search-input');
    if (browserSearch) {
      browserSearch.addEventListener('input', e => {
        state.browserQuery = e.target.value;
        state.browserPage = 1;
        renderGallery();
      });
    }

    // Browser Pagination Buttons
    const btnPrev = document.getElementById('btn-prev-page');
    const btnNext = document.getElementById('btn-next-page');
    if (btnPrev) {
      btnPrev.addEventListener('click', () => {
        if (state.browserPage > 1) {
          state.browserPage--;
          renderGallery();
        }
      });
    }
    if (btnNext) {
      btnNext.addEventListener('click', () => {
        state.browserPage++;
        renderGallery();
      });
    }

    // Modal Close Buttons
    const btnCloseSeed = document.getElementById('btn-close-seed-modal');
    if (btnCloseSeed) {
      btnCloseSeed.addEventListener('click', () => {
        closeModal(document.getElementById('seed-modal-backdrop'));
      });
    }

    const btnCloseLightbox = document.getElementById('btn-close-lightbox');
    const btnLightboxClose = document.getElementById('btn-lightbox-close');
    if (btnCloseLightbox) {
      btnCloseLightbox.addEventListener('click', () => {
        closeModal(document.getElementById('image-modal-backdrop'));
      });
    }
    if (btnLightboxClose) {
      btnLightboxClose.addEventListener('click', () => {
        closeModal(document.getElementById('image-modal-backdrop'));
      });
    }

    // Close on backdrop click
    document.querySelectorAll('.modal-backdrop').forEach(bd => {
      bd.addEventListener('click', e => {
        if (e.target === bd) closeModal(bd);
      });
    });

    document.addEventListener('keydown', event => {
      if (event.key !== 'Escape') return;
      const openBackdrop = document.querySelector('.modal-backdrop.open');
      if (openBackdrop) closeModal(openBackdrop);
    });
  }

  // Utilities
  function escapeHtml(str) {
    if (str === null || str === undefined) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

})();
