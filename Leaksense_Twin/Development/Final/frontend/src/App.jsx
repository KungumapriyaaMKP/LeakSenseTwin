import { useState, useEffect, useRef, useCallback } from 'react'
import './index.css'
import DAXConsole from './DAXConsole'
import ChatBot from './ChatBot'

// ─── Intro Video Component ──────────────────────────────────────
function IntroVideo({ onComplete }) {
  const videoRef = useRef(null);

  return (
    <div className="intro-container">
      <video 
        ref={videoRef}
        autoPlay 
        className="intro-video"
        onEnded={onComplete}
      >
        <source src="/intro_video.mp4" type="video/mp4" />
        Your browser does not support the video tag.
      </video>
      <button className="skip-btn" onClick={onComplete}>
        Skip Intro
      </button>
    </div>
  );
}

// ─── Engine Selector ──────────────────────────────────────────────
function EngineSelector({ engines, onSelect, isLoading, error }) {
  return (
    <div className="engine-select-container">
      <div className="engine-select-title">
        <h1>Select Engine to Test</h1>
        <p>Choose which Caterpillar engine is on the test stand. You can switch later from the dashboard header.</p>
        {error && <p style={{ color: 'var(--accent-red)', marginTop: '10px' }}>{error}</p>}
      </div>
      <div className="engine-select-grid">
        {engines.map(eng => (
          <div
            key={eng.id}
            className={`engine-select-card ${!eng.trained || isLoading ? 'disabled' : ''}`}
            onClick={() => eng.trained && !isLoading && onSelect(eng.id)}
          >
            <div className="engine-name">{eng.name}</div>
            <div className="engine-desc">{eng.description}</div>
            <div className="engine-badges">
              <span className="engine-badge-pill" style={{
                background: eng.calibrated ? 'rgba(16,185,129,0.15)' : 'rgba(252,212,65,0.15)',
                color: eng.calibrated ? 'var(--accent-emerald)' : 'var(--accent-yellow)',
              }}>
                {eng.calibrated ? 'Calibrated' : 'Uncalibrated'}
              </span>
              <span className="engine-badge-pill" style={{
                background: eng.trained ? 'rgba(16,185,129,0.15)' : 'rgba(239,68,68,0.15)',
                color: eng.trained ? 'var(--accent-emerald)' : 'var(--accent-red)',
              }}>
                {eng.trained ? 'Model Ready' : 'Not Trained Yet'}
              </span>
              <span className="engine-badge-pill" style={{ background: 'rgba(255,255,255,0.08)', color: 'var(--text-secondary)' }}>
                {eng.n_zones} zones
              </span>
            </div>
          </div>
        ))}
        {engines.length === 0 && !isLoading && (
          <p style={{ color: 'var(--text-secondary)' }}>No engines found — is the backend running?</p>
        )}
      </div>
    </div>
  )
}

const API_BASE = 'http://localhost:8000'
const WS_BASE = 'ws://localhost:8000'

const ZONE_NAMES = [
  'No Leak / Healthy',
  'Zone 1 — Intake',
  'Zone 2 — Charge Air',
  'Zone 3 — CAC/Manifold',
  'Zone 4 — Exhaust Manifold',
  'Zone 5 — DPF/SCR',
  'Zone 6 — Turbine Outlet/Pre-DPF',
]

const EF_CHANNELS = ['MAF', 'MAP_boost', 'MAP_cac_out', 'T_cac_out', 'T_exh', 'dP_dpf']

const SENSOR_UNITS = {
  RPM: 'rpm', MAF: 'kg/h', MAP_intake: 'kPa', MAP_boost: 'kPa',
  MAP_cac_in: 'kPa', MAP_cac_out: 'kPa', T_intake: '°C', T_boost: '°C',
  T_cac_out: '°C', T_exh_manifold: '°C', T_dpf_in: '°C', T_dpf_out: '°C',
  fuel_qty: 'mg/str', dP_dpf: 'kPa',
}

// ─── Sidebar Component ──────────────────────────────────────────
function Sidebar({ activePage, setActivePage, isConnected, engineName, onSwitchEngine }) {
  const navItems = [
    { id: 'dashboard', icon: '', label: 'Dashboard' },
    { id: 'engine', icon: '', label: 'Analysis' },
    { id: 'history', icon: '', label: 'History' },
    { id: 'dax', icon: '', label: 'DAX Console' },
    { id: 'digital-twin', icon: '', label: 'Digital Twin' },
  ]

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="sidebar-logo">
          <div className="sidebar-logo-icon">L</div>
          <div>
            <h1>EngineIQ</h1>
            <span>{engineName || 'Engine Diagnostics'}</span>
          </div>
        </div>
      </div>

      <nav className="sidebar-nav">
        {navItems.map(item => (
          <button
            key={item.id}
            id={`nav-${item.id}`}
            className={`nav-item ${activePage === item.id ? 'active' : ''}`}
            onClick={() => setActivePage(item.id)}
          >
            <span className="nav-icon">{item.icon}</span>
            {item.label}
          </button>
        ))}
      </nav>

      <div className="sidebar-footer">
        <div className="engine-badge" style={{ cursor: 'pointer' }} onClick={onSwitchEngine} title="Switch engine">
          <span className="dot" style={{
            background: isConnected ? 'var(--accent-emerald)' : 'var(--accent-red)',
          }}></span>
          <span>{engineName || 'No Engine'} • Switch</span>
        </div>
      </div>
    </aside>
  )
}

// ─── Go/No-Go Indicator ─────────────────────────────────────────
function GoNoGoIndicator({ status }) {
  const isGo = status === 'GO'
  return (
    <div className={`card go-nogo-card ${isGo ? 'go-indicator' : 'nogo-indicator'}`}>
      <div className="card-header">
        <span className="card-title">Status</span>
        <span className="card-icon"></span>
      </div>
      <div className={`go-nogo-status ${isGo ? 'go' : 'nogo'}`}>
        {status}
      </div>
      <div className="go-nogo-subtitle">
        {isGo ? 'Engine Healthy' : 'Leak Detected'}
      </div>
    </div>
  )
}

// ─── Confidence Ring ─────────────────────────────────────────────
function ConfidenceRing({ confidence, leakDetected }) {
  const pct = Math.round(confidence * 100)
  const radius = 54
  const circumference = 2 * Math.PI * radius
  const offset = circumference - (pct / 100) * circumference

  let color = 'var(--accent-emerald)'
  if (confidence > 0.7) color = 'var(--accent-red)'
  else if (confidence > 0.4) color = 'var(--accent-yellow)'

  return (
    <div className="card confidence-card">
      <div className="card-header">
        <span className="card-title">Confidence</span>
        <span className="card-icon"></span>
      </div>
      <div className="confidence-ring">
        <svg viewBox="0 0 128 128">
          <circle className="ring-bg" cx="64" cy="64" r={radius} />
          <circle
            className="ring-fill"
            cx="64" cy="64" r={radius}
            stroke={color}
            strokeDasharray={circumference}
            strokeDashoffset={offset}
          />
        </svg>
        <div className="confidence-value" style={{ color }}>{pct}%</div>
      </div>
      <div className="confidence-label">
        {leakDetected ? 'Anomaly Detected' : 'Normal Operation'}
      </div>
    </div>
  )
}

// ─── Leak Alert Card ─────────────────────────────────────────────
function LeakAlertCard({ data }) {
  const getSeverityClass = (sev) => {
    const map = { NONE: 'severity-none', SMALL: 'severity-small', MEDIUM: 'severity-medium', CRITICAL: 'severity-critical' }
    return map[sev] || 'severity-none'
  }

  return (
    <div className="card alert-card">
      <div className="card-header">
        <span className="card-title">Leak Analysis</span>
        <span className="card-icon"></span>
      </div>
      <div className="alert-content">
        <div className="alert-zone">{data.suspected_zone || ZONE_NAMES[0]}</div>
        <div>
          <span className={`alert-severity ${getSeverityClass(data.severity)}`}>
            {data.severity || 'NONE'}
          </span>
        </div>
        {data.flow_loss_pct > 0 && (
          <div className="alert-flow-loss">
            Estimated Flow Loss: <strong>{data.flow_loss_pct?.toFixed(1)}%</strong>
          </div>
        )}
        <div className="alert-action">
          {data.recommended_action || 'System operating within normal parameters.'}
        </div>
      </div>
    </div>
  )
}

// ─── Sensor Grid Card ────────────────────────────────────────────
function SensorGrid({ sensors }) {
  const entries = Object.entries(sensors || {})
  return (
    <div className="card sensors-card">
      <div className="card-header">
        <span className="card-title">Live Sensors</span>
        <span className="card-icon"></span>
      </div>
      <div className="sensor-grid">
        {entries.map(([key, val]) => (
          <div className="sensor-item" key={key}>
            <div className="sensor-label">{key.replace(/_/g, ' ')}</div>
            <div className="sensor-value">
              {typeof val === 'number' ? val.toFixed(1) : val}
              <span className="sensor-unit">{SENSOR_UNITS[key] || ''}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

// ─── Energy Field Heatmap ────────────────────────────────────────
function EnergyFieldHeatmap({ efData }) {
  const matrix = efData?.matrix || Array(6).fill(Array(6).fill(0))

  const getColor = (val) => {
    const v = Math.max(-1, Math.min(1, val))
    if (v >= 0) {
      const r = Math.round(255 * v)
      const g = Math.round(100 * (1 - v))
      const b = Math.round(50 * (1 - v))
      return `rgb(${r}, ${g}, ${b})`
    } else {
      const r = Math.round(252 * (1 + v))
      const g = Math.round(212 * (1 + v))
      const b = Math.round(65 * (1 + v))
      return `rgb(${r}, ${g}, ${b})`
    }
  }

  return (
    <div className="card heatmap-card">
      <div className="card-header">
        <span className="card-title">Energy Field</span>
        <span className="card-icon"></span>
      </div>
      <div className="heatmap-container">
        {matrix.map((row, i) => (
          <div className="heatmap-row" key={i}>
            <div className="heatmap-label">{EF_CHANNELS[i]}</div>
            {(Array.isArray(row) ? row : []).map((val, j) => (
              <div
                className="heatmap-cell"
                key={j}
                style={{ background: getColor(val || 0) }}
              >
                {typeof val === 'number' ? val.toFixed(2) : '0'}
              </div>
            ))}
          </div>
        ))}
        <div className="heatmap-col-labels">
          {EF_CHANNELS.map(ch => (
            <div className="heatmap-col-label" key={ch}>{ch}</div>
          ))}
        </div>
      </div>
      <div className="ef-metrics">
        <div className="ef-metric">
          <div className="ef-metric-value" style={{
            color: (efData?.global_deviation || 0) > 3 ? 'var(--accent-red)' : 'var(--accent-emerald)'
          }}>
            {(efData?.global_deviation || 0).toFixed(2)}
          </div>
          <div className="ef-metric-label">Global Deviation</div>
        </div>
        <div className="ef-metric">
          <div className="ef-metric-value" style={{
            color: (efData?.cosine_similarity || 1) < 0.9 ? 'var(--accent-amber)' : 'var(--accent-cyan)'
          }}>
            {(efData?.cosine_similarity || 1).toFixed(3)}
          </div>
          <div className="ef-metric-label">Cosine Similarity</div>
        </div>
        <div className="ef-metric">
          <div className="ef-metric-value" style={{ color: 'var(--accent-purple)' }}>
            {efData?.most_disrupted_sensor || 'N/A'}
          </div>
          <div className="ef-metric-label">Most Disrupted</div>
        </div>
      </div>
    </div>
  )
}

// ─── Residuals Card ──────────────────────────────────────────────
function ResidualsCard({ residuals }) {
  const entries = Object.entries(residuals || {})

  const getBarColor = (val) => {
    const abs = Math.abs(val)
    if (abs > 20) return 'var(--accent-red)'
    if (abs > 10) return 'var(--accent-yellow)'
    return 'var(--accent-yellow)'
  }

  return (
    <div className="card residuals-card">
      <div className="card-header">
        <span className="card-title">Digital Twin Residuals</span>
        <span className="card-icon"></span>
      </div>
      <div className="residuals-grid">
        {entries.map(([key, val]) => (
          <div className="residual-item" key={key}>
            <div className="residual-name">{key.replace('res_', '').replace(/_/g, ' ')}</div>
            <div className="residual-value" style={{ color: getBarColor(val) }}>
              {typeof val === 'number' ? val.toFixed(2) : val}
            </div>
            <div className="residual-bar">
              <div
                className="residual-bar-fill"
                style={{
                  width: `${Math.min(Math.abs(val || 0) * 2, 100)}%`,
                  background: getBarColor(val || 0),
                }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

// ─── Zone Probabilities ──────────────────────────────────────────
function ZoneProbabilities({ probs }) {
  const probArray = probs || [1, 0, 0, 0, 0, 0]
  const colors = [
    'var(--accent-emerald)', 'var(--accent-yellow)', 'var(--accent-yellow)',
    'var(--accent-yellow)', 'var(--accent-yellow)', 'var(--accent-red)'
  ]

  return (
    <div className="card zone-probs-card">
      <div className="card-header">
        <span className="card-title">Zone Probabilities</span>
        <span className="card-icon"></span>
      </div>
      <div className="zone-bars">
        {ZONE_NAMES.map((name, i) => (
          <div className="zone-bar-item" key={i}>
            <div className="zone-bar-label">{name.split('—')[0].trim()}</div>
            <div className="zone-bar-track">
              <div
                className="zone-bar-fill"
                style={{
                  width: `${(probArray[i] || 0) * 100}%`,
                  background: colors[i],
                }}
              >
                {(probArray[i] || 0) > 0.1 ? `${((probArray[i] || 0) * 100).toFixed(0)}%` : ''}
              </div>
            </div>
            <div className="zone-bar-value">{((probArray[i] || 0) * 100).toFixed(1)}%</div>
          </div>
        ))}
      </div>
    </div>
  )
}

// ─── Engine Diagram ──────────────────────────────────────────────
function EngineDiagram({ zoneIdx, leakDetected, onZoneClick, engineName }) {
  const zones = [
    { id: 1, name: 'Zone 1', desc: 'Intake — Airflow meter → Compressor' },
    { id: 2, name: 'Zone 2', desc: 'Charge Air — Compressor → CAC' },
    { id: 3, name: 'Zone 3', desc: 'CAC/Manifold — CAC → Intake ports' },
    { id: 4, name: 'Zone 4', desc: 'Exhaust — Manifold → Turbine' },
    { id: 5, name: 'Zone 5', desc: 'Aftertreatment — DPF → SCR' },
    { id: 6, name: 'Zone 6', desc: 'Turbine outlet → DPF inlet' },
  ]

  return (
    <div className="card engine-card">
      <div className="card-header">
        <span className="card-title">{engineName || 'Engine'} — Zone Map (Click to Simulate)</span>
        <span className="card-icon"></span>
      </div>
      <div className="engine-diagram">
        {zones.map((zone, idx) => (
          <div key={zone.id} style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <div
              className={`engine-zone ${leakDetected && zoneIdx === zone.id ? 'leak' : 'healthy'}`}
              onClick={() => onZoneClick && onZoneClick(zone.id)}
              style={{ cursor: onZoneClick ? 'pointer' : 'default' }}
            >
              <div className="engine-zone-name">{zone.name}</div>
              <div className="engine-zone-desc">{zone.desc}</div>
            </div>
            {idx < zones.length - 1 && <span className="zone-arrow">→</span>}
          </div>
        ))}
      </div>
    </div>
  )
}

// ─── History Table ───────────────────────────────────────────────
function HistoryTable({ history }) {
  return (
    <div className="card history-card">
      <div className="card-header">
        <span className="card-title">Prediction History</span>
        <span className="card-icon"></span>
      </div>
      <table className="history-table">
        <thead>
          <tr>
            <th>Time</th>
            <th>Status</th>
            <th>Confidence</th>
            <th>Zone</th>
            <th>Severity</th>
            <th>Go/No-Go</th>
          </tr>
        </thead>
        <tbody>
          {(history || []).slice(-15).reverse().map((item, i) => (
            <tr key={i}>
              <td>{item.timestamp ? new Date(item.timestamp).toLocaleTimeString() : '-'}</td>
              <td>
                <span className={`status-dot ${item.leak_detected ? 'leak' : 'healthy'}`}></span>
                {item.leak_detected ? 'LEAK' : 'OK'}
              </td>
              <td>{(item.confidence * 100).toFixed(1)}%</td>
              <td>{item.suspected_zone?.split('—')[0]?.trim() || '-'}</td>
              <td>
                <span className={`alert-severity ${
                  item.severity === 'CRITICAL' ? 'severity-critical' :
                  item.severity === 'MEDIUM' ? 'severity-medium' :
                  item.severity === 'SMALL' ? 'severity-small' : 'severity-none'
                }`}>{item.severity || 'NONE'}</span>
              </td>
              <td style={{ color: item.go_no_go === 'GO' ? 'var(--accent-emerald)' : 'var(--accent-red)', fontWeight: 700 }}>
                {item.go_no_go}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ─── Interactive Level Meters ────────────────────────────────────
function InteractiveLevelMeters({ sensors, setSensors, onSensorChange }) {
  const configs = [
    { key: 'RPM', label: 'Engine Speed', min: 1000, max: 2500, step: 10 },
    { key: 'MAF', label: 'Mass Air Flow', min: 400, max: 1500, step: 5 },
    { key: 'MAP_boost', label: 'Boost Pressure', min: 100, max: 350, step: 2 },
    { key: 'T_cac_out', label: 'Charge Air Temp', min: 20, max: 80, step: 1 },
    { key: 'T_exh_manifold', label: 'Exhaust Temp', min: 300, max: 800, step: 5 },
    { key: 'dP_dpf', label: 'DPF Delta P', min: 0, max: 20, step: 0.1 },
    { key: 'fuel_qty', label: 'Fuel Injection', min: 50, max: 200, step: 1 }
  ];

  const handleChange = (key, val) => {
    const newVal = Number(val);
    const newSensors = { ...sensors, [key]: newVal };
    setSensors(newSensors);
    onSensorChange(newSensors);
  };

  return (
    <div className="card level-meters-card" style={{ gridColumn: 'span 2' }}>
      <div className="card-header">
        <span className="card-title">Live Sensor Overrides</span>
        <span className="card-icon"></span>
      </div>
      <div className="level-meters-container">
        {configs.map(c => {
          const val = sensors[c.key] || c.min;
          const pct = ((val - c.min) / (c.max - c.min)) * 100;
          return (
            <div className="level-meter" key={c.key}>
              <div className="level-meter-label">
                <span>{c.label}</span>
                <span>{val}</span>
              </div>
              <input 
                type="range" 
                className="level-meter-input"
                min={c.min} max={c.max} step={c.step}
                value={val}
                onChange={e => handleChange(c.key, e.target.value)}
                style={{ background: `linear-gradient(to right, var(--accent-yellow) ${pct}%, var(--bg-glass) ${pct}%)` }}
              />
            </div>
          )
        })}
      </div>
    </div>
  )
}

// ─── Energy Field Graph ──────────────────────────────────────────
function EnergyFieldGraph({ efData }) {
  const matrix = efData?.matrix || Array(6).fill(Array(6).fill(0));
  const channels = ['MAF', 'Boost', 'CAC Out', 'Exhaust', 'DPF', 'RPM'];
  
  const values = matrix.map(row => {
    if (Array.isArray(row)) {
      return row.reduce((a, b) => a + Math.abs(b), 0) / (row.length || 1);
    }
    return 0;
  });

  return (
    <div className="card" style={{ gridColumn: 'span 2' }}>
      <div className="card-header">
        <span className="card-title">Energy Field Deviation</span>
        <span className="card-icon"></span>
      </div>
      <div className="ef-graph-container">
        {values.map((v, i) => (
          <div key={i} className="ef-graph-bar" style={{ height: `${Math.min(100, Math.max(5, v * 100))}%` }}>
            <div className="ef-graph-label">{channels[i] || `CH${i+1}`}</div>
          </div>
        ))}
      </div>
    </div>
  )
}

// ─── Acoustic Leak Detector Component ──────────────────────────────
function AcousticLeakDetector({ 
  audioFiles, 
  selectedCategory, 
  setSelectedCategory, 
  selectedFile, 
  setSelectedFile, 
  handleCategoryChange, 
  analyzeSound, 
  acousticResult, 
  isLoading 
}) {
  const categories = [
    { id: 'good', name: 'Healthy Sounds' },
    { id: 'leak', name: 'Leak Sounds' }
  ];

  const files = audioFiles[selectedCategory] || [];
  const audioUrl = selectedFile ? `${API_BASE}/api/audio/play/${selectedCategory}/${selectedFile}` : '';

  return (
    <div className="card acoustic-card">
      <div className="card-header">
        <span className="card-title">Acoustic Leak Detection</span>
        <span className="card-icon">🔊</span>
      </div>
      <div className="acoustic-container" style={{ display: 'flex', flexDirection: 'column', gap: '15px', padding: '10px 0' }}>
        
        <div className="acoustic-control-group" style={{ display: 'flex', gap: '10px' }}>
          <div style={{ flex: 1 }}>
            <label className="sensor-label" style={{ marginBottom: '5px', display: 'block' }}>Engine State</label>
            <select 
              className="demo-select" 
              style={{ width: '100%', padding: '8px', borderRadius: '6px', background: 'var(--bg-glass)', border: '1px solid var(--border-color)', color: 'var(--text-primary)' }}
              value={selectedCategory} 
              onChange={e => handleCategoryChange(e.target.value)}
            >
              {categories.map(cat => (
                <option key={cat.id} value={cat.id}>{cat.name}</option>
              ))}
            </select>
          </div>

          <div style={{ flex: 2 }}>
            <label className="sensor-label" style={{ marginBottom: '5px', display: 'block' }}>Select Audio Track</label>
            <select 
              className="demo-select" 
              style={{ width: '100%', padding: '8px', borderRadius: '6px', background: 'var(--bg-glass)', border: '1px solid var(--border-color)', color: 'var(--text-primary)' }}
              value={selectedFile} 
              onChange={e => setSelectedFile(e.target.value)}
            >
              {files.map(f => (
                <option key={f} value={f}>{f.replace('engine_', '').replace('.wav', '')}</option>
              ))}
            </select>
          </div>
        </div>

        {audioUrl && (
          <div className="audio-player-wrapper" style={{ display: 'flex', flexDirection: 'column', gap: '8px', background: 'rgba(255, 255, 255, 0.03)', padding: '12px', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
            <span className="sensor-label" style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Audio Player</span>
            <audio key={audioUrl} controls style={{ width: '100%', height: '32px' }}>
              <source src={audioUrl} type="audio/wav" />
              Your browser does not support the audio element.
            </audio>
          </div>
        )}

        <button 
          className="demo-btn" 
          style={{ width: '100%', background: 'linear-gradient(135deg, var(--accent-yellow) 0%, #ff9800 100%)', color: '#000', fontWeight: 'bold', border: 'none', padding: '10px', borderRadius: '6px', cursor: 'pointer' }}
          onClick={analyzeSound}
          disabled={isLoading || !selectedFile}
        >
          {isLoading ? 'Analyzing Audio Frequencies...' : '🔍 Analyze Audio Leak Signatures'}
        </button>

        {acousticResult && (
          <div className="acoustic-result-wrapper" style={{ animation: 'fadeIn 0.3s ease-in-out', display: 'flex', flexDirection: 'column', gap: '12px', marginTop: '5px' }}>
            <div style={{ 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'space-between', 
              padding: '12px', 
              borderRadius: '8px',
              border: '1px solid',
              background: acousticResult.prediction === 'LEAK' ? 'rgba(239, 83, 80, 0.1)' : 'rgba(76, 175, 80, 0.1)',
              borderColor: acousticResult.prediction === 'LEAK' ? 'var(--accent-red)' : 'var(--accent-emerald)'
            }}>
              <div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Acoustic Classification</div>
                <div style={{ 
                  fontSize: '1.4rem', 
                  fontWeight: '800', 
                  color: acousticResult.prediction === 'LEAK' ? 'var(--accent-red)' : 'var(--accent-emerald)' 
                }}>
                  {acousticResult.prediction}
                </div>
              </div>
              <div style={{ textAlign: 'right' }}>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Probability</div>
                <div style={{ 
                  fontSize: '1.4rem', 
                  fontWeight: '800',
                  color: acousticResult.prediction === 'LEAK' ? 'var(--accent-red)' : 'var(--accent-emerald)'
                }}>
                  {Math.round((acousticResult.prediction === 'LEAK' ? acousticResult.leak_probability : acousticResult.healthy_probability) * 100)}%
                </div>
              </div>
            </div>

            <div className="acoustic-features" style={{ fontSize: '0.75rem', background: 'rgba(0,0,0,0.15)', padding: '10px', borderRadius: '6px', border: '1px solid var(--border-color)', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px 15px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-secondary)' }}>RMS Energy:</span>
                <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 'bold' }}>{acousticResult.features.rms.toFixed(4)}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-secondary)' }}>Peak Amp:</span>
                <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 'bold' }}>{acousticResult.features.peak.toFixed(4)}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-secondary)' }}>Spectral Centroid:</span>
                <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 'bold', color: 'var(--accent-yellow)' }}>{acousticResult.features.spectral_centroid.toFixed(0)} Hz</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-secondary)' }}>Zero-Crossing Rate:</span>
                <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 'bold' }}>{acousticResult.features.zero_crossings.toFixed(4)}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', gridColumn: 'span 2', borderTop: '1px solid rgba(255,255,255,0.05)', paddingTop: '6px' }}>
                <span style={{ color: 'var(--text-secondary)' }}>High Frequency (4kHz+) Energy Ratio:</span>
                <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 'bold', color: acousticResult.prediction === 'LEAK' ? 'var(--accent-red)' : 'var(--text-primary)' }}>{(acousticResult.features.band_4 * 1000).toFixed(2)}‰</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// ─── Main App ────────────────────────────────────────────────────
function App() {
  const [showIntro, setShowIntro] = useState(true)
  const [activePage, setActivePage] = useState('dashboard')
  const [predictionData, setPredictionData] = useState(null)
  const [history, setHistory] = useState([])
  const [isConnected, setIsConnected] = useState(false)
  const [isDemoRunning, setIsDemoRunning] = useState(false)
  const [simZone, setSimZone] = useState(0)
  const [simSeverity, setSimSeverity] = useState(0)
  const [manualSensors, setManualSensors] = useState({
    RPM: 1800, MAF: 850, MAP_boost: 215, T_cac_out: 45, T_exh_manifold: 550, dP_dpf: 0.5, fuel_qty: 120
  })
  const wsRef = useRef(null)
  const demoWsRef = useRef(null)
  const debounceRef = useRef(null)

  // Engine selection
  const [engines, setEngines] = useState([])
  const [currentEngine, setCurrentEngine] = useState(null)
  const [showEngineSelector, setShowEngineSelector] = useState(false)
  const [isSwitchingEngine, setIsSwitchingEngine] = useState(false)
  const [engineSwitchError, setEngineSwitchError] = useState(null)

  // Always ask the user to pick an engine on first load — the backend may
  // already have one "active" from a previous session/default, but that's
  // not the same as this user having chosen one for this session.
  const fetchEngines = useCallback(() => {
    fetch(`${API_BASE}/api/engines`)
      .then(res => res.json())
      .then(data => setEngines(data.engines || []))
      .catch(err => console.error('Error fetching engines:', err))
  }, [])

  useEffect(() => {
    fetchEngines()
    setShowEngineSelector(true)
  }, [])

  const selectEngine = useCallback(async (engineId) => {
    setIsSwitchingEngine(true)
    setEngineSwitchError(null)
    try {
      const res = await fetch(`${API_BASE}/api/engine/select`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ engine: engineId }),
      })
      const data = await res.json()
      if (!res.ok) {
        throw new Error(data.detail || 'Failed to switch engine')
      }
      // Stop any running demo — it's tied to the previous engine's stream
      if (demoWsRef.current) {
        demoWsRef.current.close()
        demoWsRef.current = null
      }
      setIsDemoRunning(false)
      setIsConnected(false)
      // Reset session state — confidence/history from the old engine isn't
      // meaningful mixed with the new one
      setPredictionData(null)
      setHistory([])
      setCurrentEngine({ id: engineId, name: data.engine, n_zones: data.n_zones, calibrated: data.calibrated, trained: true, active: true })
      setShowEngineSelector(false)
      fetchEngines()
    } catch (err) {
      console.error('Engine switch error:', err)
      setEngineSwitchError(err.message || 'Failed to switch engine')
    } finally {
      setIsSwitchingEngine(false)
    }
  }, [fetchEngines])

  // Acoustic states
  const [audioFiles, setAudioFiles] = useState({ good: [], leak: [] })
  const [selectedCategory, setSelectedCategory] = useState('good')
  const [selectedFile, setSelectedFile] = useState('')
  const [acousticResult, setAcousticResult] = useState(null)
  const [isAcousticLoading, setIsAcousticLoading] = useState(false)

  const handleCategoryChange = (cat) => {
    setSelectedCategory(cat);
    const files = audioFiles[cat] || [];
    if (files.length > 0) {
      setSelectedFile(files[0]);
    } else {
      setSelectedFile('');
    }
    setAcousticResult(null);
  };

  const analyzeSound = async () => {
    if (!selectedFile) return;
    setIsAcousticLoading(true);
    setAcousticResult(null);
    try {
      const res = await fetch(`${API_BASE}/api/audio/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          category: selectedCategory,
          filename: selectedFile,
        }),
      });
      const result = await res.json();
      setAcousticResult(result);

      // Make it affect the global prediction data / status!
      const isLeak = result.prediction === 'LEAK';
      const confidence = isLeak ? result.leak_probability : result.healthy_probability;
      
      const newPredictionData = {
        leak_detected: isLeak,
        confidence: confidence,
        suspected_zone: isLeak ? "Acoustic Leak Detected" : "No Leak / Healthy",
        suspected_zone_idx: isLeak ? 2 : 0, // default to Zone 2 or 0
        zone_probabilities: isLeak ? [0.1, 0.1, 0.8, 0.0, 0.0, 0.0] : [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        severity: isLeak ? (confidence > 0.85 ? "CRITICAL" : (confidence > 0.7 ? "MEDIUM" : "SMALL")) : "NONE",
        flow_loss_pct: isLeak ? (confidence > 0.85 ? 15.0 : (confidence > 0.7 ? 8.0 : 2.0)) : 0.0,
        go_no_go: isLeak ? "NO-GO" : "GO",
        recommended_action: isLeak 
          ? `Acoustic Analysis: High-frequency leak noise signatures detected in engine sound. Inspect turbocharger, boost pipes, and manifolds immediately.`
          : "Acoustic Analysis: Engine sound pattern is normal. System healthy.",
        residuals: {
          res_MAF: isLeak ? -50.0 : 0.0,
          res_MAP_boost: isLeak ? -15.0 : 0.0,
          res_T_boost: isLeak ? 10.0 : 0.0,
          res_T_cac_out: 0.0,
          res_MAP_cac_out: 0.0,
          res_T_exh_manifold: 0.0,
          res_T_post_turbine: 0.0,
          res_dP_dpf: 0.0,
          res_MAP_intake: 0.0
        },
        sensors: {
          ...manualSensors,
          RPM: manualSensors.RPM,
        },
        energy_field: {
          matrix: isLeak ? Array(6).fill(Array(6).fill(0.5)) : Array(6).fill(Array(6).fill(0.0)),
          global_deviation: isLeak ? 4.5 : 0.5,
          cosine_similarity: isLeak ? 0.85 : 0.98,
          most_disrupted_sensor: isLeak ? "MAF" : "N/A",
          ef_suspected_zone: isLeak ? "Zone 2" : "N/A"
        }
      };

      setPredictionData(newPredictionData);
      setHistory(prev => [...prev.slice(-100), newPredictionData]);

    } catch (err) {
      console.error('Error analyzing sound:', err);
    } finally {
      setIsAcousticLoading(false);
    }
  };

  // Fetch audio files list on mount
  useEffect(() => {
    fetch(`${API_BASE}/api/audio/files`)
      .then(res => res.json())
      .then(data => {
        if (data && (data.good || data.leak)) {
          setAudioFiles(data)
          if (data.good && data.good.length > 0) {
            setSelectedFile(data.good[0])
          }
        }
      })
      .catch(err => console.error('Error fetching audio files:', err))
  }, [])

  const handleManualSensorChange = (newSensors) => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(async () => {
      try {
        const res = await fetch(`${API_BASE}/api/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(newSensors),
        })
        const data = await res.json()
        setPredictionData(data)
        setHistory(prev => [...prev.slice(-100), data])
      } catch (err) {
        console.error('Manual prediction error:', err)
      }
    }, 100);
  };

  const handleZoneClick = async (zoneId) => {
    // We use the simulation API to ensure the sensor values perfectly match 
    // the physics patterns the ML model was trained on for this specific zone.
    try {
      const res = await fetch(`${API_BASE}/api/simulate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          rpm: 1800,
          leak_zone: zoneId,
          leak_severity: 2, // Medium leak (8% flow loss) for stable detection
        }),
      })
      const data = await res.json()
      
      // Update UI with the simulation result
      setPredictionData(data)
      if (data.sensors) {
        setManualSensors(data.sensors)
      }
      setHistory(prev => [...prev.slice(-100), data])
      
      console.log(`Simulated leak in Zone ${zoneId}:`, data.suspected_zone);
    } catch (err) {
      console.error('Zone click simulation error:', err)
    }
  };

  // Fetch history on load
  useEffect(() => {
    fetch(`${API_BASE}/api/history?limit=50`)
      .then(res => res.json())
      .then(data => {
        if (Array.isArray(data)) setHistory(data)
      })
      .catch(() => {})
  }, [])

  // Demo WebSocket
  const startDemo = useCallback(() => {
    if (demoWsRef.current) {
      demoWsRef.current.close()
    }
    const ws = new WebSocket(`${WS_BASE}/ws/demo`)
    ws.onopen = () => {
      setIsConnected(true)
      setIsDemoRunning(true)
    }
    ws.onmessage = (e) => {
      const data = JSON.parse(e.data)
      setPredictionData(data)
      setHistory(prev => [...prev.slice(-100), data])
      
      if (data.sensors) {
        setManualSensors({
          RPM: data.sensors.RPM || 1800,
          MAF: data.sensors.MAF || 850,
          MAP_boost: data.sensors.MAP_boost || 215,
          T_cac_out: data.sensors.T_cac_out || 45,
          T_exh_manifold: data.sensors.T_exh_manifold || 550,
          dP_dpf: data.sensors.dP_dpf || 0.5,
          fuel_qty: data.sensors.fuel_qty || 120
        })
      }
    }
    ws.onclose = () => {
      setIsConnected(false)
      setIsDemoRunning(false)
    }
    ws.onerror = () => {
      setIsConnected(false)
      setIsDemoRunning(false)
    }
    demoWsRef.current = ws
  }, [])

  const stopDemo = useCallback(() => {
    if (demoWsRef.current) {
      demoWsRef.current.close()
      demoWsRef.current = null
    }
    setIsDemoRunning(false)
    setIsConnected(false)
  }, [])

  // Manual simulation
  const runSimulation = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/simulate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          rpm: 1800,
          leak_zone: simZone,
          leak_severity: simSeverity,
        }),
      })
      const data = await res.json()
      setPredictionData(data)
      setHistory(prev => [...prev.slice(-100), data])
    } catch (err) {
      console.error('Simulation error:', err)
    }
  }, [simZone, simSeverity])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (demoWsRef.current) demoWsRef.current.close()
      if (wsRef.current) wsRef.current.close()
    }
  }, [])

  const data = predictionData || {
    leak_detected: false,
    confidence: 0,
    suspected_zone: ZONE_NAMES[0],
    suspected_zone_idx: 0,
    zone_probabilities: [1, 0, 0, 0, 0, 0],
    severity: 'NONE',
    flow_loss_pct: 0,
    go_no_go: 'GO',
    recommended_action: 'Connect to backend to start monitoring.',
    residuals: {},
    energy_field: { matrix: Array(6).fill(Array(6).fill(0)), global_deviation: 0, cosine_similarity: 1, most_disrupted_sensor: 'N/A' },
    sensors: {},
  }

  return (
    <>
      {showIntro && <IntroVideo onComplete={() => setShowIntro(false)} />}
      {!showIntro && showEngineSelector && (
        <EngineSelector
          engines={engines}
          onSelect={selectEngine}
          isLoading={isSwitchingEngine}
          error={engineSwitchError}
        />
      )}
      <div className="app-container">
      <Sidebar
        activePage={activePage}
        setActivePage={setActivePage}
        isConnected={isConnected}
        engineName={currentEngine?.name}
        onSwitchEngine={() => { setEngineSwitchError(null); fetchEngines(); setShowEngineSelector(true) }}
      />

      <main className="main-content">
        <div className="main-header">
          <h2>
            {activePage === 'dashboard' && 'Real-Time Dashboard'}
            {activePage === 'engine' && 'Analysis'}
            {activePage === 'history' && 'Detection History'}
            {activePage === 'dax' && 'DAX Engine Query Console'}
            {activePage === 'digital-twin' && 'Digital Twin'}
          </h2>
          <div className="header-controls">
            <span className={`header-badge ${isConnected ? 'badge-connected' : 'badge-disconnected'}`}>
              <span className="status-dot" style={{
                background: isConnected ? 'var(--accent-emerald)' : 'var(--accent-red)',
                width: '6px', height: '6px'
              }}></span>
              {isConnected ? 'LIVE' : 'OFFLINE'}
            </span>
          </div>
        </div>

        {/* Demo Controls — always visible */}
        <div className="dashboard-grid" style={{ marginBottom: '20px' }}>
          <div className="demo-controls">
            <button
              id="demo-toggle"
              className={`demo-btn ${isDemoRunning ? 'active' : ''}`}
              onClick={isDemoRunning ? stopDemo : startDemo}
            >
              {isDemoRunning ? 'Stop Demo' : 'Start Live Demo'}
            </button>

            <select className="demo-select" value={simZone} onChange={e => setSimZone(Number(e.target.value))}>
              <option value={0}>Healthy (No Leak)</option>
              <option value={1}>Zone 1 — Intake</option>
              <option value={2}>Zone 2 — Charge Air</option>
              <option value={3}>Zone 3 — CAC/Manifold</option>
              <option value={4}>Zone 4 — Exhaust</option>
              <option value={5}>Zone 5 — DPF/SCR</option>
              <option value={6}>Zone 6 — Turbine Outlet/Pre-DPF</option>
            </select>

            <select className="demo-select" value={simSeverity} onChange={e => setSimSeverity(Number(e.target.value))}>
              <option value={0}>No Fault</option>
              <option value={1}>Small (2%)</option>
              <option value={2}>Medium (8%)</option>
              <option value={3}>Large (15%)</option>
            </select>

            <button id="simulate-btn" className="demo-btn danger" onClick={runSimulation}>
              Inject & Predict
            </button>
          </div>
        </div>

        {/* Dashboard Page */}
        {activePage === 'dashboard' && (
          <div className="dashboard-grid">
            <GoNoGoIndicator status={data.go_no_go} />
            <ConfidenceRing confidence={data.confidence} leakDetected={data.leak_detected} />
            <LeakAlertCard data={data} />

            <SensorGrid sensors={data.sensors} />
            <EnergyFieldHeatmap efData={data.energy_field} />

            <ResidualsCard residuals={data.residuals} />
            <ZoneProbabilities probs={data.zone_probabilities} />
          </div>
        )}

        {/* Engine Diagram Page */}
        {activePage === 'engine' && (
          <div className="dashboard-grid">
            <InteractiveLevelMeters 
              sensors={manualSensors} 
              setSensors={setManualSensors} 
              onSensorChange={handleManualSensorChange} 
            />
            <AcousticLeakDetector
              audioFiles={audioFiles}
              selectedCategory={selectedCategory}
              setSelectedCategory={setSelectedCategory}
              selectedFile={selectedFile}
              setSelectedFile={setSelectedFile}
              handleCategoryChange={handleCategoryChange}
              analyzeSound={analyzeSound}
              acousticResult={acousticResult}
              isLoading={isAcousticLoading}
            />
            <EngineDiagram
              zoneIdx={data.suspected_zone_idx}
              leakDetected={data.leak_detected}
              onZoneClick={handleZoneClick}
              engineName={currentEngine?.name}
            />
            <GoNoGoIndicator status={data.go_no_go} />
            <ConfidenceRing confidence={data.confidence} leakDetected={data.leak_detected} />
            <EnergyFieldGraph efData={data.energy_field} />
            <LeakAlertCard data={data} />
            <ZoneProbabilities probs={data.zone_probabilities} />
          </div>
        )}

        {/* History Page */}
        {activePage === 'history' && (
          <div className="dashboard-grid">
            <HistoryTable history={history} />
          </div>
        )}

        {/* DAX Console Page */}
        {activePage === 'dax' && (
          <div className="dashboard-grid">
            <DAXConsole />
          </div>
        )}

        {/* Digital Twin Page */}
        {activePage === 'digital-twin' && (
          <div className="dashboard-grid" style={{ height: '80vh', padding: '0', gridTemplateColumns: '1fr' }}>
            <iframe 
              src="/EngineIQ_v5_Fixed.html" 
              style={{ width: '100%', height: '100%', minHeight: '800px', border: 'none', borderRadius: '12px' }}
              title="Digital Twin"
            />
          </div>
        )}
      </main>
      <ChatBot />
    </div>
    </>
  )
}

export default App
