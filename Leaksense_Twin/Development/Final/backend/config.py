"""
LeakSense Twin — Engine-Agnostic Configuration
Engine-specific physics constants and zone topology now live in
engine_profiles/*.json (see engine_profiles/base.py for the schema).
This file holds only what's genuinely independent of which engine is
active: the 31-feature schema, ML hyperparameters, noise model, and the
steady-state filter.
"""

# ─── Sensor Channels (engine-agnostic feature schema) ─────────────────────────
RAW_SENSOR_COLS = [
    'RPM', 'MAF', 'MAP_intake', 'MAP_boost', 'MAP_cac_in', 'MAP_cac_out',
    'T_intake', 'T_boost', 'T_cac_out', 'T_exh_manifold', 'T_dpf_in',
    'T_dpf_out', 'fuel_qty'
]

RESIDUAL_COLS = [
    'res_MAF', 'res_MAP_boost', 'res_T_boost', 'res_T_cac_out',
    'res_MAP_cac_out', 'res_T_exh_manifold', 'res_T_post_turbine',
    'res_dP_dpf', 'res_MAP_intake'
]

EF_CHANNELS = ['MAF', 'MAP_boost', 'MAP_cac_out', 'T_cac_out',
               'T_exh_manifold', 'dP_dpf', 'RPM']

# ─── ML Model Hyperparameters ─────────────────────────────────────────────────
ML_CONFIG = {
    "input_dim": 31,
    "dropout": 0.3,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 256,
    "max_epochs": 200,
    "patience": 15,
    "leak_threshold": 0.5,
    "consecutive_alerts": 3,
}

# ─── Noise Parameters ─────────────────────────────────────────────────────────
NOISE_CONFIG = {
    "pressure_sigma_pct": 0.005,   # 0.5% for pressure sensors
    "temperature_sigma_pct": 0.01, # 1% for temperature sensors
    "flow_sigma_pct": 0.015,       # 1.5% for flow sensors
    "drift_rate_pct_hr": 0.001,    # 0.1%/hour sensor drift
}

# ─── Steady-State Filter ──────────────────────────────────────────────────────
STEADY_STATE_CONFIG = {
    "window_size": 30,        # 30 seconds at 1 Hz
    "rpm_std_max": 10.0,      # RPM
    "maf_std_pct_max": 0.02,  # 2% of mean
    "fuel_std_pct_max": 0.01, # 1% of mean
}
