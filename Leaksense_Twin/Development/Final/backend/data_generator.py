"""
LeakSense Twin — Synthetic Data Generator
Generates training samples for the active engine profile (see
engine_profiles/): 40,000 healthy baseline samples + 3,000 samples per
fault zone × severity combination.

Engine-specific physics constants come from an EngineProfile instead of
being hardcoded, so the same generator works for any calibrated engine.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import NOISE_CONFIG
from engine_profiles import active_profile


def fuel_qty_from_torque_curve(rpm: float, profile) -> float:
    """
    Map rated torque (interpolated from the engine's real published curve) to
    a fuel-quantity range, preserving the convention used throughout the rest
    of the pipeline. Torque — not power — is the right proxy for per-cycle
    fuel quantity at a given rpm.
    """
    torque = np.interp(rpm, profile.torque_curve_rpm, profile.torque_curve_nm)
    t_min, t_max = min(profile.torque_curve_nm), max(profile.torque_curve_nm)
    frac = (torque - t_min) / (t_max - t_min)
    return profile.fuel_qty_min + (profile.fuel_qty_max - profile.fuel_qty_min) * frac


def ve_model(rpm: float, profile) -> float:
    """Volumetric efficiency as a function of RPM — peaks at the profile's rated-torque RPM."""
    p = profile.intake
    ve = p.ve_base + p.ve_quadratic_coeff * (rpm - p.ve_rpm_peak) ** 2
    return np.clip(ve, p.ve_clip_min, p.ve_clip_max)


def generate_healthy_sample(rpm: float, ambient_variation: float = 0.0, profile=None) -> dict:
    """Generate a single healthy-state sensor reading at a given RPM, using the active engine profile."""
    profile = profile or active_profile()
    p = profile

    # Ambient conditions with slight variation
    T_ambient_K = p.t_ambient_k + ambient_variation * 5.0  # ±5K variation
    P_ambient = p.map_ambient + ambient_variation * 2.0     # ±2 kPa variation

    # Fuel quantity scales with the engine's real published torque curve
    fuel_qty = fuel_qty_from_torque_curve(rpm, p)
    fuel_qty = np.clip(fuel_qty, p.fuel_qty_min, p.fuel_qty_max)

    # --- Intake Zone ---
    ve = ve_model(rpm, p)
    rho_air = P_ambient * 1000.0 / (p.r_air * T_ambient_K)
    vol_per_cyl = (p.intake.displacement_L / 1000.0) / p.intake.n_cylinders  # m³
    n_strokes = rpm / 2.0 / 60.0  # 4-stroke
    MAF = ve * vol_per_cyl * p.intake.n_cylinders * n_strokes * rho_air * 3600.0  # kg/h

    # --- Charge Air System ---
    ca = p.charge_air
    corr_flow = MAF * np.sqrt(T_ambient_K / 288.15) / (P_ambient / 101.325)
    corr_speed = rpm / np.sqrt(T_ambient_K / 288.15)

    c = ca.compressor_coeffs
    PR = (c['c00'] + c['c10'] * corr_flow + c['c01'] * corr_speed +
          c['c20'] * corr_flow**2 + c['c11'] * corr_flow * corr_speed + c['c02'] * corr_speed**2)
    PR = np.clip(PR, ca.pr_clip_min, ca.pr_clip_max)

    MAP_boost = PR * P_ambient
    T_boost_K = T_ambient_K * (1.0 + (PR**((p.gamma-1)/p.gamma) - 1.0) / ca.turbo_eff_compressor)
    T_cac_out_K = T_boost_K - ca.cac_effectiveness * (T_boost_K - T_ambient_K)

    # CAC pressure drop
    MAP_cac_in = MAP_boost - ca.cac_duct_loss_kpa
    MAP_cac_out = MAP_boost - ca.cac_dp_coeff * (MAF / 100.0)**2
    MAP_intake = MAP_cac_out - ca.manifold_runner_loss_kpa

    # --- Exhaust Zone ---
    e = p.exhaust
    fuel_mass_flow = fuel_qty * 1e-6 * n_strokes * e.n_cylinders  # kg/s
    air_mass_flow = MAF / 3600.0
    total_flow = air_mass_flow + fuel_mass_flow

    if total_flow > 0:
        delta_T = (fuel_mass_flow * e.lhv_diesel * 1000.0 * e.combustion_eff) / (total_flow * p.cp_air)
        T_exh_manifold = T_cac_out_K + delta_T
    else:
        T_exh_manifold = T_cac_out_K + 400.0

    T_exh_manifold = np.clip(T_exh_manifold, e.t_exh_clip_min_k, e.t_exh_clip_max_k)

    # Turbine
    PR_turbine = e.turbine_pr_const + e.turbine_pr_rpm_coeff * (rpm / e.turbine_rpm_norm) + \
        e.turbine_pr_maf_coeff * (MAF / e.turbine_maf_norm)
    PR_turbine = np.clip(PR_turbine, e.turbine_pr_clip_min, e.turbine_pr_clip_max)
    T_post_turbine = T_exh_manifold * (1.0 - e.turbo_eff_turbine * (1.0 - PR_turbine**(-(p.gamma-1)/p.gamma)))

    # DPF
    dP_dpf = e.dpf_dp_coeff * (air_mass_flow ** e.dpf_dp_exponent)
    T_dpf_in = T_post_turbine - e.dpf_in_loss_k
    T_dpf_out = T_dpf_in - e.dpf_out_loss_k

    return {
        'RPM': rpm,
        'MAF': MAF,
        'MAP_intake': MAP_intake,
        'MAP_boost': MAP_boost,
        'MAP_cac_in': MAP_cac_in,
        'MAP_cac_out': MAP_cac_out,
        'T_intake': T_ambient_K - 273.15,      # °C
        'T_boost': T_boost_K - 273.15,          # °C
        'T_cac_out': T_cac_out_K - 273.15,      # °C
        'T_exh_manifold': T_exh_manifold - 273.15,  # °C
        'T_dpf_in': T_dpf_in - 273.15,
        'T_dpf_out': T_dpf_out - 273.15,
        'fuel_qty': fuel_qty,
        'T_post_turbine': T_post_turbine - 273.15,
        'dP_dpf': dP_dpf,
    }


def inject_leak(sample: dict, zone: int, severity: int) -> dict:
    """
    Inject a leak into a healthy sample. Leak fractions are generic
    percentages of flow loss, not engine-specific — only the healthy
    baseline they're applied to comes from the engine profile.

    Args:
        sample: healthy sensor reading dict
        zone: 1-6 (leak zone)
        severity: 1=small(2%), 2=medium(8%), 3=large(15%)

    Returns:
        modified sample with leak effects
    """
    s = sample.copy()
    severity_map = {1: 0.02, 2: 0.08, 3: 0.15}
    leak_fraction = severity_map[severity]

    if zone == 1:
        # Zone 1: Airflow meter → Compressor inlet
        s['MAF'] *= (1.0 - leak_fraction)
        s['MAP_boost'] *= (1.0 - leak_fraction * 0.7)
        s['T_boost'] += leak_fraction * 5.0  # slight temp rise

    elif zone == 2:
        # Zone 2: Post-compressor → CAC inlet
        s['MAP_boost'] *= (1.0 - leak_fraction * 0.8)
        s['MAP_cac_in'] *= (1.0 - leak_fraction * 0.9)
        s['T_intake'] += leak_fraction * 8.0  # less cooling

    elif zone == 3:
        # Zone 3: CAC → Intake manifold
        s['MAP_cac_out'] *= (1.0 - leak_fraction * 0.85)
        s['MAP_intake'] *= (1.0 - leak_fraction * 0.9)
        s['T_cac_out'] += leak_fraction * 6.0

    elif zone == 4:
        # Zone 4: Exhaust manifold → Turbo turbine
        s['T_exh_manifold'] *= (1.0 - leak_fraction * 0.3)
        s['MAP_boost'] *= (1.0 - leak_fraction * 0.5)  # cascading boost drop
        s['T_dpf_in'] *= (1.0 - leak_fraction * 0.2)

    elif zone == 5:
        # Zone 5: DPF/SCR area (internal — e.g. cracked canister/substrate)
        s['dP_dpf'] *= (1.0 - leak_fraction * 0.6)
        s['T_dpf_out'] *= (1.0 - leak_fraction * 0.15)
        s['T_dpf_in'] *= (1.0 - leak_fraction * 0.1)

    elif zone == 6:
        # Zone 6: Turbine outlet -> DPF inlet (duct/joint leak ahead of the
        # aftertreatment, distinct from an internal DPF/SCR leak in zone 5 —
        # ambient dilution happens before the T_dpf_in sensor, not after it,
        # so T_dpf_in and the post-turbine reading take the largest hit here.
        s['T_post_turbine'] *= (1.0 - leak_fraction * 0.4)
        s['T_dpf_in'] *= (1.0 - leak_fraction * 0.5)
        s['dP_dpf'] *= (1.0 - leak_fraction * 0.4)

    return s


def add_noise(sample: dict) -> dict:
    """Add realistic Gaussian sensor noise to a sample."""
    s = sample.copy()
    nc = NOISE_CONFIG

    # Pressure sensors: 0.5% noise
    for key in ['MAP_intake', 'MAP_boost', 'MAP_cac_in', 'MAP_cac_out', 'dP_dpf']:
        if key in s:
            s[key] += np.random.normal(0, abs(s[key]) * nc['pressure_sigma_pct'])

    # Temperature sensors: 1% noise
    for key in ['T_intake', 'T_boost', 'T_cac_out', 'T_exh_manifold', 'T_dpf_in', 'T_dpf_out', 'T_post_turbine']:
        if key in s:
            s[key] += np.random.normal(0, max(abs(s[key]) * nc['temperature_sigma_pct'], 0.5))

    # Flow sensors: 1.5% noise
    if 'MAF' in s:
        s['MAF'] += np.random.normal(0, abs(s['MAF']) * nc['flow_sigma_pct'])

    # RPM noise (small)
    s['RPM'] += np.random.normal(0, 3.0)

    # Fuel qty noise
    s['fuel_qty'] += np.random.normal(0, abs(s['fuel_qty']) * 0.005)

    return s


def generate_dataset(n_healthy: int = 40000, n_fault_per_combo: int = 667,
                     seed: int = 42, profile=None) -> pd.DataFrame:
    """
    Generate the complete training dataset for the active (or given) engine profile.

    Args:
        n_healthy: number of healthy samples
        n_fault_per_combo: samples per zone×severity combination
        seed: random seed for reproducibility
        profile: EngineProfile to generate for (defaults to active_profile())

    Returns:
        DataFrame with all samples, labels, and metadata
    """
    profile = profile or active_profile()
    np.random.seed(seed)
    records = []

    rpm_range = np.linspace(profile.operating_range.rpm_min, profile.operating_range.rpm_max,
                            profile.operating_range.rpm_steps)
    rpm_min, rpm_max = profile.operating_range.rpm_min, profile.operating_range.rpm_max
    n_zones = profile.zones.n_zones

    print(f"Generating healthy samples for {profile.name}...")
    for i in range(n_healthy):
        rpm = np.random.choice(rpm_range) + np.random.uniform(-25, 25)
        rpm = np.clip(rpm, rpm_min, rpm_max)
        ambient_var = np.random.uniform(-1.0, 1.0)

        sample = generate_healthy_sample(rpm, ambient_var, profile)
        sample = add_noise(sample)
        sample['leak_zone'] = 0
        sample['leak_severity'] = 0
        sample['is_steady_state'] = 1
        records.append(sample)

    print("Generating fault samples...")
    for zone in range(1, n_zones + 1):
        for severity in range(1, 4):
            for i in range(n_fault_per_combo):
                rpm = np.random.choice(rpm_range) + np.random.uniform(-25, 25)
                rpm = np.clip(rpm, rpm_min, rpm_max)
                ambient_var = np.random.uniform(-1.0, 1.0)

                sample = generate_healthy_sample(rpm, ambient_var, profile)
                sample = inject_leak(sample, zone, severity)
                sample = add_noise(sample)
                sample['leak_zone'] = zone
                sample['leak_severity'] = severity
                sample['is_steady_state'] = 1
                records.append(sample)

    df = pd.DataFrame(records)
    df['timestamp'] = pd.date_range(start='2026-01-01', periods=len(df), freq='1s')
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    print(f"Dataset generated: {len(df)} samples")
    print(f"  Healthy: {(df['leak_zone'] == 0).sum()}")
    print(f"  Fault:   {(df['leak_zone'] > 0).sum()}")
    for z in range(1, n_zones + 1):
        print(f"    Zone {z}: {(df['leak_zone'] == z).sum()}")

    return df


if __name__ == "__main__":
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)

    profile = active_profile()
    df = generate_dataset(profile=profile)
    out_path = output_dir / f"training_data_{profile.artifact_dir}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved dataset to {out_path}")
