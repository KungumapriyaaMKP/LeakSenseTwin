"""
LeakSense Twin — Zone 1: Intake Digital Twin
Physics: Mass continuity — predicts MAF from RPM + ambient conditions.
If actual MAF < predicted MAF by threshold: Zone 1 leak detected.

Engine-specific coefficients come from an EngineProfile (see
engine_profiles/) instead of being hardcoded, so the same physics model
works for any engine with a calibrated profile.
"""

import numpy as np
import joblib


class IntakeTwinModel:
    """
    Physics-based Intake Zone Digital Twin.
    Predicts expected Mass Air Flow (MAF) using a volumetric efficiency model.
    """

    def __init__(self, profile):
        self.profile = profile
        self.correction_factor = 1.0  # Online learning correction

    def ve_model(self, rpm: float) -> float:
        """Volumetric efficiency as a function of RPM (peaks near rated torque RPM)."""
        p = self.profile.intake
        ve = p.ve_base + p.ve_quadratic_coeff * (rpm - p.ve_rpm_peak) ** 2
        return np.clip(ve, p.ve_clip_min, p.ve_clip_max)

    def predict_maf(self, rpm: float, map_ambient_kpa: float = None,
                    t_ambient_k: float = None, fuel_qty_mg: float = 120.0) -> float:
        """
        Predict expected Mass Air Flow (MAF) in kg/h.

        Physics: MAF = VE × V_cyl × N_cyl × (RPM/2/60) × ρ_air × 3600
        """
        p = self.profile
        map_ambient_kpa = p.map_ambient if map_ambient_kpa is None else map_ambient_kpa
        t_ambient_k = p.t_ambient_k if t_ambient_k is None else t_ambient_k

        ve = self.ve_model(rpm)
        p_ambient_pa = map_ambient_kpa * 1000.0
        rho_air = p_ambient_pa / (p.r_air * t_ambient_k)
        vol_per_cyl = (p.intake.displacement_L / 1000.0) / p.intake.n_cylinders
        n_strokes_per_sec = rpm / 2.0 / 60.0
        maf_kgs = ve * vol_per_cyl * p.intake.n_cylinders * n_strokes_per_sec * rho_air
        maf_kgh = maf_kgs * 3600.0
        return maf_kgh * self.correction_factor

    def predict(self, rpm, map_ambient=None, t_ambient_k=None, fuel_qty=120.0):
        """Convenience method returning predicted MAF."""
        return self.predict_maf(rpm, map_ambient, t_ambient_k, fuel_qty)

    def update_online(self, actual_maf: float, rpm: float,
                      map_ambient: float = None,
                      t_ambient_k: float = None,
                      alpha: float = 0.01):
        """
        Online learning via Exponentially Weighted Average.
        Slowly adapts the correction factor to handle engine aging.
        """
        predicted = self.predict_maf(rpm, map_ambient, t_ambient_k)
        if predicted > 0:
            ratio = actual_maf / predicted
            self.correction_factor = (1 - alpha) * self.correction_factor + alpha * ratio

    def save(self, path: str):
        """Serialize model to disk."""
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> 'IntakeTwinModel':
        """Load model from disk."""
        return joblib.load(path)
