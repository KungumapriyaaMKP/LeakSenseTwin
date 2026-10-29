"""
LeakSense Twin — Zone 3/4/5/6: Exhaust Digital Twin
Physics: Energy balance — predicts exhaust temperatures and back-pressure.

Engine-specific coefficients come from an EngineProfile (see
engine_profiles/) instead of being hardcoded, so the same physics model
works for any engine with a calibrated profile.
"""

import numpy as np
import joblib


class ExhaustTwinModel:
    """
    Physics-based Exhaust Zone Digital Twin.
    Models: Combustion → exhaust manifold → turbine expansion → aftertreatment.
    """

    def __init__(self, profile):
        self.profile = profile
        self.correction_t_exh = 1.0
        self.correction_dp = 1.0
        # DPF loading factor (increases with soot accumulation)
        self.dpf_loading = 1.0

    def turbine_pr(self, rpm: float, maf_kgh: float) -> float:
        """Turbine pressure ratio as a function of RPM and mass flow."""
        e = self.profile.exhaust
        rpm_norm = rpm / e.turbine_rpm_norm
        maf_norm = maf_kgh / e.turbine_maf_norm
        PR = e.turbine_pr_const + e.turbine_pr_rpm_coeff * rpm_norm + e.turbine_pr_maf_coeff * maf_norm
        return np.clip(PR, e.turbine_pr_clip_min, e.turbine_pr_clip_max)

    def dpf_pressure_drop(self, maf_kgh: float, rpm: float) -> float:
        """DPF back-pressure model — depends on exhaust flow rate and filter loading state."""
        e = self.profile.exhaust
        flow_rate = maf_kgh / 3600.0  # kg/s
        dp_base = e.dpf_dp_coeff * (flow_rate ** e.dpf_dp_exponent)  # kPa
        dp_loaded = dp_base * self.dpf_loading
        return dp_loaded * self.correction_dp

    def predict(self, maf_kgh: float, fuel_qty_mg: float, rpm: float,
                t_cac_out_k: float = 318.15) -> dict:
        """
        Predict exhaust system state.

        Args:
            maf_kgh: Mass air flow in kg/h
            fuel_qty_mg: Fuel injection quantity in mg/stroke
            rpm: Engine speed in RPM
            t_cac_out_k: CAC outlet temperature in Kelvin

        Returns:
            dict with T_exh_manifold_pred, T_post_turbine_pred, dP_dpf_pred
        """
        p = self.profile
        e = p.exhaust

        # Fuel mass flow rate (kg/s)
        # fuel_qty is per stroke per cylinder; 4-stroke → 1 injection per 2 revs
        fuel_mass_flow = fuel_qty_mg * 1e-6 * (rpm / 2.0 / 60.0) * e.n_cylinders

        # Air mass flow (kg/s)
        air_mass_flow = maf_kgh / 3600.0

        # Total exhaust mass flow
        total_flow = air_mass_flow + fuel_mass_flow

        # Combustion temperature (first law of thermodynamics)
        if total_flow > 0 and p.cp_air > 0:
            delta_T = (fuel_mass_flow * e.lhv_diesel * 1000.0 * e.combustion_eff) / \
                      (total_flow * p.cp_air)
            T_exh_manifold = t_cac_out_k + delta_T
        else:
            T_exh_manifold = t_cac_out_k + 400.0

        T_exh_manifold *= self.correction_t_exh

        # Clamp to realistic range
        T_exh_manifold = np.clip(T_exh_manifold, e.t_exh_clip_min_k, e.t_exh_clip_max_k)

        # Turbine expansion
        PR_turbine = self.turbine_pr(rpm, maf_kgh)
        T_post_turbine = T_exh_manifold * (
            1.0 - e.turbo_eff_turbine * (
                1.0 - PR_turbine ** (-(p.gamma - 1) / p.gamma)
            )
        )

        # DPF pressure drop
        dP_dpf = self.dpf_pressure_drop(maf_kgh, rpm)

        # DPF inlet/outlet temperatures (with heat loss)
        T_dpf_in = T_post_turbine - e.dpf_in_loss_k
        T_dpf_out = T_dpf_in - e.dpf_out_loss_k

        return {
            'T_exh_manifold_pred': float(T_exh_manifold),
            'T_post_turbine_pred': float(T_post_turbine),
            'dP_dpf_pred': float(dP_dpf),
            'T_dpf_in_pred': float(T_dpf_in),
            'T_dpf_out_pred': float(T_dpf_out),
        }

    def update_online(self, actual_t_exh: float, predicted_t_exh: float,
                      alpha: float = 0.01):
        """Online learning via EWA."""
        if predicted_t_exh > 0:
            ratio = actual_t_exh / predicted_t_exh
            self.correction_t_exh = (1 - alpha) * self.correction_t_exh + alpha * ratio

    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> 'ExhaustTwinModel':
        return joblib.load(path)
