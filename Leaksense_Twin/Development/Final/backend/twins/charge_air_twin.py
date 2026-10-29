"""
LeakSense Twin — Zone 2: Charge Air System Digital Twin
Physics: Turbocharger compressor map + CAC heat exchanger model.
Predicts: MAP_boost, T_boost, T_cac_out, MAP_cac_out, MAP_intake

Engine-specific coefficients come from an EngineProfile (see
engine_profiles/) instead of being hardcoded, so the same physics model
works for any engine with a calibrated profile.
"""

import numpy as np
import joblib


class ChargeAirTwinModel:
    """
    Physics-based Charge Air Zone Digital Twin.
    Models: Compressor map → boost pressure → CAC cooling → manifold pressure.
    """

    def __init__(self, profile):
        self.profile = profile
        self.correction_boost = 1.0
        self.correction_temp = 1.0

    def compressor_map(self, corrected_flow: float, corrected_speed: float) -> float:
        """
        Compressor map returning pressure ratio.
        PR = c00 + c10*Q + c01*N + c20*Q² + c11*Q*N + c02*N²
        """
        c = self.profile.charge_air.compressor_coeffs
        Q = corrected_flow
        N = corrected_speed
        PR = (c['c00'] + c['c10'] * Q + c['c01'] * N +
              c['c20'] * Q**2 + c['c11'] * Q * N + c['c02'] * N**2)
        return np.clip(PR, self.profile.charge_air.pr_clip_min, self.profile.charge_air.pr_clip_max)

    def cac_pressure_drop(self, maf_kgh: float) -> float:
        """CAC pressure drop via Darcy-Weisbach approximation (dP proportional to mass_flow²)."""
        return self.profile.charge_air.cac_dp_coeff * (maf_kgh / 100.0) ** 2

    def predict(self, maf_actual: float, rpm: float,
                t_ambient_k: float = None,
                map_ambient: float = None) -> dict:
        """
        Predict charge air system state.

        Returns dict with: MAP_boost_pred, T_boost_pred, T_cac_out_pred,
                          MAP_cac_out_pred, MAP_intake_pred
        """
        p = self.profile
        t_ambient_k = p.t_ambient_k if t_ambient_k is None else t_ambient_k
        map_ambient = p.map_ambient if map_ambient is None else map_ambient

        # Corrected flow and speed for compressor map
        corrected_flow = maf_actual * np.sqrt(t_ambient_k / 288.15) / (map_ambient / 101.325)
        corrected_speed = rpm / np.sqrt(t_ambient_k / 288.15)

        # Pressure ratio from compressor map
        PR = self.compressor_map(corrected_flow, corrected_speed)

        # Boost pressure
        map_boost = PR * map_ambient * self.correction_boost

        # Compressor outlet temperature (isentropic + efficiency)
        T_boost = t_ambient_k * (1.0 + (PR ** ((p.gamma - 1) / p.gamma) - 1.0)
                                  / p.charge_air.turbo_eff_compressor)

        # CAC heat exchanger — temperature drop
        T_cac_out = T_boost - p.charge_air.cac_effectiveness * (T_boost - t_ambient_k)
        T_cac_out *= self.correction_temp

        # Pressure after CAC (small pressure drop)
        MAP_cac_out = map_boost - self.cac_pressure_drop(maf_actual)

        # Intake manifold pressure (slight additional loss)
        MAP_intake = MAP_cac_out - p.charge_air.manifold_runner_loss_kpa

        return {
            'MAP_boost_pred': float(map_boost),
            'T_boost_pred': float(T_boost),
            'T_cac_out_pred': float(T_cac_out),
            'MAP_cac_out_pred': float(MAP_cac_out),
            'MAP_intake_pred': float(MAP_intake),
        }

    def update_online(self, actual_boost: float, predicted_boost: float,
                      actual_temp: float, predicted_temp: float,
                      alpha: float = 0.01):
        """Online learning via EWA for drift compensation."""
        if predicted_boost > 0:
            ratio_b = actual_boost / predicted_boost
            self.correction_boost = (1 - alpha) * self.correction_boost + alpha * ratio_b
        if predicted_temp > 0:
            ratio_t = actual_temp / predicted_temp
            self.correction_temp = (1 - alpha) * self.correction_temp + alpha * ratio_t

    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> 'ChargeAirTwinModel':
        return joblib.load(path)
