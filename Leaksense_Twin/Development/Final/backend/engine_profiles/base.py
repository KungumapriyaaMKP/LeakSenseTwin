"""
LeakSense Twin — Engine Profile Schema
Pulls every engine-specific physics coefficient and zone-topology definition
into a swappable profile, so adding a new Caterpillar engine is a config
change (write a profile JSON from its real published spec sheet), not a
code change. See engine_profiles/c18.json and c9_3b.json for two real,
independently-calibrated examples.

Design notes:
  - Plain dataclasses, not pydantic — kept framework-free so this is
    importable from data_generator.py/ml/train.py without pulling in FastAPI.
  - from_dict()/from_json() raise on any missing field rather than silently
    defaulting — a missing field is almost always a transcription error,
    not an intentional omission.
  - `calibrated` + `notes` document exactly what's real (transcribed from a
    published Cat spec sheet) vs. engineering-estimated (e.g. compressor map
    polynomials, which no public document specifies for either engine) —
    see each profile's `notes` field rather than assuming uniform precision.
"""

from dataclasses import dataclass, field
import json


@dataclass
class IntakeProfile:
    displacement_L: float
    n_cylinders: int
    ve_base: float
    ve_rpm_peak: float
    ve_quadratic_coeff: float
    ve_clip_min: float
    ve_clip_max: float


@dataclass
class ChargeAirProfile:
    turbo_eff_compressor: float
    cac_effectiveness: float
    compressor_coeffs: dict
    pr_clip_min: float
    pr_clip_max: float
    cac_dp_coeff: float
    cac_duct_loss_kpa: float
    manifold_runner_loss_kpa: float


@dataclass
class ExhaustProfile:
    n_cylinders: int
    turbo_eff_turbine: float
    lhv_diesel: float
    combustion_eff: float
    turbine_pr_const: float
    turbine_pr_rpm_coeff: float
    turbine_pr_maf_coeff: float
    turbine_rpm_norm: float
    turbine_maf_norm: float
    turbine_pr_clip_min: float
    turbine_pr_clip_max: float
    dpf_dp_coeff: float
    dpf_dp_exponent: float
    t_exh_clip_min_k: float
    t_exh_clip_max_k: float
    dpf_in_loss_k: float
    dpf_out_loss_k: float


@dataclass
class OperatingRangeProfile:
    rpm_min: float
    rpm_max: float
    rpm_steps: int


@dataclass
class ZoneTopologyProfile:
    n_zones: int
    zone_names: list
    action_map: dict

    @property
    def n_classes(self) -> int:
        return self.n_zones + 1


@dataclass
class EngineProfile:
    name: str
    description: str
    calibrated: bool
    notes: str
    artifact_dir: str
    r_air: float
    cp_air: float
    gamma: float
    map_ambient: float
    t_ambient_k: float
    torque_curve_rpm: list
    torque_curve_nm: list
    fuel_qty_min: float
    fuel_qty_max: float
    intake: IntakeProfile
    charge_air: ChargeAirProfile
    exhaust: ExhaustProfile
    operating_range: OperatingRangeProfile
    zones: ZoneTopologyProfile

    @property
    def n_classes(self) -> int:
        return self.zones.n_classes

    @staticmethod
    def from_dict(d: dict) -> "EngineProfile":
        def require(container, key, ctx):
            if key not in container:
                raise KeyError(f"Engine profile missing required field '{key}' in {ctx}")
            return container[key]

        intake = IntakeProfile(**d["intake"])
        charge_air = ChargeAirProfile(**d["charge_air"])
        exhaust = ExhaustProfile(**d["exhaust"])
        operating_range = OperatingRangeProfile(**d["operating_range"])

        zones_dict = dict(d["zones"])
        # JSON object keys are always strings — action_map is looked up by
        # int zone index everywhere else, so cast once here.
        zones_dict["action_map"] = {int(k): v for k, v in zones_dict["action_map"].items()}
        zones = ZoneTopologyProfile(**zones_dict)

        if len(zones.zone_names) != zones.n_zones + 1:
            raise ValueError(
                f"zone_names has {len(zones.zone_names)} entries, expected n_zones+1={zones.n_zones + 1}"
            )
        if len(zones.action_map) != zones.n_zones + 1:
            raise ValueError(
                f"action_map has {len(zones.action_map)} entries, expected n_zones+1={zones.n_zones + 1}"
            )
        if len(d["torque_curve_rpm"]) != len(d["torque_curve_nm"]):
            raise ValueError("torque_curve_rpm and torque_curve_nm must be the same length")

        return EngineProfile(
            name=require(d, "name", "root"),
            description=require(d, "description", "root"),
            calibrated=require(d, "calibrated", "root"),
            notes=require(d, "notes", "root"),
            artifact_dir=require(d, "artifact_dir", "root"),
            r_air=require(d, "r_air", "root"),
            cp_air=require(d, "cp_air", "root"),
            gamma=require(d, "gamma", "root"),
            map_ambient=require(d, "map_ambient", "root"),
            t_ambient_k=require(d, "t_ambient_k", "root"),
            torque_curve_rpm=d["torque_curve_rpm"],
            torque_curve_nm=d["torque_curve_nm"],
            fuel_qty_min=require(d, "fuel_qty_min", "root"),
            fuel_qty_max=require(d, "fuel_qty_max", "root"),
            intake=intake,
            charge_air=charge_air,
            exhaust=exhaust,
            operating_range=operating_range,
            zones=zones,
        )

    @staticmethod
    def from_json(path: str) -> "EngineProfile":
        with open(path, "r") as f:
            return EngineProfile.from_dict(json.load(f))
