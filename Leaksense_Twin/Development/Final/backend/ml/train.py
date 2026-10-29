"""
LeakSense Twin — Training Pipeline
Trains the production pipeline for the active engine profile
(LEAKSENSE_ENGINE env var, default "c18" — see engine_profiles/):
  1. Generate synthetic data calibrated to that engine
  2. Compute features via digital twins
  3. Train LeakNet (single model — leak detection + zone localization)
  4. Fit Energy Field detector
  5. Save all models under models/<engine>/

Model choice (single LeakNet over the old dual-net + RandomForest +
GradientBoosting ensemble) is based on measured comparison data — see
ml/benchmark.py and ml/benchmark_results.md.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support, accuracy_score
import joblib
import os
import sys
from pathlib import Path

# Setup paths — go up one level from ml/ to backend/
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

from config import ML_CONFIG
from engine_profiles import active_profile
from ml.models import LeakNet
from ml.energy_field import EnergyFieldDetector
from ml.regional_scaler import RegionalScaler
from twins.intake_twin import IntakeTwinModel
from twins.charge_air_twin import ChargeAirTwinModel
from twins.exhaust_twin import ExhaustTwinModel
from data_generator import generate_dataset


def compute_features_fast(df: pd.DataFrame,
                          intake_twin: IntakeTwinModel,
                          charge_twin: ChargeAirTwinModel,
                          exhaust_twin: ExhaustTwinModel,
                          profile) -> np.ndarray:
    """Compute 31-feature vector for the dataset (optimized)."""
    n = len(df)
    map_ambient = profile.map_ambient
    n_cylinders = profile.intake.n_cylinders

    # Raw sensor features (13)
    raw_cols = ['RPM', 'MAF', 'MAP_intake', 'MAP_boost', 'MAP_cac_in', 'MAP_cac_out',
                'T_intake', 'T_boost', 'T_cac_out', 'T_exh_manifold', 'T_dpf_in',
                'T_dpf_out', 'fuel_qty']
    raw_features = df[raw_cols].values

    # Residual features (9) — vectorized where possible
    residuals = np.zeros((n, 9))
    print("  Computing digital twin residuals...")
    for i in range(n):
        if i % 10000 == 0:
            print(f"    Processing sample {i}/{n}...")
        row = df.iloc[i]
        rpm = row['RPM']
        maf = row['MAF']
        t_intake_k = row['T_intake'] + 273.15
        fuel = row['fuel_qty']
        t_cac_k = row['T_cac_out'] + 273.15

        maf_pred = intake_twin.predict(rpm, map_ambient, t_intake_k, fuel)
        residuals[i, 0] = maf - maf_pred

        ca = charge_twin.predict(maf, rpm, t_intake_k, map_ambient)
        residuals[i, 1] = row['MAP_boost'] - ca['MAP_boost_pred']
        residuals[i, 2] = row['T_boost'] - (ca['T_boost_pred'] - 273.15)
        residuals[i, 3] = row['T_cac_out'] - (ca['T_cac_out_pred'] - 273.15)
        residuals[i, 4] = row['MAP_cac_out'] - ca['MAP_cac_out_pred']

        ex = exhaust_twin.predict(maf, fuel, rpm, t_cac_k)
        residuals[i, 5] = row['T_exh_manifold'] - (ex['T_exh_manifold_pred'] - 273.15)
        t_post = row.get('T_post_turbine', 400.0)
        residuals[i, 6] = t_post - (ex['T_post_turbine_pred'] - 273.15)
        dP = row.get('dP_dpf', 0.0)
        residuals[i, 7] = dP - ex['dP_dpf_pred']
        residuals[i, 8] = row['MAP_intake'] - ca['MAP_intake_pred']

    # Derived ratio features (6)
    ratios = np.zeros((n, 6))
    ratios[:, 0] = df['MAP_boost'].values / (map_ambient + 1e-8)
    t_boost = df['T_boost'].values
    t_intake = df['T_intake'].values
    t_cac = df['T_cac_out'].values
    denom = np.abs(t_boost - t_intake) + 1e-8
    ratios[:, 1] = (t_boost - t_cac) / denom
    ratios[:, 2] = df['T_exh_manifold'].values / (t_cac + 273.15 + 1e-8)
    ratios[:, 3] = df['MAP_boost'].values / (df['MAF'].values + 1e-8)
    dP_vals = df['dP_dpf'].values if 'dP_dpf' in df.columns else np.zeros(n)
    ratios[:, 4] = dP_vals / (df['MAP_boost'].values + 1e-8)
    fuel_flow = df['fuel_qty'].values * 1e-6 * (df['RPM'].values / 120.0) * n_cylinders
    ratios[:, 5] = (df['MAF'].values / 3600.0) / (fuel_flow + 1e-8)

    # Statistical features (3)
    stats = np.zeros((n, 3))
    stats[:, 0] = np.abs(residuals[:, 0]) * 0.1
    stats[:, 1] = np.abs(residuals[:, 1]) * 0.1
    stats[:, 2] = np.abs(residuals[:, 5]) * 0.1

    features = np.hstack([raw_features, residuals, ratios, stats])
    return features


def train_leak_net(X_train_t, y_train_t, X_val_t, y_val_t,
                   input_dim: int, n_classes: int, config: dict) -> LeakNet:
    """Train the unified LeakNet (leak detection + zone localization)."""
    model = LeakNet(input_dim=input_dim, n_classes=n_classes)
    # No label smoothing — measured to strictly improve accuracy, macro-F1, AND
    # lower the healthy false-positive rate vs. label_smoothing=0.05 on this
    # data (smoothing softened the decision boundary in a way that specifically
    # hurt recall on the smallest, hardest-to-detect leaks). See ml/benchmark.py
    # history / project notes for the comparison.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(),
                            lr=config['learning_rate'],
                            weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['max_epochs'])

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_dl = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)

    best_acc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(config['max_epochs']):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_dl:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t).argmax(dim=1)
            acc = float((val_pred == y_val_t).float().mean())

        if acc > best_acc:
            best_acc = acc
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 20 == 0:
            avg_loss = total_loss / len(train_dl)
            print(f"    Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Val Acc: {acc:.4f} | Best: {best_acc:.4f}")

        if patience_counter >= config['patience']:
            print(f"    Early stopping at epoch {epoch}")
            break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    print(f"  LeakNet final best validation accuracy: {best_acc:.4f}")
    return model


def fit_energy_field(df_healthy: pd.DataFrame) -> EnergyFieldDetector:
    """Fit the energy field detector on healthy data."""
    ef = EnergyFieldDetector()

    ef_cols = ['MAF', 'MAP_boost', 'MAP_cac_out', 'T_cac_out',
               'T_exh_manifold', 'dP_dpf', 'RPM']

    if 'dP_dpf' not in df_healthy.columns:
        df_healthy = df_healthy.copy()
        df_healthy['dP_dpf'] = 0.0

    data = df_healthy[ef_cols].values
    window_size = 30
    windows = []
    for i in range(0, len(data) - window_size, window_size):
        windows.append(data[i:i+window_size])

    if len(windows) > 0:
        ef.fit(windows)
        print(f"  Energy Field fitted on {len(windows)} windows")
    else:
        print("  Warning: Not enough data for energy field fitting")

    return ef


def run_training(profile=None):
    """Main training pipeline for the given (or active) engine profile."""
    profile = profile or active_profile()

    print("=" * 60)
    print(f"LeakSense Twin — Training Pipeline ({profile.name})")
    print("=" * 60)

    models_dir = Path(BACKEND_DIR) / "models" / profile.artifact_dir
    data_dir = Path(BACKEND_DIR) / "data"
    models_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(exist_ok=True)

    # Step 1: Generate data
    print("\n[1/6] Generating synthetic training data...")
    data_file = data_dir / f"training_data_{profile.artifact_dir}.csv"
    if data_file.exists():
        print("  Loading existing dataset...")
        df = pd.read_csv(data_file)
    else:
        df = generate_dataset(profile=profile)
        df.to_csv(data_file, index=False)

    if 'dP_dpf' not in df.columns:
        df['dP_dpf'] = 0.0
    if 'T_post_turbine' not in df.columns:
        df['T_post_turbine'] = 400.0

    # Step 2: Initialize digital twins
    print("\n[2/6] Initializing digital twin models...")
    intake_twin = IntakeTwinModel(profile)
    charge_twin = ChargeAirTwinModel(profile)
    exhaust_twin = ExhaustTwinModel(profile)

    intake_twin.save(str(models_dir / "intake_twin.joblib"))
    charge_twin.save(str(models_dir / "charge_air_twin.joblib"))
    exhaust_twin.save(str(models_dir / "exhaust_twin.joblib"))
    print("  Digital twins saved.")

    # Step 3: Compute features
    print("\n[3/6] Computing feature vectors (31 features)...")
    features = compute_features_fast(df, intake_twin, charge_twin, exhaust_twin, profile)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    y_zone = df['leak_zone'].astype(np.int64).values
    print(f"  Feature matrix shape: {features.shape}")
    print(f"  Healthy: {(y_zone==0).sum()}, Leak: {(y_zone>0).sum()}")

    # Step 4: Train/val/test split
    print("\n[4/6] Splitting data (70/15/15)...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        features, y_zone, test_size=0.30, random_state=42, stratify=y_zone
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    # Operating-point-conditioned scaling — see RESEARCH_NOTES.md
    scaler = RegionalScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    joblib.dump(scaler, str(models_dir / "scaler.joblib"))

    X_train_t = torch.FloatTensor(X_train_s)
    X_val_t = torch.FloatTensor(X_val_s)
    X_test_t = torch.FloatTensor(X_test_s)
    y_train_t = torch.LongTensor(y_train)
    y_val_t = torch.LongTensor(y_val)

    input_dim = X_train_s.shape[1]
    print(f"  Input dimension: {input_dim}")

    # Step 5: Train LeakNet
    print("\n[5/6] Training LeakNet (unified detection + localization)...")
    config = ML_CONFIG.copy()
    leak_net = train_leak_net(X_train_t, y_train_t, X_val_t, y_val_t, input_dim, profile.n_classes, config)
    torch.save(leak_net.state_dict(), str(models_dir / "leak_net.pth"))

    # Step 6: Fit Energy Field
    print("\n[6/6] Fitting Energy Field detector...")
    df_healthy = df[df['leak_zone'] == 0]
    ef_detector = fit_energy_field(df_healthy)
    ef_detector.save(str(models_dir / "energy_field_detector.joblib"))

    # ── Evaluation ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"EVALUATION ON TEST SET ({profile.name})")
    print("=" * 60)

    leak_net.eval()
    with torch.no_grad():
        test_zones = leak_net(X_test_t).argmax(dim=1).numpy()

    zone_names = profile.zones.zone_names
    print("\n Zone Localization (LeakNet):")
    print(classification_report(y_test, test_zones, target_names=zone_names, zero_division=0))

    yb_test = (y_test > 0).astype(int)
    yb_pred = (test_zones > 0).astype(int)
    test_acc = accuracy_score(y_test, test_zones)
    test_macro_f1 = f1_score(y_test, test_zones, average='macro', zero_division=0)
    leak_p, leak_r, leak_f1, _ = precision_recall_fscore_support(yb_test, yb_pred, average='binary', zero_division=0)

    print(f"\n Overall accuracy: {test_acc:.4f}")
    print(f" Macro F1: {test_macro_f1:.4f}")
    print(f" Leak detection — Precision: {leak_p:.4f}, Recall: {leak_r:.4f}, F1: {leak_f1:.4f}")

    config_info = {
        'engine': profile.name,
        'input_dim': input_dim,
        'n_classes': profile.n_classes,
        'n_samples': len(df),
        'n_features': input_dim,
        'test_accuracy': float(test_acc),
        'test_macro_f1': float(test_macro_f1),
        'test_leak_precision': float(leak_p),
        'test_leak_recall': float(leak_r),
        'test_leak_f1': float(leak_f1),
    }
    joblib.dump(config_info, str(models_dir / "config.joblib"))

    print("\n" + "=" * 60)
    print("All models saved to:", models_dir)
    print("Training complete!")
    print("=" * 60)

    return config_info


if __name__ == "__main__":
    run_training()
