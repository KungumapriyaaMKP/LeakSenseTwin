"""
LeakSense Twin — Model Selection Benchmark
Trains every candidate model on the identical data split and reports
accuracy, latency, and footprint, so model choice is driven by measured
numbers rather than assumption. See backend/ml/benchmark_results.md for
the most recent run's output.

Candidates:
  - Logistic Regression          (cheap baseline)
  - RandomForestClassifier
  - GradientBoostingClassifier   (sklearn)
  - LightGBM
  - XGBoost
  - LeakNet (PyTorch)            (production model — see ml/models.py, ml/train.py)

For every tree/boosting/linear family, two patterns are tested:
  - "dual"   : separate binary detector + separate 6-class zone localizer (today's pattern)
  - "single" : one 6-class classifier; binary = (argmax != 0), confidence = 1 - P(class 0)
"""

import io
import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_sample_weight
import lightgbm as lgb
import xgboost as xgb

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

from engine_profiles import active_profile
from twins.intake_twin import IntakeTwinModel
from twins.charge_air_twin import ChargeAirTwinModel
from twins.exhaust_twin import ExhaustTwinModel
from ml.train import compute_features_fast, train_leak_net

RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results.md")


# ─── Data loading ───────────────────────────────────────────────────────────
def load_data(profile=None):
    profile = profile or active_profile()
    data_path = os.path.join(BACKEND_DIR, "data", f"training_data_{profile.artifact_dir}.csv")
    df = pd.read_csv(data_path)
    if 'dP_dpf' not in df.columns:
        df['dP_dpf'] = 0.0
    if 'T_post_turbine' not in df.columns:
        df['T_post_turbine'] = 400.0

    intake_twin = IntakeTwinModel(profile)
    charge_twin = ChargeAirTwinModel(profile)
    exhaust_twin = ExhaustTwinModel(profile)
    features = compute_features_fast(df, intake_twin, charge_twin, exhaust_twin, profile)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    y_zone = df['leak_zone'].astype(np.int64).values
    y_binary = (y_zone > 0).astype(np.float32)

    X_train, X_temp, yb_train, yb_temp, yz_train, yz_temp = train_test_split(
        features, y_binary, y_zone, test_size=0.30, random_state=42, stratify=y_zone
    )
    X_val, X_test, yb_val, yb_test, yz_val, yz_test = train_test_split(
        X_temp, yb_temp, yz_temp, test_size=0.50, random_state=42, stratify=yz_temp
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    return {
        'X_train': X_train_s, 'X_val': X_val_s, 'X_test': X_test_s,
        'yb_train': yb_train, 'yb_val': yb_val, 'yb_test': yb_test,
        'yz_train': yz_train, 'yz_val': yz_val, 'yz_test': yz_test,
        'n_classes': int(y_zone.max()) + 1,
        'input_dim': X_train_s.shape[1],
    }


# ─── Metrics helpers ─────────────────────────────────────────────────────────
def binary_metrics_from_zone_pred(yz_true, zone_pred):
    yb_true = (yz_true > 0).astype(int)
    yb_pred = (zone_pred > 0).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(yb_true, yb_pred, average='binary', zero_division=0)
    return p, r, f1


def measure_latency_sklearn(model, X_test, n=300):
    times = []
    n = min(n, len(X_test))
    for i in range(n):
        row = X_test[i:i + 1]
        t0 = time.perf_counter()
        model.predict_proba(row)
        times.append(time.perf_counter() - t0)
    times = np.array(times)
    return float(times.mean() * 1e6), float(np.percentile(times, 95) * 1e6)  # microseconds


def measure_latency_torch(model, X_test, n=300):
    times = []
    n = min(n, len(X_test))
    model.eval()
    with torch.no_grad():
        for i in range(n):
            row = torch.FloatTensor(X_test[i:i + 1])
            t0 = time.perf_counter()
            model(row)
            times.append(time.perf_counter() - t0)
    times = np.array(times)
    return float(times.mean() * 1e6), float(np.percentile(times, 95) * 1e6)


def model_size_bytes_sklearn(model):
    buf = io.BytesIO()
    joblib.dump(model, buf)
    return len(buf.getvalue())


def model_size_bytes_torch(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return len(buf.getvalue())


def fmt_kb(n_bytes):
    return f"{n_bytes / 1024:.1f} KB"


RESULTS = []


def record(name, pattern, acc, macro_f1, leak_p, leak_r, leak_f1, lat_mean_us, lat_p95_us, size_bytes, train_s):
    RESULTS.append({
        'name': name, 'pattern': pattern, 'accuracy': acc, 'macro_f1': macro_f1,
        'leak_precision': leak_p, 'leak_recall': leak_r, 'leak_f1': leak_f1,
        'latency_mean_us': lat_mean_us, 'latency_p95_us': lat_p95_us,
        'size': fmt_kb(size_bytes), 'train_s': train_s,
    })


# ─── Tree / linear family candidates ────────────────────────────────────────
def run_sklearn_family(name, make_binary_model, make_multiclass_model, data):
    X_train, X_test = data['X_train'], data['X_test']
    yb_train, yb_test = data['yb_train'], data['yb_test']
    yz_train, yz_test = data['yz_train'], data['yz_test']

    # --- dual pattern: separate binary + separate multiclass ---
    sw_b = compute_sample_weight('balanced', yb_train)
    t0 = time.perf_counter()
    bin_model = make_binary_model()
    try:
        bin_model.fit(X_train, yb_train, sample_weight=sw_b)
    except TypeError:
        bin_model.fit(X_train, yb_train)
    sw_z = compute_sample_weight('balanced', yz_train)
    loc_model = make_multiclass_model()
    try:
        loc_model.fit(X_train, yz_train, sample_weight=sw_z)
    except TypeError:
        loc_model.fit(X_train, yz_train)
    train_time = time.perf_counter() - t0

    zone_pred = loc_model.predict(X_test)
    acc = accuracy_score(yz_test, zone_pred)
    macro_f1 = f1_score(yz_test, zone_pred, average='macro', zero_division=0)
    yb_pred = bin_model.predict(X_test)
    p, r, f1, _ = precision_recall_fscore_support((yz_test > 0).astype(int), yb_pred, average='binary', zero_division=0)
    lat_mean, lat_p95 = measure_latency_sklearn(bin_model, X_test)
    size = model_size_bytes_sklearn(bin_model) + model_size_bytes_sklearn(loc_model)
    record(name, 'dual', acc, macro_f1, p, r, f1, lat_mean, lat_p95, size, train_time)

    # --- single pattern: one multiclass model, derive binary from it ---
    t0 = time.perf_counter()
    single_model = make_multiclass_model()
    try:
        single_model.fit(X_train, yz_train, sample_weight=sw_z)
    except TypeError:
        single_model.fit(X_train, yz_train)
    train_time = time.perf_counter() - t0

    zone_pred = single_model.predict(X_test)
    acc = accuracy_score(yz_test, zone_pred)
    macro_f1 = f1_score(yz_test, zone_pred, average='macro', zero_division=0)
    p, r, f1 = binary_metrics_from_zone_pred(yz_test, zone_pred)
    lat_mean, lat_p95 = measure_latency_sklearn(single_model, X_test)
    size = model_size_bytes_sklearn(single_model)
    record(name, 'single', acc, macro_f1, p, r, f1, lat_mean, lat_p95, size, train_time)


# ─── LeakNet (production model) ─────────────────────────────────────────────
def run_leak_net(data, max_epochs=80, patience=10):
    """Trains the actual production model/trainer (ml/models.py, ml/train.py)
    so this benchmark stays a true comparison against what's deployed."""
    n_classes = data['n_classes']
    cfg = {'learning_rate': 1e-3, 'weight_decay': 1e-4, 'batch_size': 256,
           'max_epochs': max_epochs, 'patience': patience, 'n_classes': n_classes}

    X_train_t = torch.FloatTensor(data['X_train'])
    y_train_t = torch.LongTensor(data['yz_train'])
    X_val_t = torch.FloatTensor(data['X_val'])
    y_val_t = torch.LongTensor(data['yz_val'])
    X_test_t = torch.FloatTensor(data['X_test'])

    t0 = time.perf_counter()
    model = train_leak_net(X_train_t, y_train_t, X_val_t, y_val_t, data['input_dim'], n_classes, cfg)
    train_time = time.perf_counter() - t0

    model.eval()
    with torch.no_grad():
        zone_pred = model(X_test_t).argmax(dim=1).numpy()
    acc = accuracy_score(data['yz_test'], zone_pred)
    macro_f1 = f1_score(data['yz_test'], zone_pred, average='macro', zero_division=0)
    p, r, f1 = binary_metrics_from_zone_pred(data['yz_test'], zone_pred)
    lat_mean, lat_p95 = measure_latency_torch(model, data['X_test'])
    size = model_size_bytes_torch(model)
    record("LeakNet (production)", "single", acc, macro_f1, p, r, f1, lat_mean, lat_p95, size, train_time)


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    print("Loading data and computing features...")
    data = load_data()
    print(f"  Train/Val/Test: {len(data['X_train'])}/{len(data['X_val'])}/{len(data['X_test'])}, "
          f"input_dim={data['input_dim']}")

    print("\nLogistic Regression...")
    run_sklearn_family(
        "Logistic Regression",
        lambda: LogisticRegression(max_iter=2000),
        lambda: LogisticRegression(max_iter=2000),
        data,
    )

    print("RandomForest...")
    run_sklearn_family(
        "RandomForest",
        lambda: RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
        lambda: RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
        data,
    )

    print("GradientBoosting (sklearn)...")
    run_sklearn_family(
        "GradientBoosting (sklearn)",
        lambda: GradientBoostingClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42),
        lambda: GradientBoostingClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42),
        data,
    )

    print("LightGBM...")
    run_sklearn_family(
        "LightGBM",
        lambda: lgb.LGBMClassifier(n_estimators=200, max_depth=8, learning_rate=0.08, random_state=42, verbosity=-1),
        lambda: lgb.LGBMClassifier(n_estimators=200, max_depth=8, learning_rate=0.08, random_state=42, verbosity=-1,
                                    objective='multiclass', num_class=data['n_classes']),
        data,
    )

    print("XGBoost...")
    run_sklearn_family(
        "XGBoost",
        lambda: xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.08, random_state=42,
                                   eval_metric='logloss', verbosity=0),
        lambda: xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.08, random_state=42,
                                   eval_metric='mlogloss', verbosity=0, num_class=data['n_classes'],
                                   objective='multi:softprob'),
        data,
    )

    print("LeakNet (production)...")
    run_leak_net(data)

    # ── Report ──────────────────────────────────────────────────────────────
    df = pd.DataFrame(RESULTS)
    df = df.sort_values(['macro_f1'], ascending=False)
    pd.set_option('display.width', 160)
    pd.set_option('display.max_columns', 20)
    print("\n" + "=" * 100)
    print(df.to_string(index=False))
    print("=" * 100)

    with open(RESULTS_PATH, 'w') as f:
        f.write("# LeakSense Twin — Model Benchmark Results\n\n")
        f.write(f"Dataset: `data/training_data.csv` ({len(data['X_train']) + len(data['X_val']) + len(data['X_test'])} samples, "
                f"{data['input_dim']} features). Test set: {len(data['X_test'])} samples.\n\n")
        f.write("`dual` = separate binary detector + separate 6-class zone localizer (today's pattern).\n")
        f.write("`single` = one 6-class classifier; binary leak flag and confidence derived from it.\n\n")
        f.write("```\n")
        f.write(df.to_string(index=False))
        f.write("\n```\n")
    print(f"\nSaved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
