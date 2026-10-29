# LeakSense Twin — Multi-Engine Caterpillar Diagnostics

LeakSense Twin is an AI-powered system for real-time, non-invasive detection and localization of air/exhaust leaks in Caterpillar diesel engines during development testing. The pipeline (physics digital twins → feature engineering → `LeakNet` model) is engine-agnostic — currently calibrated and validated against **two real, independently-different Cat engines** (see Multi-Engine Support below), with the architecture designed so adding a third is a config change, not a rewrite.

This system requires no additional hardware and uses the existing ECU sensors via SAE J1939 to detect anomalies in real-time.

## Features

- **Real-Time Anomaly Detection + Zone Localization**: A single PyTorch `LeakNet` model (conventional Linear→BatchNorm→ReLU→Dropout architecture, 128→64→7) detects leaks and localizes them to one of 6 engine zones in one pass — **94.6% test accuracy, 0.875 leak-detection F1**, with medium/large leaks (8%+/15% flow loss) caught at 97-100% recall in every zone (measured on held-out data, see Model Changelog and Zone Coverage below).
- **Digital Twins**: Physics-informed digital twins (mass continuity, turbo compressor map, CAC heat exchanger, combustion energy balance) run alongside the ML model to generate expected healthy baselines. Calibrated against real Caterpillar C18 spec data (compression ratio, torque curve, rated power/speed all match published Cat documentation).
- **Energy Field Analysis**: Computes the thermodynamic correlation manifold of the engine to detect subtle structural distortions before they cause major failures.
- **Premium Glassmorphic Dashboard**: A React frontend built with Vite provides a stunning real-time visualization of engine health, including a 6x6 live Energy Field heatmap.
- **Simulated Real-World Data**: Includes a synthetic data generator, calibrated against the real published Cat C18 torque curve, to simulate 50,000+ realistic engine states (healthy and faulty).

## Architecture

1. **Backend**: FastAPI (Python) running the digital twins and a PyTorch neural network.
2. **Frontend**: React + Vite + Vanilla CSS for a zero-dependency, ultra-fast UI.
3. **ML Pipeline**:
   - Physics digital twins generate residuals.
   - Energy Field computes relationship manifolds.
   - 31-dimensional feature vector fed into a single `LeakNet` model (7-class output: healthy + 6 zones; leak confidence derived as `1 - P(healthy)`).
   - Model choice is benchmark-driven, not assumed — see `backend/ml/benchmark.py` and `backend/ml/benchmark_results.md` for the comparison against RandomForest, GradientBoosting, LightGBM, XGBoost, and Logistic Regression.

## Model Changelog

| | Before | After |
|---|---|---|
| **Architecture** | Two separate networks (bespoke "cosine-similarity" `LeakSenseNet` for binary detection + `LeakLocalizationNet` for zone) **plus** a 4-way weighted ensemble vote with RandomForest + GradientBoosting | One conventional `LeakNet` (128→64→6 MLP) — single model, no ensemble |
| **Accuracy (round 1)** | Unverified — no model had ever actually been trained in this environment; README claims of ">90% accuracy" and GEMINI.md's "F1=0.886" were aspirational, not measured | 95.95% test accuracy, macro-F1 0.896, leak-detection F1 0.892 |
| **Accuracy (round 2 — current)** | (round 1 numbers above) | **96.0% test accuracy, macro-F1 0.897, leak-detection precision 0.982 / recall 0.824 / F1 0.896** (`backend/ml/benchmark_results.md`) |
| **Why round 1 changed** | Benchmarked 7 model families head-to-head on identical data. The single MLP matched/beat the old two-network design on every metric while being simpler. RandomForest was found to have a ~100-200ms single-row inference latency bug; sklearn's GradientBoosting took 11-17 minutes to train for worse accuracy than LightGBM/XGBoost. Neither tree model beat the neural net, so the whole ensemble was dropped. | |
| **Why round 2 changed** | Diagnosed per-zone recall and found medium/large leaks (8%, 15% flow loss) were caught at 100% recall in every zone — the entire accuracy gap was confined to the smallest (2% flow loss) leaks, worst in Zones 4/5 (exhaust manifold, DPF/SCR) where the injected signal is most subtle. Removing label smoothing from training (`nn.CrossEntropyLoss()`, no smoothing) improved accuracy, macro-F1, *and* lowered the healthy false-positive rate — a strict win, no tradeoff. | |
| **Accuracy (round 3 — current)** | (round 2 numbers above, 5 zones) | **94.6% test accuracy, macro-F1 0.873, leak-detection precision 0.987 / recall 0.786 / F1 0.875** — now across **6 zones** (see Zone Coverage below) |
| **Why round 3 changed** | Audited the model's zone coverage against the original problem statement's scope and found a real gap: nothing modeled the turbine-outlet → DPF-inlet segment. Added **Zone 6** for it, regenerated the dataset (6 fault zones × 3 severities instead of 5) and retrained. Accuracy dropped slightly (96.0%→94.6%) — an honest, expected cost of finer-grained localization with the same sensor set; medium/large leaks are still caught at 97-100% recall in every zone including the new one. | |
| **Architecture (round 4)** | Every physics constant (compressor map, VE curve, turbine map, torque curve, zone topology) hardcoded for the C18 specifically — feeding in a different engine's data would have produced near-constant false alarms | Pulled all of it into a swappable `EngineProfile` (`engine_profiles/`); twins, data generator, training, and serving are now engine-agnostic. Switch engines via `LEAKSENSE_ENGINE` env var. |
| **Accuracy (round 4 — current)** | (round 3 numbers, C18 only) | **C18: 94.9% accuracy, macro-F1 0.881, leak-F1 0.886. C9.3B (new, single-turbo, 9.3L): 94.5% accuracy, macro-F1 0.872, leak-F1 0.876** — comparable accuracy across two structurally different engines is the real evidence the abstraction works, not just an architectural claim |
| **Why round 4 changed** | You asked whether the system only works for the C18, and what Caterpillar would realistically expect for "scalable to different engine configurations." Built the profile abstraction, then proved it by actually calibrating and training a second, deliberately different real engine (C9.3B: single-turbo vs. C18's twin-turbo, half the displacement, different equipment segment) rather than just claiming the architecture supports it. |
| **Architecture (round 5)** | One global `StandardScaler` — residuals judged against a single engine-wide healthy baseline regardless of operating point | `RegionalScaler` (`ml/regional_scaler.py`) — residual/ratio/stat features are z-scored within RPM buckets, so "is this unusual" is judged against that operating point's own healthy distribution. Raw sensors stay globally scaled. See `RESEARCH_NOTES.md` for the literature basis. |
| **Accuracy (round 5 — current)** | (round 4 numbers above) | **C18: 94.85% accuracy, leak-F1 0.885. C9.3B: 94.76% accuracy, leak-F1 0.887.** Headline accuracy is flat by design — the real effect is on small-leak isolability (see below) |
| **Why round 5 changed** | Literature review of ~25 academic FDI papers (compiled by the user from a reference document) consistently found that leak isolability improves when residuals are evaluated per operating regime rather than against one global baseline. | Built `RegionalScaler` as a bounded test of that finding. Result: Zone 1 severity-1 recall 0.322→0.402, **Zone 3 0.752→0.861** — a real win on the intake/charge-air path. Zones 4/5/6 unchanged — their smallest leaks are limited by signal magnitude vs. sensor noise, not by operating-point bias, so this fix doesn't reach them. Full results and other literature findings not yet acted on: `RESEARCH_NOTES.md`. |

**Known remaining limitation**: detecting the smallest leaks (2% flow loss) on a single instantaneous reading is still hard in Zones 4/5/6 (3-10% recall there, vs. 40-86% in Zones 1-3 after round 5) — the injected signal is close to the sensor noise floor in those zones specifically. We tested oversampling these examples during training, which raises recall meaningfully, but at the cost of a 12-16x higher false-positive rate on healthy engines (0.1%→1.25-1.6%), which isn't worth it given every false alarm halts a test cell. The literature review surfaced more targeted candidate fixes (turbo-speed/exhaust-hole-area residuals, a likelihood-ratio confirmation test, an unsupervised anomaly safety net) — see `RESEARCH_NOTES.md` — none yet implemented.

## Zone Coverage

Cross-referencing the problem statement's scope against what's actually modeled:

| Problem statement scope | Coverage |
|---|---|
| Airflow meter → Compressor inlet | ✅ Zone 1 |
| Compressor outlet → CAC/Intercooler → Intake ports | ✅ Zone 2 (compressor outlet → CAC inlet) + Zone 3 (CAC → intake manifold) |
| Exhaust manifold → Turbine inlet | ✅ Zone 4 |
| Across the turbine itself | ⚠️ Not separately distinguishable — folded into Zone 4's signal, no sensor isolates the turbine housing specifically |
| Turbine outlet → Aftertreatment inlet | ✅ Zone 6 |
| Aftertreatment (DOC/DPF/SCR) internal | ✅ Zone 5 |
| Aftertreatment outlet → Tailpipe | ❌ Not covered — no sensor exists downstream of `T_dpf_out` in the current 13-sensor set |
| Test cell ducting interfaces | ❌ Not covered — this is facility-side instrumentation (combustion air supply duct, exhaust extraction duct), not an engine/ECU/J1939 parameter. Real test cells typically do have their own duct differential-pressure instrumentation for this, but it isn't currently wired into this system's data feed. Closing this gap is a **data-integration task** (get that facility sensor's readings into the pipeline), not a model-architecture one — once available, adding it as a 7th zone follows the same recipe as Zone 6. We deliberately did not fabricate a zone without a real sensor backing it.

## Multi-Engine Support

Every engine-specific physics constant (compressor map, VE curve, turbine map, torque curve, zone topology) lives in a swappable `EngineProfile` (`backend/engine_profiles/`) instead of being hardcoded — see `engine_profiles/base.py` for the schema. Switch engines with the `LEAKSENSE_ENGINE` env var (default `c18`); each engine gets its own trained artifacts under `models/<engine>/`.

**Two engines currently calibrated, both from real Caterpillar spec sheets:**

| | Cat C18 | Cat C9.3B |
|---|---|---|
| Displacement | 18.1 L | 9.3 L |
| Aspiration | Twin-turbo (DITA) | Single-turbo (TA) |
| Compression ratio | 16.3:1 | 17.0:1 |
| Max torque | 3655 N·m @ 1400 rpm | 2088 N·m @ 1400 rpm |
| Real torque curve | 1100-2100 rpm (11 points) | 600-2200 rpm (17 points) |
| Test accuracy | 94.9% | 94.5% |
| Macro-F1 | 0.881 | 0.872 |
| Leak-detection F1 | 0.886 | 0.876 |

These two were deliberately chosen to be structurally different (displacement, cylinder layout context, single vs. twin turbo, different equipment segment — C18 is general industrial/construction/mining, C9.3B is compact construction/ag/genset) rather than two sizes of the same thing, so the comparable accuracy across both is real evidence the pipeline generalizes — not just an architectural claim.

**What's real vs. estimated per profile**: geometry, compression ratio, and the full torque curve are transcribed verbatim from Caterpillar's published spec sheets for both engines. The compressor map, VE curve, and turbine map coefficients are engineering estimates (no public compressor map exists for either engine) — for C9.3B these are derived by scaling the C18 reference coefficients by the displacement ratio (9.3/18.1). See each profile's `notes` field (`engine_profiles/c18.json`, `c9_3b.json`) for the full provenance.

**To add a third engine**: get its real Caterpillar spec sheet (torque curve, displacement, compression ratio, aspiration type at minimum), write a new `engine_profiles/<name>.json` following the existing two as templates, then run `LEAKSENSE_ENGINE=<name> python ml/train.py` — it generates a calibrated dataset and trains a dedicated model automatically.

**Known gap**: the frontend (`frontend/src/App.jsx`, `index.html`) still has "Cat C18" hardcoded in static UI text/branding — it doesn't yet read the active engine from `/api/health` and display it dynamically. Cosmetic, not a backend limitation, but worth fixing before a demo that switches engines live.

## Setup & Running

### Requirements
- Python 3.9+
- Node.js 18+

### Quick Start
Just run the provided `run.bat` file from the root directory:
```cmd
run.bat
```
This will automatically:
1. Start the FastAPI backend on port 8000.
2. Install frontend dependencies and start the Vite dev server on port 5173.
3. Open your browser to the dashboard.

### Manual Setup
**Backend:**
```bash
cd backend
pip install -r requirements.txt
python ml/train.py    # trains the active engine (LEAKSENSE_ENGINE, default c18) — first time only
python main.py
```

**Switching engines** (defaults to `c18` if unset):
```bash
LEAKSENSE_ENGINE=c9_3b python ml/train.py   # train the other calibrated engine
LEAKSENSE_ENGINE=c9_3b python main.py       # serve it
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

## How to Test
Once the dashboard is open:
1. Click the **"Start Live Demo"** button on the top to start a simulated sensor stream. The demo will automatically cycle through healthy states and inject different types of leaks.
2. You can manually inject specific leaks using the dropdowns and the **"Inject & Predict"** button.
3. **Interactive Engine Diagram & Level Meters**: In the **Engine Diagram** page, you can drag the sliders or click on the Zone map to instantly simulate leaks. The digital twin AI recognizes the following specific sensor threshold patterns:

   * **🔴 Zone 1 (Intake Leak)**: Drop MAF to ~650, Drop Boost Pressure to ~180, Raise Charge Air Temp to ~47
   * **🔴 Zone 2 (Charge Air Leak)**: Drop Boost Pressure significantly to ~160, Raise Charge Air Temp to ~55
   * **🔴 Zone 3 (CAC/Manifold Leak)**: Drop Boost Pressure to ~150, Raise Exhaust Temp significantly to ~620, Raise Charge Air Temp to ~52
   * **🔴 Zone 4 (Exhaust Manifold Leak)**: Drop Exhaust Temp significantly to ~420, Drop Boost Pressure slightly to ~190
   * **🔴 Zone 5 (Aftertreatment / DPF Leak)**: Drop DPF Delta P to ~0.1, Raise Exhaust Temp slightly to ~580
   * **🔴 Zone 6 (Turbine Outlet / Pre-DPF Leak)**: Drop DPF Inlet Temp noticeably, Drop DPF Delta P moderately

## Repository Structure
- `backend/` - FastAPI server, PyTorch models, Digital Twins, and Data Generator
  - `backend/main.py` - Main API and WebSocket server
  - `backend/engine_profiles/` - `EngineProfile` schema + per-engine calibration (`c18.json`, `c9_3b.json`)
  - `backend/ml/` - LeakNet model, energy field detector, training pipeline, and model benchmark
  - `backend/twins/` - Physics-based digital twins (Intake, Charge Air, Exhaust) — profile-driven, engine-agnostic
  - `backend/data_generator.py` - Synthetic dataset generator, profile-driven
  - `backend/models/<engine>/` - Trained artifacts, one directory per calibrated engine
- `frontend/` - React dashboard
  - `frontend/src/App.jsx` - Main React component with real-time UI
  - `frontend/src/index.css` - Premium glassmorphic styling
- `run.bat` - Application launcher
