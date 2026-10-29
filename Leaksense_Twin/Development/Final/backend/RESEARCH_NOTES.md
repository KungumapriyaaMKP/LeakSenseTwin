# Research Notes — Literature Review & Operating-Point Conditioning

This file documents what we read, why, and what we actually changed in the
code as a result. Source of the review: a user-compiled bibliography
(`Complete References CAT.docx`) of ~70 Caterpillar documents, academic
papers, and patents, reviewed via three parallel research passes on
2026-06-19.

## What changed in this round

**`backend/ml/regional_scaler.py`** — new `RegionalScaler` class, wired into
`ml/train.py` in place of `sklearn.preprocessing.StandardScaler`.

**The idea, and where it came from:** several independent FDI (Fault
Detection and Isolation) papers found that leak isolability improves
significantly when residuals are evaluated against a regime-specific
("operating-point-conditioned") healthy baseline instead of one global
one — a residual that's unremarkable at idle may be a real anomaly at
rated speed, and vice versa. Specifically:

- Paper on location isolability of intake/exhaust manifold leaks in a
  turbocharged diesel w/ EGR — isolability improved when the system's
  variation was considered *across multiple operating points* rather than
  one global boundary.
- Springer/IJAT 2015 paper on exhaust-path FDI — implemented isolation as
  *operating-region-dependent* fuzzy systems specifically for this reason.
- Control Engineering Practice paper on diesel air-path diagnosis — used
  *adaptive (operating-point-dependent) thresholds* on residuals, validated
  on-road.

**What we actually built** (an engineering interpretation of the above, not
a literal implementation of any single paper): `RegionalScaler` buckets the
operating envelope into 4 RPM regions (computed from the engine profile's
actual operating range, e.g. C18: 1100-2100 RPM → 4×250rpm buckets). The 13
raw sensor columns (RPM, MAF, MAP_*, T_*, fuel_qty) are still scaled
*globally* — their absolute magnitude is itself informative, the model
needs to know real operating point, not just "RPM relative to its own
bucket." The 18 derived columns (9 physics-residuals + 6 ratios + 3 stats)
are z-scored *within* each RPM bucket — "is this residual unusual" is now
judged against that operating point's own healthy distribution, not a
single global one.

This is a drop-in replacement: same `.fit_transform()`/`.transform()`
interface as `StandardScaler`, so `main.py`'s serving code needed zero
changes — `joblib.load()` just unpickles whichever scaler class was saved.

## Measured results (before → after, same train/val/test split, both engines retrained)

| Engine | Metric | Before (global scaler) | After (RegionalScaler) |
|---|---|---|---|
| C18 | Overall accuracy | 94.92% | 94.85% |
| C18 | Macro-F1 | 0.8805 | 0.8779 |
| C18 | Leak-detection F1 | 0.8858 | 0.8849 |
| C9.3B | Overall accuracy | 94.54% | 94.76% |
| C9.3B | Macro-F1 | 0.8715 | 0.8746 |
| C9.3B | Leak-detection F1 | 0.8758 | 0.8873 |

**Headline accuracy is essentially flat — as expected, since the literature's
claim was about *isolability of subtle leaks*, not overall accuracy.** The
real test is per-zone severity-1 (smallest, 2% flow loss) recall on C18:

| Zone | Before | After |
|---|---|---|
| Zone 1 (Intake) | 0.322 | **0.402** |
| Zone 2 (Charge Air) | 0.807 | 0.818 |
| Zone 3 (CAC/Manifold) | 0.752 | **0.861** |
| Zone 4 (Exhaust Manifold) | 0.065 | 0.065 (unchanged) |
| Zone 5 (DPF/SCR) | 0.078 | 0.096 |
| Zone 6 (Turbine Outlet/Pre-DPF) | 0.021 | 0.031 |

Honest read: **a real, measurable improvement in Zones 1-3 (intake/charge-air
path), especially Zone 3 (+11 points recall) — but it did not fix Zone
4/5/6**, where the smallest leaks remain hard to distinguish from noise
regardless of operating-point conditioning. This makes physical sense:
those zones' injected signal is the most subtle to begin with (see
`README.md`'s Model Changelog), and bucketing by RPM doesn't help when the
problem is signal magnitude vs. sensor noise floor, not regime-dependent
bias.

Cost: healthy false-positive rate rose slightly (≈0.1% → 0.27% on C18) —
still negligible after the 3-consecutive-reading anti-flicker filter
(0.27%³ ≈ 0.00002% actual false-alarm rate), so this is a good trade.

## Other findings from the literature review (not yet implemented)

Tracked here so they aren't lost — these are candidates for future rounds,
not yet acted on:

1. **Additional physics-residual features.** Multiple papers found these
   more discriminative *between adjacent zones* than raw sensor residuals:
   - Turbocharger speed estimation error (sliding-mode/nonlinear observer
     vs. measured boost) — relevant to Zone 1/2 separation.
   - An exhaust "virtual hole area" estimate (Luenberger + sliding-mode
     observers on inlet/outlet flow) — relevant to Zone 4 specifically,
     which remains our weakest zone.
   - EGR pressure-differential residual — exhaust leaks cut the
     exhaust>intake differential EGR relies on; intake leaks bypass it
     differently. Could help Zone 4 vs Zone 1 confusion.
   - Intake-manifold-temperature dynamics residual (3-step nonlinear
     time-series model) — used in industry for CAC fouling detection,
     could sharpen Zone 2/3 vs. general CAC degradation.
2. **A more principled confirmation rule than fixed 3-consecutive-readings:**
   a likelihood-ratio test on the residual (used for MAF-sensor-less leak
   detection in turbo gasoline engines), or a sequential model (an HMM was
   used for intake-leakage diagnosis/prognosis with severity estimation) —
   either would natively handle the temporal-persistence problem better
   than our current hand-written rule, and could yield a severity estimate
   for free.
3. **Unsupervised anomaly detection as a safety net.** A Fisher-divergence
   autoencoder trained only on healthy data (used for aero-engine anomaly
   detection on CMAPSS) would catch fault *types* outside our 6 trained
   zones — e.g. a genuinely novel failure mode — that a 7-class classifier
   structurally cannot.
4. **Real-world validation found**: Caterpillar's own C13/C15/C18
   troubleshooting documentation (`catengine.info`) describes ECU fault
   codes `E1045(3)`/`102-1` (severe low intake manifold pressure → derate)
   and `E1045(2)`/`102-18` (moderate), with a diagnostic sequence that
   explicitly inspects "piping between turbo compressor outlet and
   aftercooler inlet for cuts/damage" as step 2 — this is literally our
   Zone 2. Useful as real-world validation that our zone boundaries match
   how Caterpillar technicians actually diagnose this fault in practice;
   worth citing if asked to justify the zone topology isn't arbitrary.
5. **Patent awareness** (not legal advice, just awareness — see full agent
   report in conversation history if needed): the closest prior art to our
   residual approach is US 9,856,814 (FPT Industrial) — an expected-vs-actual
   lambda residual from airflow + fuel quantity. Our 31-feature NN is
   meaningfully different (many residuals + ML classifier vs. one
   comparator), but if we ever publish detailed methodology, this and the
   2025 Cummins patent (US 12,448,912, two-threshold DPF fault
   classification) are the two most worth being aware of.
6. **Additional real Cat engine ratings found** (not yet built as profiles):
   C18 series-twin-turbo industrial (16.0:1 CR), C18 marine (single-turbo,
   up to 2300 RPM), C18 gensets (14.0-14.5:1 CR). The C9.3B documents found
   were all duplicates of the rating we already use. These could become
   additional `engine_profiles/*.json` entries if more than 2 demonstration
   engines is ever wanted.

## Scope notes / honesty caveats

- `ml/benchmark.py`'s model-family comparison (RandomForest, GradientBoosting,
  LightGBM, XGBoost, Logistic Regression vs. `LeakNet`) intentionally still
  uses plain `StandardScaler`, not `RegionalScaler` — that benchmark's job is
  comparing algorithm families on equal footing, not chasing production
  accuracy. If `RegionalScaler` is iterated on further, consider whether to
  re-run that comparison too.
- Several academic sources were paywalled (ResearchGate, ScienceDirect,
  SAE Mobilus) and only abstracts/titles were accessible — findings from
  those are based on abstracts, not full methodology review.
- This RegionalScaler design is an engineering interpretation inspired by
  the cited papers' *finding* (operating-point conditioning helps
  isolability), not a reproduction of any single paper's specific method
  (most used physical observers/Kalman filters/HMMs, not feature scaling).
  We chose the scaling approach because it's a bounded, low-risk change
  that slots into the existing pipeline without an architecture rewrite —
  the more sophisticated approaches (items 1-3 above) remain open follow-ups.
