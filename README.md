# Kalman Filters from Scratch — Lane Marker Estimation

Implementations of **KF**, **EKF**, and **UKF** built entirely from scratch — no `filterpy`, no `pykalman`.  
Applied to **camera-based lane marker estimation** (C0 lateral offset, left + right) on a German Autobahn highway transition scenario.

Later phase: C implementation using hand-rolled matrix utilities.

---

## Why This Exists

Most tutorials either use high-level filter libraries (hiding the math) or toy examples (hiding the engineering).  
This repo shows the full chain: **math → code → real-world sensor model → comparison**.

---

## Application: Lane Marker C0 Estimation

A front-facing camera detects lane markings and reports polynomial coefficients per marker:

| Coefficient | Symbol | Meaning |
|---|---|---|
| Lateral offset | **C0** | Distance (m) from vehicle center to lane marking |
| Heading angle | **C1** | Angle (rad) between vehicle heading and lane |
| Curvature | **C2** | Lane curvature (1/m) |
| Curvature rate | **C3** | Rate of change of curvature (1/m²) |

We track **left** and **right** markers independently.  
The camera provides noisy C0 measurements; the filter estimates the full state `[C0, C1, C2, C3]`.

### Scenario: A9 → A8 Autobahn Interchange (Munich region)

```
t=0–30s    Straight A9 Autobahn          ~120 km/h, lane width 3.75m
t=30–50s   Curve into Autobahndreieck    curvature builds to 0.003 1/m
t=50–70s   Transition road               lane narrows (3.75→3.5m), ego-drift
t=70–90s   Joined A8, straight           stabilisation
t=90–120s  Mild S-curve on A8            curvature ±0.002 1/m
```

---

## Filter Models

### KF — Linear Clothoid Model

State `x = [C0, C1, C2, C3]ᵀ`. Pure kinematic propagation (no ego-motion compensation):

```
F(dt) = [[1,  dt,  dt²/2,  dt³/6],
         [0,   1,     dt,  dt²/2],
         [0,   0,      1,     dt],
         [0,   0,      0,      1]]
```

Measurement: camera reports C0 → `H = [1, 0, 0, 0]`

### EKF — Nonlinear Ego-Motion Compensation

Nonlinear state transition using vehicle speed `v` and yaw rate `ω`:

```
C0'  = C0 - v·dt·sin(C1)          ← nonlinear term
C1'  = C1 + C2·v·dt - ω·dt
C2'  = C2 + C3·v·dt
C3'  = C3
```

Jacobian `F_jac` computed analytically; linearised at current estimate.

### UKF — Sigma-Point Propagation

Same nonlinear `f(·)` as EKF. Uses Van der Merwe scaled sigma points (2n+1 = 9 points) — **no Jacobian needed**.  
Better handles the sin(C1) nonlinearity during high-curvature transitions.

---

## Repository Structure

```
kalman-filters/
├── python/
│   ├── filters/
│   │   ├── base.py          # Abstract BaseFilter
│   │   ├── kf.py            # Linear Kalman Filter
│   │   ├── ekf.py           # Extended Kalman Filter
│   │   └── ukf.py           # Unscented Kalman Filter
│   ├── models/
│   │   └── lane_model.py    # State/noise matrices, transition functions, Jacobians
│   ├── data/
│   │   └── generate_lane_data.py   # Synthetic A9 -> A8 scenario generator
│   ├── utils/
│   │   ├── visualization.py # Plotting helpers
│   │   └── metrics.py       # RMSE, MAE, NIS, NEES
│   └── examples/
│       ├── lane_estimation_kf.py
│       ├── lane_estimation_ekf.py
│       ├── lane_estimation_ukf.py
│       └── compare_filters.py      # ← start here
├── c/                        # Phase 2 (coming)
│   ├── include/
│   │   ├── matrix.h
│   │   ├── kf.h
│   │   ├── ekf.h
│   │   └── ukf.h
│   └── src/
│       ├── matrix.c          # All linear algebra utilities
│       ├── kf.c
│       ├── ekf.c
│       └── ukf.c
└── docs/
    └── theory.md             # Derivations and tuning notes
```

---

## Quick Start

```bash
pip install -r requirements.txt
cd python/examples
python compare_filters.py
```

This generates the synthetic highway dataset, runs all three filters on the same noisy measurements, and produces comparison plots + RMSE table.

---

## Dependencies

```
numpy
matplotlib
scipy        # for chi2 NIS/NEES consistency checks only
```

No `filterpy`. No `pykalman`. Every predict/update loop is written by hand.

---

## Tuning Guide

| Parameter | Symbol | Typical value | Effect |
|---|---|---|---|
| Process noise C0 | q_C0 | 0.01–0.1 m² | Tracks fast lateral changes |
| Process noise C1 | q_C1 | 0.001–0.01 rad² | Tracks heading changes |
| Process noise C2 | q_C2 | 1e-5–1e-4 | Tracks curvature changes |
| Camera noise | R | 0.0025–0.01 m² | Reflects camera detection accuracy |

Higher Q → trusts measurements more (responsive but noisy).  
Higher R → trusts model more (smooth but laggy).

---

## Coming: C Implementation

Phase 2 adds a C port using hand-rolled matrix operations in `matrix.c`:
- `mat_mul`, `mat_add`, `mat_transpose`, `mat_inv` (via LU decomposition)
- Same filter algorithms, stack-allocated fixed-size matrices
- No heap allocation in the hot path
- Makefile + unit tests
