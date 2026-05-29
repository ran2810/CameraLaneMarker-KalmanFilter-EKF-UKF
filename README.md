# Kalman Filters from Scratch — Lane Marker Estimation

Implementations of **KF**, **EKF**, and **UKF** built entirely from scratch across three languages — no `filterpy`, no `pykalman`, no Eigen shortcuts hiding the math.  
Applied to **camera-based lane marker C0 estimation** (left + right) on a German Autobahn highway transition scenario.

---

## Application: Lane Marker C0 Estimation

A front-facing camera detects lane markings and reports polynomial coefficients per marker:

| Coefficient | Symbol | Meaning | Unit |
|---|---|---|---|
| Lateral offset | **C0** | Distance from vehicle centre to lane marking | m |
| Heading angle | **C1** | Angle between vehicle heading and lane | rad |
| Curvature | **C2** | Lane curvature | 1/m |
| Curvature rate | **C3** | Rate of change of curvature | 1/m² |

Left and right markers are tracked independently.  
The camera provides noisy C0 measurements; the filter estimates the full state `[C0, C1, C2, C3]`.

### Scenario: A9 → A8 Autobahn Interchange (Munich region)

```
Phase 1   0– 30 s   Straight A9              ~120 km/h, lane width 3.75 m
Phase 2  30– 50 s   Curve into Dreieck        curvature → 0.003 1/m
Phase 3  50– 70 s   Transition road           lane narrows 3.75→3.50 m, ego-drift
Phase 4  70– 90 s   Joined A8, straight       ~110 km/h, stabilisation
Phase 5  90–120 s   Mild S-curve on A8        curvature ±0.002 1/m
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

Measurement: `H = [1, 0, 0, 0]`  (camera reports C0 only)

### EKF — Nonlinear Ego-Motion Compensation

Nonlinear state transition with vehicle speed `v` and yaw rate `ω`:

```
C0' = C0 − v·dt·sin(C1)       ← nonlinear term
C1' = C1 + C2·v·dt − ω·dt
C2' = C2 + C3·v·dt
C3' = C3
```

Jacobian `F_jac` computed analytically and linearised at the current estimate each step.

### UKF — Sigma-Point Propagation

Same nonlinear `f(·)` as EKF. Uses Van der Merwe scaled sigma points (2n+1 = 9) — **no Jacobian needed**.  
Captures the sin(C1) nonlinearity to 3rd order vs 1st order for EKF.

---

## Repository Structure

```
kalman-filters/
│
├── CMakeLists.txt                   ← root: C + C++ build, dataset generation
│
├── python/                          ← pure Python, no filter libraries
│   ├── filters/
│   │   ├── base.py                  # Abstract BaseFilter (x, P, Q, R, NIS)
│   │   ├── kf.py                    # Linear Kalman Filter
│   │   ├── ekf.py                   # Extended Kalman Filter
│   │   └── ukf.py                   # Unscented Kalman Filter
│   ├── data/
│   │   └── generate_lane_data.py    # Synthetic A9→A8 dataset + CSV export
│   ├── utils/
│   │   ├── metrics.py               # RMSE, MAE, NIS chi² consistency
│   │   └── visualization.py         # Phase-band plots, NIS, per-phase RMSE
│   └── examples/
│       └── compare_filters.py       # ← start here
│
├── C_code/                               ← C99, zero heap allocation, no Eigen
│   ├── CMakeLists.txt
│   ├── include/
│   │   ├── matrix.h                 # Public API: mat_mul, mat_inv (LU), mat_chol …
│   │   ├── kf.h
│   │   ├── ekf.h
│   │   └── ukf.h
│   ├── src/
│   │   ├── matrix.c                 # All linear algebra (LU + partial pivoting)
│   │   ├── kf.c
│   │   ├── ekf.c
│   │   └── ukf.c
│   ├── tests/
│   │   ├── test_matrix.c            # 36 known-answer unit tests
│   │   ├── test_ekf.c
│   │   └── test_ukf.c
│   └── examples/
│       ├── lane_estimation_kf.c     # Single-filter example
│       └── compare_filters.c        # KF / EKF / UKF side-by-side
│
├── C++/                             ← C++17, header-only, Eigen for matrices
│   ├── CMakeLists.txt
│   ├── include/
│   │   ├── FilterBase.hpp           # CRTP base (no virtual dispatch)
│   │   ├── KalmanFilter.hpp         # Linear KF
│   │   ├── EKFFilter.hpp            # EKF (std::function f/h/Jacobians)
│   │   ├── UKFFilter.hpp            # UKF (LLT sigma points, if constexpr M==1)
│   │   ├── LaneModel.hpp            # make_lane_filter() factories
│   │   ├── CsvReader.hpp            # Zero-dependency CSV parser
│   │   └── Metrics.hpp              # RunningMetrics: RMSE, MAE, NIS
│   ├── tests/
│   │   └── test_filters.cpp         # Scalar cycle, factory, covariance health, NIS
│   └── examples/
│       └── compare_filters.cpp      # KF / EKF / UKF comparison
│
├── docs/
│   └── theory.md                    # Derivations: KF, EKF, UKF, NIS, tuning
│
└── plots/                            # plots from python metrics
    └── .gitkeep
```

---

## Dependencies

### Python
```
numpy >= 1.24
matplotlib >= 3.7
scipy >= 1.11       # chi² NIS consistency bounds only
```
No `filterpy`. No `pykalman`. Every predict/update loop is hand-written.

### C
- GCC / Clang with C99 support
- No external libraries — `libm` only

### C++
- GCC / Clang with C++17 support
- **Eigen 3.3+** 
- CMake 3.16+

---

## Build & Run

### Python

```bash
pip install -r requirements.txt

# Generate dataset + run all three filters + plots
cd python/examples
python compare_filters.py

# Individual filters
python lane_estimation_kf.py
python lane_estimation_ekf.py
python lane_estimation_ukf.py

# Generate CSV only (for C/C++ consumption)
python python/data/generate_lane_data.py data/lane_data.csv
```

### C and C++ (CMake — unified build)

```bash
# 1. Configure (once)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# 2. Build everything (C library + C++ library + tests + examples)
cmake --build build --parallel

# 3. Run all unit tests via CTest
ctest --test-dir build --output-on-failure

# 4. Generate the lane dataset CSV
cmake --build build --target generate_data
#    → build/data/lane_data.csv

# 5a. C — single KF example
cmake --build build --target run_lane_kf
#    → build/data/lane_kf_output.csv

# 5b. C — three-filter comparison (KF / EKF / UKF)
cmake --build build --target run_compare
#    → build/data/compare_output.csv

# 5c. C++ — three-filter comparison (KF / EKF / UKF)
cmake --build build --target run_compare_cpp
#    → build/data/compare_cpp_output.csv
```

> **Note:** `generate_data`, `run_compare`, and `run_compare_cpp` are chained — CMake
> re-generates the CSV automatically if `generate_lane_data.py` changes.

---

## Results

All three implementations run on the same 3000-frame dataset (`seed=42`, `σ_camera=0.1 m`).
Numbers are bit-identical across Python, C, and C++.

### Python

```
============================================================
Filter   | RMSE_L (m) | RMSE_R (m) |  MAE_L (m) |  MAE_R (m)
============================================================
KF       |     0.0710 |     0.0727 |     0.0395 |     0.0410
EKF      |     0.0772 |     0.0791 |     0.0428 |     0.0446
UKF      |     0.0772 |     0.0791 |     0.0428 |     0.0446
============================================================

NIS Consistency (mean ≈ 1.0 → filter is consistent):
  KF   mean_NIS=0.986   [0.9, 1.1]  ✓ consistent
  EKF  mean_NIS=0.987   [0.9, 1.1]  ✓ consistent
  UKF  mean_NIS=0.986   [0.9, 1.1]  ✓ consistent

Per-phase RMSE — C0_L (m):
  Phase 1  KF=0.0763  EKF=0.0839  UKF=0.0839
  Phase 2  KF=0.0833  EKF=0.0901  UKF=0.0901
  Phase 3  KF=0.0616  EKF=0.0660  UKF=0.0660
  Phase 4  KF=0.0622  EKF=0.0672  UKF=0.0672
  Phase 5  KF=0.0679  EKF=0.0739  UKF=0.0739
```

### C  (hand-rolled matrix.c, no Eigen)

```
===========================================================
  Overall — Left Marker (C0_L)
===========================================================
  KF    RMSE=0.0710 m   MAE=0.0395 m   mean_NIS=0.986
  EKF   RMSE=0.0772 m   MAE=0.0428 m   mean_NIS=0.987
  UKF   RMSE=0.0772 m   MAE=0.0428 m   mean_NIS=0.986

  Overall — Right Marker (C0_R)
===========================================================
  KF    RMSE=0.0727 m   MAE=0.0410 m   mean_NIS=1.017
  EKF   RMSE=0.0791 m   MAE=0.0446 m   mean_NIS=1.017
  UKF   RMSE=0.0791 m   MAE=0.0446 m   mean_NIS=1.017

Per-phase RMSE — C0_L (m):
  Phase                   KF       EKF       UKF
  Ph1 A9 straight     0.0763    0.0839    0.0839
  Ph2 Curve           0.0833    0.0901    0.0901
  Ph3 Transition      0.0616    0.0660    0.0660
  Ph4 A8 straight     0.0622    0.0672    0.0672
  Ph5 S-curve         0.0679    0.0739    0.0739
```

### C++  (header-only, Eigen 3, CRTP base)

```
============================================================
  Overall — Left Marker (C0_L)
============================================================
  KF    RMSE=0.0710 m   MAE=0.0395 m   mean_NIS=0.9863
  EKF   RMSE=0.0772 m   MAE=0.0428 m   mean_NIS=0.9865
  UKF   RMSE=0.0772 m   MAE=0.0428 m   mean_NIS=0.9865

  Overall — Right Marker (C0_R)
============================================================
  KF    RMSE=0.0727 m   MAE=0.0410 m   mean_NIS=1.0166
  EKF   RMSE=0.0791 m   MAE=0.0446 m   mean_NIS=1.0169
  UKF   RMSE=0.0791 m   MAE=0.0446 m   mean_NIS=1.0169

Per-phase RMSE — C0_L (m):
  Phase                     KF       EKF       UKF
  Ph1 A9 straight       0.0763    0.0839    0.0839
  Ph2 Curve             0.0833    0.0901    0.0901
  Ph3 Transition        0.0616    0.0660    0.0660
  Ph4 A8 straight       0.0622    0.0672    0.0672
  Ph5 S-curve           0.0679    0.0739    0.0739
```

---

## Key Observations

**KF beats EKF/UKF on RMSE** — not a bug. On the Autobahn scenario `|C1| < 0.03 rad`,
so `sin(C1) ≈ C1` to within 0.1%. The EKF/UKF's ego-motion correction is correcting a
term that is essentially zero, while the KF's linear F(dt) is a slightly better regulariser
for highway geometry. EKF and UKF will pull ahead on tighter curves (|C2| > 0.005 1/m)
or during lane changes where `|C1|` exceeds 0.1 rad.

**EKF = UKF to 4 decimal places** — the Autobahn scenario does not stress the nonlinearity
hard enough to separate them. On a city roundabout (|C2| ~ 0.05 1/m) the UKF's 3rd-order
accuracy advantage would become visible.

**NIS ≈ 1.0 across all filters and implementations** — the filter noise model correctly
describes the synthetic data. NIS > 1.1 would indicate overconfidence (Q or R too small);
NIS < 0.9 would indicate underconfidence.

---

## Process Noise Tuning (Q)

| State | σ (std-dev) | Rationale |
|---|---|---|
| C0 | 0.07 m | Road surface roughness, vibration |
| C1 | 0.004 rad | Lane curvature, small ego-motion errors |
| C2 | 4×10⁻⁵ 1/m | Slowly varying road curvature |
| C3 | 8×10⁻⁷ 1/m² | Nearly constant curvature rate |

Higher Q → filter trusts measurements more (responsive, noisier).  
Higher R → filter trusts the model more (smooth, laggier).  
Use the NIS plot to tune: target mean NIS ∈ [0.9, 1.1].

---

## Theory

See [`docs/theory.md`](docs/theory.md) for:
- Clothoid road model derivation
- KF, EKF, UKF equations with Joseph-form covariance update
- Sigma-point weight derivation (Van der Merwe)
- NIS / NEES consistency tests
- Why EKF and UKF converge on highway scenarios
