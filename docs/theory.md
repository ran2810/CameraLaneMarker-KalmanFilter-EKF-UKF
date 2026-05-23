# Theory Notes — KF / EKF / UKF for Lane Marker Estimation

## 1. State Space Model

Each lane marker (left, right) is tracked independently with state:

```
x = [C0, C1, C2, C3]ᵀ
```

| State | Physical meaning | Unit |
|---|---|---|
| C0 | Lateral offset to lane marking (ego frame) | m |
| C1 | Heading angle between vehicle and lane | rad |
| C2 | Lane curvature | 1/m |
| C3 | Rate of change of curvature | 1/m² |

Sign convention: positive C0 = marking is to the **left** of vehicle centreline.

---

## 2. Process Model

### 2a. Linear (used by KF)

Assumes constant curvature rate and **small** heading angle (sin C1 ≈ C1):

```
x_{k+1} = F(dt) · x_k + w_k,   w_k ~ N(0, Q)

F(dt) = [[1,  dt,  dt²/2,  dt³/6],
         [0,   1,     dt,  dt²/2],
         [0,   0,      1,     dt],
         [0,   0,      0,      1]]
```

Derived from the clothoid (Euler spiral) road model:
- Roads are designed with linear curvature variation (C3 = const)
- This gives a third-order Taylor expansion for C0

### 2b. Nonlinear (used by EKF & UKF)

Adds explicit ego-motion compensation using vehicle speed `v` and yaw rate `ω`:

```
f₁ = C0 − v·dt·sin(C1)
f₂ = C1 + C2·v·dt − ω·dt
f₃ = C2 + C3·v·dt
f₄ = C3
```

**Physical interpretation:**

- `f₁`: As the vehicle moves forward at speed `v` with heading `C1` relative to the
  lane, the lateral distance to the marking decreases by `v·sin(C1)·dt`.  This is
  the key nonlinear term — the linearisation (KF) replaces sin(C1) ≈ C1.

- `f₂`: The relative heading changes because:
  - The lane bends at rate C2 (over arc-length v·dt)
  - The vehicle turns at rate ω (from IMU/odometry)

- `f₃–f₄`: Kinematic curvature propagation.

---

## 3. Measurement Model

The camera reports C0 measurements (OpenCV `fitLine` or ADAS polynomial fit):

```
z_k = H · x_k + v_k,   v_k ~ N(0, R)

H = [1, 0, 0, 0]
R = σ²_camera   (scalar, ~0.0025 m²  for a well-calibrated forward camera)
```

For a stereo camera or when C1 is also reported, H can be extended to [2×4].

---

## 4. Kalman Filter Equations

### Predict
```
x̂⁻ = F · x̂         (or f(x̂, u) for EKF/UKF)
P⁻  = F · P · Fᵀ + Q
```

### Update
```
y  = z − H · x̂⁻      (innovation)
S  = H · P⁻ · Hᵀ + R  (innovation covariance)
K  = P⁻ · Hᵀ · S⁻¹   (Kalman gain)
x̂  = x̂⁻ + K · y
P  = (I − KH) P⁻ (I − KH)ᵀ + KRKᵀ   ← Joseph form (numerically stable)
```

**Why Joseph form?**  The standard `P = (I-KH)P⁻` is algebraically equivalent
but loses symmetry due to floating-point rounding.  The Joseph form guarantees
P remains symmetric and positive-semidefinite.

---

## 5. EKF — First-Order Linearisation

Replace `F` in predict with the Jacobian `∂f/∂x` evaluated at x̂:

```
F_jac = [[1, −v·dt·cos(C1), 0,      0   ],
         [0,      1,        v·dt,   0   ],
         [0,      0,         1,    v·dt ],
         [0,      0,         0,     1   ]]
```

The EKF approximation error is O(dt³·(C1 curvature terms)).  For typical
highway speeds and small heading errors this is negligible, but during
aggressive lane changes or tight interchange curves (|C2| > 0.005 1/m)
the UKF provides a better approximation.

---

## 6. UKF — Sigma-Point Transform

### Sigma points (Van der Merwe scaled, α=1e-3, β=2, κ=0)

```
λ = α²(n + κ) − n     (n = 4 for lane tracking)

X₀     = x̂
Xᵢ     = x̂ + (√((n+λ)P))ᵢ    i = 1…n
Xᵢ₊ₙ   = x̂ − (√((n+λ)P))ᵢ    i = 1…n
```

### Weights
```
Wₘ⁰ = λ/(n+λ)
Wᶜ⁰ = λ/(n+λ) + (1 − α² + β)
Wₘⁱ = Wᶜⁱ = 1/(2(n+λ))     i = 1…2n
```

### Predicted mean and covariance
```
x̂⁻ = Σᵢ Wₘⁱ f(Xᵢ, u, dt)
P⁻  = Σᵢ Wᶜⁱ (f(Xᵢ)−x̂⁻)(f(Xᵢ)−x̂⁻)ᵀ + Q
```

The UKF is accurate to the **3rd order** for Gaussian distributions, vs 1st
order for EKF.  It also never requires Jacobian derivation — useful when f
is provided as a black box.

---

## 7. Process Noise Tuning (Q)

```
Q = diag(q_C0, q_C1, q_C2, q_C3)
```

| Parameter | Meaning | Typical (highway) |
|---|---|---|
| q_C0 | Lateral noise (road surface, vibration) | 0.001–0.0025 m² |
| q_C1 | Heading noise (lane curvature, ego-motion error) | 1e-5–1e-4 rad² |
| q_C2 | Curvature modelling error | 1e-9–1e-8 1/m² |
| q_C3 | Curvature-rate modelling error | 1e-12–1e-11 |

**Rule of thumb**: start by setting Q so that 3σ covers the worst expected
model deviation over one sample interval.  Then tune with the NIS check.

---

## 8. Filter Consistency — NIS Test

The **Normalised Innovation Squared** should follow χ²(m) if the filter
is statistically consistent (i.e., noise matrices match reality):

```
NIS_k = yₖᵀ Sₖ⁻¹ yₖ  ~  χ²(m)
```

- E[NIS] = m (observation dimension)
- NIS >> m  → filter is **overconfident** (Q or R too small)
- NIS << m  → filter is **underconfident** (Q or R too large)

Plot the time-averaged NIS and check it lies within the 95% χ²(m) band.

---

## 9. Differences: KF vs EKF vs UKF (Summary)

| Property | KF | EKF | UKF |
|---|---|---|---|
| Model | Linear F·x | Linearised at x̂ | Sigma-point propagation |
| Jacobian required | No | Yes | No |
| Accuracy | O(dt) | O(dt²) | O(dt³) |
| Cost per step | O(n²m) | O(n²m) + Jacobian | O(n²) × (2n+1) sigma evals |
| Handles sin/cos | Only for small angles | Yes (1st order) | Yes (3rd order) |
| Recommended for | Straight highway | Moderate curves | Tight curves, transitions |
