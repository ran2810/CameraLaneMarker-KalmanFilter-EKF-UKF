"""
kml_lane_data.py — Generate lane marker dataset from a GPS route KML file.

Pipeline
--------
1.  Parse KML — extract LineString coordinates and Point landmarks.
2.  Project WGS84 → UTM 32N (EPSG:25832).
3.  Clean: remove duplicate nodes, outlier spikes (GPS noise).
4.  Fit cubic spline, re-sample at uniform camera frame rate.
5.  Compute κ(s), κ'(s), ψ(s) from spline derivatives.
6.  Speed profile: urban default 40 km/h; drops to 20 km/h on tight curves.
7.  Lane-change detection: segments where |κ| > threshold and heading
    reverses ±sign — these are the turning / lane-change manoeuvres that
    stress the nonlinear filter.
8.  Camera measurements: z = [C0, C1] with Gaussian noise + dropouts.
    C1 heading transients are physically modelled during detected manoeuvres.

Usage
-----
    # Python API
    from python.data.kml_lane_data import generate_kml_lane_data
    ds = generate_kml_lane_data("data/route.kml")
    ds.save_csv("data/kml_lane_data.csv")

    # CLI
    python python/data/kml_lane_data.py route.kml
    python python/data/kml_lane_data.py route.kml --out data/out.csv --dt 0.04

     .\python\data\kml_lane_data.py "H:\GitHub\KalmanFilter\python\data\munich.kml" --out data/out.csv    
"""

from __future__ import annotations
import sys as _sys
import argparse
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from pyproj import Transformer
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

import os as _os

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

# =========================================================================
#  coordinate transforms
# =========================================================================

_WGS84_TO_UTM32 = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)
_UTM32_TO_WGS84 = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True)
_WGS84_TO_WEBM  = Transformer.from_crs("EPSG:4326", "EPSG:3857",  always_xy=True)

#  KML namespace 
_KML_NS = {"k": "http://www.opengis.net/kml/2.2"}


# #  road geometry constants  Munich standard) - urban 
# LANE_WIDTH_URBAN    = 3.25   # m  — standard Munich urban lane
# LANE_WIDTH_ARTERIAL = 3.50   # m  — larger arterial / Bundesstrasse
# SPEED_DEFAULT_KMH   = 50.0   # km/h
# SPEED_CURVE_KMH     = 20.0   # km/h on κ > KAPPA_SLOW
# KAPPA_SLOW          = 0.05   # 1/m — threshold to slow down (~R<20m)

# Auto-detected from curvature profile; overridable per call.
LANE_WIDTH_HIGHWAY  = 3.75   # m  — Autobahn/motorway (StVO/RAA standard)
LANE_WIDTH_RAMP     = 3.50   # m  — on/off ramp, motorway_link
LANE_WIDTH_URBAN    = 3.25   # m  — urban / city roads
LANE_WIDTH_ARTERIAL = 3.50   # m  — arterial / Bundesstrasse

# Speed profiles (km/h)
SPEED_HIGHWAY_KMH   = 120.0  # motorway cruise
SPEED_RAMP_KMH      =  80.0  # on/off ramp
SPEED_URBAN_KMH     =  40.0  # urban default
SPEED_CURVE_KMH     =  60.0  # slows on κ > KAPPA_SLOW (ramps)
KAPPA_SLOW          = 0.008  # 1/m — start slowing (~R < 125m)

#  manoeuvre detection 
# Highway: only ramp curves (κ > 0.008 1/m, R < 125m) qualify.
# Urban:   tighter turns from 0.030 1/m.
# Auto-selected based on route type detection.
KAPPA_MANOEUVRE_HIGHWAY = 0.008  # 1/m — ramp / interchange curve
KAPPA_MANOEUVRE_URBAN   = 0.030  # 1/m — urban turn
KAPPA_MANOEUVRE         = 0.008  # default (overridden by route_type)
MIN_MANOEUVRE_M         = 30.0   # highway ramps are longer than urban turns
STEERING_TAU            = 0.40   # s  — highway steering response (slower)
C1_PEAK_MANOEUVRE       = 0.06   # rad — highway lane change (shallower than urban)


# #  lane change / manoeuvre detection  - urban 
# KAPPA_MANOEUVRE     = 0.030  # 1/m — |κ| threshold for a manoeuvre
# MIN_MANOEUVRE_M     = 5.0    # minimum arc-length (m) to count as a manoeuvre
# STEERING_TAU        = 0.35   # s  — heading-error first-order time constant
# C1_PEAK_MANOEUVRE   = 0.12   # rad — peak heading error during a manoeuvre

_CSV_HEADER = (
    "t,speed,yaw_rate,"
    "C0_L_true,C0_R_true,C1_true,C2_true,C3_true,"
    "meas_L,meas_R\n"
)
 
 
@dataclass
class _LaneDataset:
    t:         np.ndarray
    speed:     np.ndarray
    yaw_rate:  np.ndarray
    C0_L_true: np.ndarray
    C0_R_true: np.ndarray
    C1_true:   np.ndarray
    C2_true:   np.ndarray
    C3_true:   np.ndarray
    meas_L:    np.ndarray
    meas_R:    np.ndarray
    dt: float = 0.04
 
    @property
    def T(self) -> int:
        return len(self.t)
 
    def summary(self) -> str:
        dL = int(np.isnan(self.meas_L).sum())
        dR = int(np.isnan(self.meas_R).sum())
        return (f"LaneDataset  T={self.T}  dt={self.dt}s  "
                f"duration={self.t[-1]:.0f}s  dropouts L={dL} R={dR}")
 
    def save_csv(self, path: str) -> None:
        _os.makedirs(_os.path.dirname(_os.path.abspath(path)), exist_ok=True)
        rows = []
        for k in range(self.T):
            mL = self.meas_L[k]
            mR = self.meas_R[k]
            s_mL = "nan" if np.isnan(mL) else f"{mL:.6f}"
            s_mR = "nan" if np.isnan(mR) else f"{mR:.6f}"
            rows.append(
                f"{self.t[k]:.4f},"
                f"{self.speed[k]:.6f},{self.yaw_rate[k]:.8f},"
                f"{self.C0_L_true[k]:.6f},{self.C0_R_true[k]:.6f},"
                f"{self.C1_true[k]:.8f},{self.C2_true[k]:.8f},"
                f"{self.C3_true[k]:.8f},{s_mL},{s_mR}"
            )
        with open(path, "w") as f:
            f.write(_CSV_HEADER)
            f.write("\n".join(rows))
            f.write("\n")
        print(f"Saved {self.T} frames -> {path}")
 
# =========================================================================
#  KML parsing
# =========================================================================
@dataclass
class KmlRoute:
    """Parsed KML route."""
    name:        str
    coords_ll:   np.ndarray     # [N×2] lon, lat (WGS84)
    waypoints:   List[Tuple[float, float, str]]  # [(lon, lat, name), …]


def parse_kml(path: str | Path) -> KmlRoute:
    """
    Parse a KML file and return the route LineString + named Points.

    Handles:
    - Standard KML 2.2 namespace
    - Coordinates with or without altitude field
    - Multiple folders / nested Placemarks (takes first LineString found)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"KML file not found: {path}")

    tree = ET.parse(path)
    root = tree.getroot()

    # Strip namespace for easier querying
    def _tag(el):
        return el.tag.split("}")[-1] if "}" in el.tag else el.tag

    # Find the document name
    name = "unnamed"
    for el in root.iter():
        if _tag(el) == "name" and el.text:
            name = el.text.strip()
            break

    # Extract LineString coordinates
    line_coords = None
    for el in root.iter():
        if _tag(el) == "coordinates" and el.text:
            raw = el.text.strip().split()
            if len(raw) > 10:   # LineString has many points; Points have 1
                pts = []
                for c in raw:
                    parts = [float(x) for x in c.split(",") if x.strip()]
                    if len(parts) >= 2:
                        pts.append([parts[0], parts[1]])   # lon, lat
                if len(pts) > 5:
                    line_coords = np.array(pts)
                    break

    if line_coords is None:
        raise ValueError("No LineString with >5 coordinates found in KML")

    # Extract Point placemarks (start, end, landmarks)
    waypoints = []
    for pm in root.iter():
        if _tag(pm) != "Placemark":
            continue
        pt_el = None
        for child in pm.iter():
            if _tag(child) == "Point":
                pt_el = child
                break
        if pt_el is None:
            continue
        coord_el = None
        for child in pt_el.iter():
            if _tag(child) == "coordinates":
                coord_el = child
                break
        if coord_el is None or not coord_el.text:
            continue
        parts = [float(x) for x in coord_el.text.strip().split(",") if x.strip()]
        if len(parts) < 2:
            continue
        wname_el = pm.find(".//{http://www.opengis.net/kml/2.2}name")
        if wname_el is None:
            for child in pm.iter():
                if _tag(child) == "name":
                    wname_el = child
                    break
        wname = wname_el.text.strip() if (wname_el is not None and wname_el.text) else "waypoint"
        waypoints.append((parts[0], parts[1], wname))

    return KmlRoute(name=name, coords_ll=line_coords, waypoints=waypoints)


# =========================================================================
# Geometry helpers
# =========================================================================

def _to_utm(coords_ll: np.ndarray) -> np.ndarray:
    e, n = _WGS84_TO_UTM32.transform(coords_ll[:, 0], coords_ll[:, 1])
    return np.column_stack([e, n])


def _clean_coords(coords_utm: np.ndarray,
                  max_step_m: float = 200.0,
                  min_step_m: float = 0.3) -> np.ndarray:
    """
    Remove:
    - Duplicate nodes (step < min_step_m)
    - GPS spike outliers (step > max_step_m)
    """
    keep = [0]
    for i in range(1, len(coords_utm)):
        d = np.linalg.norm(coords_utm[i] - coords_utm[keep[-1]])
        if min_step_m <= d <= max_step_m:
            keep.append(i)
    return coords_utm[keep]


def _build_spline(coords_utm: np.ndarray,
                  smooth_sigma: float = 1.5
                  ) -> Tuple[CubicSpline, CubicSpline, np.ndarray]:
    """Fit cubic spline on smoothed UTM centreline."""
    e_sm = gaussian_filter1d(coords_utm[:, 0], sigma=smooth_sigma)
    n_sm = gaussian_filter1d(coords_utm[:, 1], sigma=smooth_sigma)
    ds   = np.sqrt(np.diff(e_sm)**2 + np.diff(n_sm)**2)
    s    = np.concatenate([[0.0], np.cumsum(ds)])
    return CubicSpline(s, e_sm), CubicSpline(s, n_sm), s


def _geometry(cs_e, cs_n, s_eval: np.ndarray,
              smooth_sigma: float = 2.0
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute (e, n, ψ, κ, κ') at arc-length samples."""
    de  = cs_e(s_eval, 1);  dn  = cs_n(s_eval, 1)
    d2e = cs_e(s_eval, 2);  d2n = cs_n(s_eval, 2)
    spd = np.maximum(np.sqrt(de**2 + dn**2), 1e-9)
    psi   = np.arctan2(dn, de)
    kappa = (d2e * dn - d2n * de) / spd**3
    kappa = gaussian_filter1d(kappa, sigma=smooth_sigma)
    kappa_dot = gaussian_filter1d(np.gradient(kappa, s_eval), sigma=smooth_sigma)
    return cs_e(s_eval), cs_n(s_eval), psi, kappa, kappa_dot


# =========================================================================
# Manoeuvre (lane-change / turn) detection
# =========================================================================
@dataclass
class Manoeuvre:
    """One detected turning / lane-change manoeuvre."""
    idx_start:  int         # frame index
    idx_end:    int
    s_start:    float       # arc-length (m)
    s_end:      float
    direction:  str         # "left" (κ>0) or "right" (κ<0)
    kappa_peak: float       # peak |κ| during manoeuvre (1/m)
    label:      str         # human-readable

def detect_manoeuvres(s_eval: np.ndarray,
                      kappa:  np.ndarray,
                      kappa_threshold: float = KAPPA_MANOEUVRE,
                      min_arc_m:       float = MIN_MANOEUVRE_M,
                      ) -> List[Manoeuvre]:
    """
    Segment the route into straight sections and manoeuvres.

    A manoeuvre is a contiguous run of frames where |κ| ≥ kappa_threshold,
    spanning at least min_arc_m metres.  Adjacent runs within 10 m are merged.
    """
    active = np.abs(kappa) >= kappa_threshold
    manoeuvres = []

    # Find contiguous active runs
    i = 0
    while i < len(active):
        if not active[i]:
            i += 1
            continue
        j = i
        while j < len(active) and active[j]:
            j += 1
        arc = s_eval[j - 1] - s_eval[i]
        if arc >= min_arc_m:
            seg_kappa = kappa[i:j]
            peak      = float(np.abs(seg_kappa).max())
            direction = "left" if seg_kappa.mean() > 0 else "right"
            label     = f"Turn {direction}  κ={peak:.3f} 1/m"
            manoeuvres.append(Manoeuvre(
                idx_start  = i,
                idx_end    = j - 1,
                s_start    = float(s_eval[i]),
                s_end      = float(s_eval[j - 1]),
                direction  = direction,
                kappa_peak = peak,
                label      = label,
            ))
        i = j

    # Merge manoeuvres that are within 10 m of each other
    merged = []
    for m in manoeuvres:
        if merged and m.s_start - merged[-1].s_end < 10.0:
            prev = merged.pop()
            peak = max(prev.kappa_peak, m.kappa_peak)
            kappa_mean = (kappa[prev.idx_start:m.idx_end + 1]).mean()
            direction  = "left" if kappa_mean > 0 else "right"
            merged.append(Manoeuvre(
                idx_start  = prev.idx_start,
                idx_end    = m.idx_end,
                s_start    = prev.s_start,
                s_end      = m.s_end,
                direction  = direction,
                kappa_peak = peak,
                label      = f"Turn {direction}  κ={peak:.3f} 1/m",
            ))
        else:
            merged.append(m)

    return merged

# =========================================================================
# C1 heading-error model
# =========================================================================
def _build_c1_profile(T: int,
                      dt: float,
                      kappa: np.ndarray,
                      speed: np.ndarray,
                      manoeuvres: List[Manoeuvre],
                      tau: float = STEERING_TAU,
                      c1_peak: float = C1_PEAK_MANOEUVRE,
                      ) -> np.ndarray:
    """
    Simulate realistic heading-error (C1) profile.

    Steady state: C1 ≈ 0.
    During a manoeuvre: C1 rises to c1_peak proportional to κ/κ_max,
    then decays back to zero with time-constant τ.
    This creates the nonlinearity regime where EKF/UKF outperform KF.
    """
    C1 = np.zeros(T)
    kappa_max = max(float(np.abs(kappa).max()), 1e-6)
    dkappa = np.gradient(kappa)

    for k in range(1, T):
        # Curvature-rate excitation (steering delay)
        excite = dkappa[k] * speed[k] * dt * tau

        # Additional peak during detected manoeuvres
        for m in manoeuvres:
            if m.idx_start <= k <= m.idx_end:
                dur    = max(m.idx_end - m.idx_start, 1)
                local  = k - m.idx_start
                # Sine pulse: zero → peak → zero over manoeuvre duration
                scale  = m.kappa_peak / kappa_max
                sign   = +1.0 if m.direction == "left" else -1.0
                excite += sign * c1_peak * scale * math.sin(math.pi * local / dur) * dt / tau
                break

        C1[k] = C1[k-1] * (1.0 - dt / tau) + excite

    return np.clip(C1, -0.25, 0.25)


# =========================================================================
# Speed profile
# =========================================================================
def _speed_profile(kappa: np.ndarray,
                   v_default: float = SPEED_HIGHWAY_KMH / 3.6,
                   v_curve:   float = SPEED_RAMP_KMH   / 3.6,
                   kappa_slow: float = KAPPA_SLOW,
                   smooth_sigma: float = 15.0,
                   ) -> np.ndarray:
    """Variable speed: slow on tight curves, fast on straights."""
    v = np.where(np.abs(kappa) > kappa_slow, v_curve, v_default)
    return gaussian_filter1d(v.astype(float), sigma=smooth_sigma)


# =========================================================================
# Main generator
# =========================================================================
def generate_kml_lane_data(
    kml_path:       str | Path,
    dt:             float = 0.04,
    seed:           int   = 42,
    sigma_c0:       float = 0.05,
    sigma_c1:       float = 0.005,
    outlier_rate:   float = 0.04,
    dropout_rate:   float = 0.03,
    lane_width:     float = LANE_WIDTH_URBAN,
    smooth_sigma:   float = 1.5,
    kappa_threshold: float = KAPPA_MANOEUVRE,
):
    """
    Generate lane marker dataset from a KML GPS route file.

    Returns a LaneDataset (same schema as generate_lane_data.py) plus extra
    geometry attributes for Bokeh visualisation and a list of Manoeuvre events.
  """
    # LaneDataset is defined locally so this script works standalone
    # (no dependency on the project package layout at runtime).
    try:
        from python.data.generate_lane_data import LaneDataset
    except ImportError:
        LaneDataset = _LaneDataset  # fall back to inline definition

    rng  = np.random.default_rng(seed)
    path = Path(kml_path)

    # Parse KML 
    route = parse_kml(path)
    print(f"KML: '{route.name}'  →  {len(route.coords_ll)} waypoints")

    # Project + clean 
    coords_utm = _to_utm(route.coords_ll)
    # Remove GPS noise clusters (nodes < 1.5m apart create spurious curvature)
    coords_utm = _clean_coords(coords_utm, min_step_m=1.5, max_step_m=500.0)
    print(f"After clean: {len(coords_utm)} nodes")

    # Spline 
    cs_e, cs_n, s_knots = _build_spline(coords_utm, smooth_sigma=smooth_sigma)
    total_s = s_knots[-1]
    print(f"Route length: {total_s/1000:.2f} km")

    # Auto-detect route type from curvature statistics 
    # Build a separate heavily-smoothed spline (σ=5 nodes) solely for
    # classification — suppresses GPS noise that would inflate κ estimates.
    _cs_e_hvy, _cs_n_hvy, _s_hvy = _build_spline(coords_utm, smooth_sigma=5.0)
    _s_det = np.linspace(0, _s_hvy[-1], 500)
    _, _, _, _kappa_hvy, _ = _geometry(_cs_e_hvy, _cs_n_hvy, _s_det,
                                        smooth_sigma=5.0)
    _kappa_p75 = float(np.percentile(np.abs(_kappa_hvy), 75))
    _kappa_p90 = float(np.percentile(np.abs(_kappa_hvy), 90))
    _kappa_max = float(np.abs(_kappa_hvy).max())
    # Highway: 75th-pct κ < 0.005 (motorway + ramps still mostly straight)
    # Urban:   75th-pct κ > 0.005 (frequent turns dominate even percentiles)
    _is_highway = _kappa_p75 < 0.005

    if _is_highway:
        _lw_default   = lane_width if lane_width != LANE_WIDTH_URBAN else LANE_WIDTH_HIGHWAY
        _kappa_man    = KAPPA_MANOEUVRE_HIGHWAY
        _min_man_m    = MIN_MANOEUVRE_M          # 30m minimum arc
        _v_default    = SPEED_HIGHWAY_KMH / 3.6
        _v_curve      = SPEED_RAMP_KMH    / 3.6
        _kappa_slow   = KAPPA_SLOW               # 0.008
        _c1_peak      = C1_PEAK_MANOEUVRE        # 0.06 rad
        _tau          = STEERING_TAU             # 0.40s
        route_type    = "highway"
    else:
        _lw_default   = lane_width
        _kappa_man    = KAPPA_MANOEUVRE_URBAN
        _min_man_m    = 5.0
        _v_default    = SPEED_URBAN_KMH   / 3.6
        _v_curve      = SPEED_CURVE_KMH   / 3.6
        _kappa_slow   = 0.05
        _c1_peak      = 0.12
        _tau          = 0.35
        route_type    = "urban"

    print(f"Route type: {route_type}  "
          f"(κ_p75={_kappa_p75:.4f} 1/m  κ_p90={_kappa_p90:.4f} 1/m  "
          f"κ_max={_kappa_max:.4f} 1/m)")

    # Re-sample at camera rate via time integration 
    # Rough speed profile on knot grid for time integration
    e_k, n_k, psi_k, kappa_k, _ = _geometry(cs_e, cs_n, s_knots, smooth_sigma=smooth_sigma)
    v_knots = _speed_profile(kappa_k,
                             v_default=_v_default,
                             v_curve=_v_curve,
                             kappa_slow=_kappa_slow)
    
    v_fn = interp1d(s_knots, v_knots, kind="linear", fill_value="extrapolate")

    s_arr = [0.0]; t_arr = [0.0]
    while s_arr[-1] < total_s * 0.999:
        v = float(v_fn(s_arr[-1]))
        ds = max(v * dt, 0.01)
        s_new = min(s_arr[-1] + ds, total_s)
        s_arr.append(s_new); t_arr.append(t_arr[-1] + dt)
        if s_new >= total_s: break

    s_eval = np.array(s_arr)
    t_eval = np.array(t_arr)
    T      = len(s_eval)
    print(f"Frames: T={T}  duration={t_eval[-1]:.1f} s  dt={dt} s")

    # Full geometry at re-sampled points 
    e_ctr, n_ctr, psi, kappa, kappa_dot = _geometry(cs_e, cs_n, s_eval, smooth_sigma=smooth_sigma)

    speed   = _speed_profile(kappa)
    speed  += rng.normal(0, 0.2, T)

    # Yaw rate: dψ/dt = κ·v
    yaw_rate = np.gradient(np.unwrap(psi), t_eval)
    yaw_rate += rng.normal(0, 5e-4, T)

    # Lane markers (perpendicular offset) 
    hw = _lw_default / 2.0
    psi_perp = psi + np.pi / 2.0
    nx = np.cos(psi_perp); ny = np.sin(psi_perp)

    e_L = e_ctr + hw * nx;  n_L = n_ctr + hw * ny
    e_R = e_ctr - hw * nx;  n_R = n_ctr - hw * ny

    # Detect manoeuvres 
    manoeuvres = detect_manoeuvres(s_eval, kappa,
                                    kappa_threshold=_kappa_man,
                                    min_arc_m=_min_man_m)
    print(f"Detected manoeuvres: {len(manoeuvres)}")
    for m in manoeuvres:
        print(f"  t={t_eval[m.idx_start]:.1f}–{t_eval[m.idx_end]:.1f}s  "
              f"{m.label}  arc={m.s_end-m.s_start:.0f}m")

    # C1 heading-error profile 
    C1_true = _build_c1_profile(T, dt, kappa, speed, manoeuvres)

    #  True C0 (lane offset) 
    C0_L_true = +hw + 0.5 * speed * dt * C1_true
    C0_R_true = -hw + 0.5 * speed * dt * C1_true
    C2_true   = kappa
    C3_true   = kappa_dot

    # Camera measurements 
    def _noisy(C0, sigma):
        m = C0 + rng.normal(0, sigma, T)
        om = rng.random(T) < outlier_rate
        m[om] += rng.choice([-1, 1], om.sum()) * rng.uniform(0.2, 0.6, om.sum())
        m[rng.random(T) < dropout_rate] = np.nan
        return m

    meas_L    = _noisy(C0_L_true, sigma_c0)
    meas_R    = _noisy(C0_R_true, sigma_c0)
    meas_C1_L = C1_true + rng.normal(0, sigma_c1, T)
    meas_C1_R = C1_true + rng.normal(0, sigma_c1, T)
    meas_C1_L[rng.random(T) < dropout_rate] = np.nan
    meas_C1_R[rng.random(T) < dropout_rate] = np.nan

    # Coordinate transforms for Bokeh 
    lon_ctr, lat_ctr = _UTM32_TO_WGS84.transform(e_ctr, n_ctr)
    lon_L,   lat_L   = _UTM32_TO_WGS84.transform(e_L,   n_L)
    lon_R,   lat_R   = _UTM32_TO_WGS84.transform(e_R,   n_R)
    wx_ctr, wy_ctr   = _WGS84_TO_WEBM.transform(lon_ctr, lat_ctr)
    wx_L,   wy_L     = _WGS84_TO_WEBM.transform(lon_L,   lat_L)
    wx_R,   wy_R     = _WGS84_TO_WEBM.transform(lon_R,   lat_R)

    # Build dataset 
    ds = LaneDataset(
        t          = t_eval,
        speed      = speed,
        yaw_rate   = yaw_rate,
        C0_L_true  = C0_L_true,
        C0_R_true  = C0_R_true,
        C1_true    = C1_true,
        C2_true    = C2_true,
        C3_true    = C3_true,
        meas_L     = meas_L,
        meas_R     = meas_R,
        dt         = dt,
    )

    # Extra attributes for Bokeh
    ds.e_ctr      = e_ctr;  ds.n_ctr  = n_ctr
    ds.e_L        = e_L;    ds.n_L    = n_L
    ds.e_R        = e_R;    ds.n_R    = n_R
    ds.wx_ctr     = wx_ctr; ds.wy_ctr = wy_ctr
    ds.wx_L       = wx_L;   ds.wy_L   = wy_L
    ds.wx_R       = wx_R;   ds.wy_R   = wy_R
    ds.lon_ctr    = lon_ctr; ds.lat_ctr = lat_ctr
    ds.manoeuvres = manoeuvres
    ds.meas_C1_L  = meas_C1_L
    ds.meas_C1_R  = meas_C1_R
    ds.phase_arr  = np.zeros(T, dtype=int)  # single phase (urban)
    ds.route_name = route.name
    ds.waypoints  = route.waypoints

    return ds


# =========================================================================
# CLI 
# =========================================================================
def _cli():
    p = argparse.ArgumentParser(
        description="Generate lane dataset from a KML GPS route.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("kml",          help="Input KML file")
    p.add_argument("--out",        default=None,  help="Output CSV path")
    p.add_argument("--dt",         type=float, default=0.04)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--sigma-c0",   type=float, default=0.05, metavar="M")
    p.add_argument("--sigma-c1",   type=float, default=0.005, metavar="RAD")
    p.add_argument("--lane-width", type=float, default=LANE_WIDTH_URBAN, metavar="M")
    args = p.parse_args()

    out = args.out or str(
        Path(args.kml).parent.parent.parent / "data" /
        (Path(args.kml).stem + "_lane_data.csv")
    )
    ds = generate_kml_lane_data(
        args.kml, dt=args.dt, seed=args.seed,
        sigma_c0=args.sigma_c0, sigma_c1=args.sigma_c1,
        lane_width=args.lane_width,
    )
    ds.save_csv(out)
    print(ds.summary())
    print(f"κ range: [{ds.C2_true.min():.4f}, {ds.C2_true.max():.4f}] 1/m")
    print(f"C1 range: [{ds.C1_true.min():.4f}, {ds.C1_true.max():.4f}] rad")


if __name__ == "__main__":
    _cli()
