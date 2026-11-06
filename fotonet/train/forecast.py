"""Final-mAP forecast for a training run (first used by the fotonet v1.2 trainer).

``probe.forecast@1.2``. After each validation the trainer fits the run's mAP50-95 curve so far and
writes ``<run>/forecast.json``: the expected final mAP, an 80% range, and the expected curve for
every epoch left. Studio's training tab draws it.

The model knows the run's own plan (the learning rate of every epoch, the image-size steps and the
epoch mosaic closes), so it sees the decay coming before it happens:

    raw(e) = M - A * (S(e) + s0)^-alpha                                    progress by LR area
           + C * sum_k drop_k * (1 - exp(-(S(e) - S(k-1)) / lam))          the gain from lowering the LR
           + g_size * sum_j (1 - exp(-(e - e_j) / 15))                     image-size steps (finding 3)
           + g_mosaic * (1 - exp(-(e - e_m) / 5))                          mosaic off
           + sum_i g_i * (1 - exp(-(e - e_i) / 4))                         live or scheduled knob changes
    observed(e) = EMA of raw over the run's EMA window (validation scores EMA weights)

``S(e)`` is the summed learning rate (relative to lr0) up to epoch ``e`` and ``drop_k`` the fall in
learning rate entering epoch ``k``: a constant rate earns no annealing gain, a decay earns it as
training continues at the lower rate. It is a reduced form of the multi-power law for learning-rate
schedules (Luo et al., 2025). The parameters start from fotonetn r2's 500-epoch curve (``PARAMS``)
and give way to the run's own scores as they come in. The 80% range combines the fit's own
uncertainty with a model-error term that shrinks with the share of the run still to go, calibrated
on r2 (``MODEL_SPREAD``, ``SPREAD_POWER``).
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np

TAU_SIZE = 15.0
TAU_MOSAIC = 5.0
TAU_KNOB = 4.0
# name: (prior, prior sigma, lower bound, upper bound). Prior values fitted on runs/v11_full500_r2
# (fotonetn, recipe fotonetn_scratch, 500 epochs, knob change at epoch 205) with fit_prior(): RMSE 0.24 AP
# over epochs 20-500. r2's scores leave a ridge between the power law and the annealing gain, so the
# power law's shape (s0, alpha) is held near one point of it. Backtest on r2: the 80% range held the
# final mAP at all 13 cut points, and from epoch 250 on the error was 0.6 AP or less.
PARAMS = {
    "M": (23.17, 15.0, 0.0, 100.0),
    "log_A": (8.14, 1.5, -2.0, 9.0),
    "log_s0": (4.50, 0.15, 3.5, 4.6),
    "log_alpha": (0.25, 0.15, -0.5, 0.25),
    "C": (21.57, 5.0, 0.0, 40.0),
    "log_lam": (4.475, 0.7, 0.0, 6.5),
    "g_size": (1.24, 0.8, -2.0, 6.0),
    "g_mosaic": (0.61, 0.4, -1.0, 3.0),
}
KNOB_PRIOR = (0.0, 1.5, -5.0, 8.0)
MODEL_SPREAD = 4.0
SPREAD_POWER = 1.25
OBS_SIGMA = 0.15
Z80 = 1.2816


def plan_arrays(lr, imgsz, mosaic):
    """Per-epoch plan as arrays: learning rate relative to its peak, image size, mosaic probability."""
    lr = np.asarray(lr, dtype=np.float64)
    peak = float(lr.max()) if len(lr) else 1.0
    return {"eta": lr / max(peak, 1e-12), "imgsz": np.asarray(imgsz, dtype=np.float64),
            "mosaic": np.asarray(mosaic, dtype=np.float64)}


def _events(plan):
    """Epochs (1-based) where the image size grows, and the first epoch with mosaic off."""
    sizes, mos = plan["imgsz"], plan["mosaic"]
    steps = [i + 1 for i in range(1, len(sizes)) if sizes[i] > sizes[i - 1]]
    closes = [i + 1 for i in range(1, len(mos)) if mos[i - 1] > 0 and mos[i] <= 0]
    return steps, closes[:1]


def _rise(epochs, e0, tau):
    return np.where(epochs >= e0, 1.0 - np.exp(-(epochs - e0 + 1.0) / tau), 0.0)


def _curve(theta, plan, tau_ema, knobs=()):
    """Expected validation mAP (EMA weights) for every epoch of the plan."""
    eta = plan["eta"]
    E = len(eta)
    S = np.cumsum(eta)
    A, s0, alpha = math.exp(theta["log_A"]), math.exp(theta["log_s0"]), math.exp(theta["log_alpha"])
    raw = theta["M"] - A * (S + s0) ** (-alpha)
    drops = np.clip(np.concatenate(([0.0], eta[:-1] - eta[1:])), 0.0, None)
    nz = np.nonzero(drops > 1e-9)[0]
    if len(nz):
        lam = math.exp(theta["log_lam"])
        S_before = np.concatenate(([0.0], S[:-1]))
        diff = np.clip(S[:, None] - S_before[None, nz], 0.0, None)
        live = np.arange(E)[:, None] >= nz[None, :]
        raw = raw + theta["C"] * (np.where(live, 1.0 - np.exp(-diff / lam), 0.0) * drops[nz]).sum(1)
    epochs = np.arange(1, E + 1, dtype=np.float64)
    steps, closes = _events(plan)
    for e0 in steps:
        raw = raw + theta["g_size"] * _rise(epochs, e0, TAU_SIZE)
    for e0 in closes:
        raw = raw + theta["g_mosaic"] * _rise(epochs, e0, TAU_MOSAIC)
    for e0 in knobs:
        raw = raw + theta.get(f"g_knob{int(e0)}", 0.0) * _rise(epochs, e0, TAU_KNOB)
    keep = math.exp(-1.0 / max(float(tau_ema), 1e-6))
    out = np.empty(E)
    acc = raw[0]
    for i in range(E):
        acc = keep * acc + (1.0 - keep) * raw[i]
        out[i] = acc
    return out


def _table(prior, knobs):
    table = dict(PARAMS)
    for key, value in dict(prior or {}).items():
        if key in table:
            p, s, lo, hi = table[key]
            table[key] = (min(max(float(value), lo), hi), s, lo, hi)
    for e0 in knobs:
        table[f"g_knob{int(e0)}"] = KNOB_PRIOR
    return table


def _fit(obs_e, obs_y, plan, tau_ema, table, knobs=(), weights=None, prior_scale=1.0):
    from scipy.optimize import least_squares

    keys = list(table)
    x0 = np.array([table[k][0] for k in keys])
    sig = np.array([table[k][1] for k in keys]) * float(prior_scale)
    lo = np.array([table[k][2] for k in keys])
    hi = np.array([table[k][3] for k in keys])
    idx = np.asarray(obs_e, dtype=np.int64) - 1
    y = np.asarray(obs_y, dtype=np.float64)
    w = np.ones_like(y) if weights is None else np.asarray(weights, dtype=np.float64)

    def residuals(x):
        model = _curve(dict(zip(keys, x)), plan, tau_ema, knobs)[idx]
        return np.concatenate(((model - y) / OBS_SIGMA * w, (x - x0) / sig))

    res = least_squares(residuals, np.clip(x0, lo + 1e-9, hi - 1e-9), bounds=(lo, hi), loss="soft_l1",
                        f_scale=3.0, max_nfev=600)
    theta = dict(zip(keys, res.x))
    try:
        cov = np.linalg.pinv(res.jac.T @ res.jac)
    except Exception:
        cov = np.diag(sig ** 2)
    return theta, cov, keys


def fit_prior(epochs, maps, plan, tau_ema, knobs=()):
    """Refit the prior from a finished run, with loose priors. Returns ``{name: value}``."""
    theta, _cov, _keys = _fit(epochs, maps, plan, tau_ema, _table(None, knobs), knobs, prior_scale=4.0)
    return {k: round(float(v), 4) for k, v in theta.items()}


def forecast(obs_e, obs_y, plan, tau_ema, knobs=(), prior=None, samples=200, seed=0):
    """The forecast for validation scores ``obs_y`` (AP points) at epochs ``obs_e``.

    ``knobs`` are epochs where a live or scheduled setting changed. Returns None until there are at
    least 4 scores and one from epoch 10 on.
    """
    obs_e = np.asarray(obs_e, dtype=np.int64)
    obs_y = np.asarray(obs_y, dtype=np.float64)
    keep = obs_e >= 3
    obs_e, obs_y = obs_e[keep], obs_y[keep]
    E = len(plan["eta"])
    if len(obs_e) < 4 or obs_e.max() < 10 or E < 2:
        return None
    last = int(obs_e.max())
    knobs = sorted({int(k) for k in knobs if 3 <= int(k) <= last})
    weights = np.clip(obs_e / 30.0, 0.3, 1.0)   # the first epochs move fastest and fit worst
    theta, cov, keys = _fit(obs_e, obs_y, plan, tau_ema, _table(prior, knobs), knobs, weights)
    best = _curve(theta, plan, tau_ema, knobs)
    x = np.array([theta[k] for k in keys])
    rng = np.random.default_rng(seed)
    lo = np.array([_table(prior, knobs)[k][2] for k in keys])
    hi = np.array([_table(prior, knobs)[k][3] for k in keys])
    try:
        draws = np.clip(rng.multivariate_normal(x, cov, size=int(samples), method="eigh"), lo, hi)
        param_var = np.var(np.stack([_curve(dict(zip(keys, d)), plan, tau_ema, knobs) for d in draws]), axis=0)
    except Exception:
        param_var = np.zeros(E)
    left = np.clip((np.arange(1, E + 1) - last) / E, 0.0, None)
    model_sd = MODEL_SPREAD * left ** SPREAD_POWER
    # The fit's own spread never says more than the model-error term allows.
    spread = np.sqrt(np.minimum(param_var, (1.5 * model_sd) ** 2 + 0.04) + model_sd ** 2)
    fitted = best[obs_e - 1]
    rmse = float(np.sqrt(np.mean((fitted - obs_y) ** 2)))
    future = np.arange(last + 1, E + 1)
    final_sd = math.sqrt(float(spread[-1]) ** 2 + (rmse if last < E else 0.0) ** 2 * 0.25)
    return {
        "epoch": last,
        "epochs": int(E),
        "fraction": round(last / E, 4),
        "final": round(float(best[-1]), 2),
        "low": round(float(best[-1] - Z80 * final_sd), 2),
        "high": round(float(best[-1] + Z80 * final_sd), 2),
        "band": 80,
        "best_so_far": round(float(obs_y.max()), 2),
        "fit_rmse": round(rmse, 3),
        "points": int(len(obs_e)),
        "knobs": knobs,
        "curve": [[int(e), round(float(best[e - 1]), 2)] for e in future],
        "curve_low": [[int(e), round(float(best[e - 1] - Z80 * spread[e - 1]), 2)] for e in future],
        "curve_high": [[int(e), round(float(best[e - 1] + Z80 * spread[e - 1]), 2)] for e in future],
        "fitted": [[int(e), round(float(v), 2)] for e, v in zip(obs_e, fitted)],
        "observed": [[int(e), round(float(v), 2)] for e, v in zip(obs_e, obs_y)],
        "params": {k: round(float(v), 4) for k, v in theta.items()},
        "method": "LR-area power law plus annealing gain (reduced multi-power law), prior from fotonetn r2",
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def scores_from_rows(rows, metric="mAP50_95"):
    """``(epochs, AP points, knob epochs)`` from a run's ``metrics.jsonl`` rows."""
    pts, knobs = {}, set()
    for row in rows:
        epoch = int(row.get("epoch", 0) or 0)
        value = row.get(metric)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            pts[epoch] = float(value) * (100.0 if float(value) <= 1.0 else 1.0)
        changed = set(dict(row.get("live_changes") or {})) | set(dict(row.get("scheduled_changes") or {}))
        if changed - {"preview_minutes", "probe_period", "workers", "val_period", "patience", "fast_val", "val_amp"}:
            knobs.add(epoch)
    epochs = sorted(pts)
    return epochs, [pts[e] for e in epochs], sorted(knobs)


def write_forecast(out_dir, rows, plan, tau_ema, metric="mAP50_95"):
    """Forecast from a run's ``metrics.jsonl`` rows; writes ``forecast.json`` and returns the result."""
    epochs, scores, knobs = scores_from_rows(rows, metric)
    if not epochs:
        return None
    result = forecast(epochs, scores, plan, tau_ema, knobs)
    if result is not None:
        tmp = os.path.join(out_dir, "forecast.json.tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(result, handle)
        os.replace(tmp, os.path.join(out_dir, "forecast.json"))
    return result


__all__ = ["PARAMS", "fit_prior", "forecast", "plan_arrays", "scores_from_rows", "write_forecast"]
