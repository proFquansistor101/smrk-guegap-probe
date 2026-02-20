# pre-registered scan runner
# SMRK-GUEGAP Probe — scans/run_scan_v1.py
# Pre-registered v1 grid scan (dense, deterministic)
#
# Run:
#   python scans/run_scan_v1.py
#
# Output:
#   runs/scan_v1_<UTCSTAMP>/
#     scan_config.json
#     scan_table.csv
#     scan_report.json
#     jobs/<run_id>_input.json
#     jobs/<run_id>_output.json
#
# Notes:
# - v1 is dense diagonalization, intended for N <= 2048.
# - No randomness is used.
# - This script is intentionally conservative and audit-friendly.

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

# Allow "python scans/run_scan_v1.py" from repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROBE_DIR = os.path.join(REPO_ROOT, "probe")
if PROBE_DIR not in os.sys.path:
    os.sys.path.insert(0, PROBE_DIR)

from build_info import get_build_info, get_platform_info
from canonical_json import canonical_json
from hash_utils import sha256_str
from matrix_builder import build_matrix
from diagonalize import diagonalize_dense
from observables import compute_gap, spacing_ratios


# ----------------------------
# v1 scan configuration
# ----------------------------

PROBE_VERSION = "smrk-guegap-v1"
OPERATOR_VARIANT = "H_phi_v1"

N_LADDER = [256, 512, 1024, 2048]
M_DEFAULT = 8
BOUNDARY_DEFAULT = "cyclic"

BETA0 = 0.2
BETA1 = 0.8
K_MIN = 10

# Primary v1 scan axes (Section 5)
GRID_KAPPA = [0.01, 0.02, 0.05, 0.1, 0.2]
GRID_LAMBDA = [0.0, 0.05, 0.1, 0.2]
GRID_PHASE_DELTA = [0.0, 0.05, 0.1]

# Fixed defaults
S_DEFAULT = 1.0
T_DEFAULT = 0.0
ETA_DEFAULT = 0.5
ALPHA_DEFAULT = 0.6180339887498948  # golden ratio conjugate

# Acceptance thresholds (v1 defaults)
EPS_R = 0.03
# Reference mean for GUE r-stat (Atas et al. / standard RMT value)
R_GUE_MEAN = 0.5996

# ----------------------------
# Helpers
# ----------------------------

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def phi_offsets_uniform(M: int, phase_delta: float) -> List[float]:
    """
    Base offsets phi_m = 2π m/M, then apply scan perturbation:
      phi_m <- phi_m + 2π * phase_delta * m
    """
    out = []
    for m in range(1, M + 1):
        base = 2.0 * np.pi * (m / M)
        pert = 2.0 * np.pi * phase_delta * m
        out.append(float(base + pert))
    return out


def build_input_record(
    *,
    N: int,
    M: int,
    boundary: str,
    kappa: float,
    lam: float,
    phase_delta: float,
) -> Dict[str, Any]:
    # Note: keep input minimal but schema-aligned enough for v1
    theta = {
        "kappa": float(kappa),
        "lambda": float(lam),
        "mu": 0.0,
        "s": float(S_DEFAULT),
        "t": float(T_DEFAULT),
        "eta": float(ETA_DEFAULT),
        "alpha": float(ALPHA_DEFAULT),
        "phi_offsets": phi_offsets_uniform(M, phase_delta),
        "phase_delta": float(phase_delta),
    }

    record = {
        "probe_version": PROBE_VERSION,
        "operator_variant": OPERATOR_VARIANT,
        "run_params": {
            "N": int(N),
            "M": int(M),
            "boundary": boundary,
            "bulk_window": {"beta0": float(BETA0), "beta1": float(BETA1)},
            "k_min": int(K_MIN),
            "amplitude_model": "liouville_decay_v1",
            "phase_model": "golden_gauge_v1",
            "potential_model": "V_log1p_v1",
            "theta": theta,
            "quantize": False,
            "quantize_rule": None,
        },
        "scan": {
            "scan_id": "primary_v1",
            "grid_axes": {
                "kappa": GRID_KAPPA,
                "lambda": GRID_LAMBDA,
                "phase_delta": GRID_PHASE_DELTA,
            },
            "thresholds": {
                "eps_r": float(EPS_R),
                "r_gue_mean": float(R_GUE_MEAN),
            },
        },
        "build": get_build_info(REPO_ROOT),
        "platform": get_platform_info(),
    }
    return record


def evaluate_job(input_record: Dict[str, Any]) -> Dict[str, Any]:
    run_id = sha256_str(canonical_json(input_record))
    rp = input_record["run_params"]
    theta = rp["theta"]

    # Build + diagonalize (dense)
    H = build_matrix(rp)
    eigs = diagonalize_dense(H)
    eigs = np.array(eigs, dtype=np.float64)
    eigs.sort()

    # Observables
    delta1, deltak = compute_gap(eigs, k_min=K_MIN)
    r_vals = spacing_ratios(eigs, beta0=BETA0, beta1=BETA1)

    r_mean = float(np.mean(r_vals)) if len(r_vals) else float("nan")
    h1_pass = (not np.isnan(r_mean)) and (abs(r_mean - R_GUE_MEAN) <= EPS_R)

    output = {
        "probe_version": input_record["probe_version"],
        "run_id": run_id,
        "status": "ok",
        "results": {
            "eigs_low": eigs[: (K_MIN + 1)].tolist(),
            "gap": {
                "delta1": float(delta1),
                "deltak": float(deltak) if deltak is not None else None,
            },
            "bulk_r": {
                "r_mean": r_mean,
                "count": int(len(r_vals)),
                "ref": {"kind": "GUE", "r_mean": float(R_GUE_MEAN), "eps_r": float(EPS_R)},
            },
            "acceptance": {
                "H1_pass_mean_only": bool(h1_pass),
            },
        },
        "diagnostics": {
            "matrix": {
                "N": int(rp["N"]),
                "M": int(rp["M"]),
                "boundary": rp["boundary"],
                "dtype": "complex128",
            }
        },
        "build": input_record["build"],
        "platform": input_record["platform"],
    }

    commit_hash = sha256_str(canonical_json(output))
    output["commit_hash"] = commit_hash
    return output


def write_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=True)


def main() -> None:
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    out_dir = os.path.join(REPO_ROOT, "runs", f"scan_v1_{stamp}")
    jobs_dir = os.path.join(out_dir, "jobs")
    ensure_dir(jobs_dir)

    scan_config = {
        "probe_version": PROBE_VERSION,
        "operator_variant": OPERATOR_VARIANT,
        "N_ladder": N_LADDER,
        "M": M_DEFAULT,
        "boundary": BOUNDARY_DEFAULT,
        "grid": {
            "kappa": GRID_KAPPA,
            "lambda": GRID_LAMBDA,
            "phase_delta": GRID_PHASE_DELTA,
        },
        "bulk_window": {"beta0": BETA0, "beta1": BETA1},
        "k_min": K_MIN,
        "thresholds": {"eps_r": EPS_R, "r_gue_mean": R_GUE_MEAN},
        "build": get_build_info(REPO_ROOT),
        "platform": get_platform_info(),
    }
    write_json(os.path.join(out_dir, "scan_config.json"), scan_config)

    rows: List[Dict[str, Any]] = []

    total_points = len(N_LADDER) * len(GRID_KAPPA) * len(GRID_LAMBDA) * len(GRID_PHASE_DELTA)
    done = 0
    t0 = time.time()

    for kappa in GRID_KAPPA:
        for lam in GRID_LAMBDA:
            for phase_delta in GRID_PHASE_DELTA:
                for N in N_LADDER:
                    input_rec = build_input_record(
                        N=N,
                        M=M_DEFAULT,
                        boundary=BOUNDARY_DEFAULT,
                        kappa=kappa,
                        lam=lam,
                        phase_delta=phase_delta,
                    )
                    run_id = sha256_str(canonical_json(input_rec))

                    # Persist input
                    in_path = os.path.join(jobs_dir, f"{run_id}_input.json")
                    write_json(in_path, input_rec)

                    # Evaluate
                    out = evaluate_job(input_rec)

                    # Persist output
                    out_path = os.path.join(jobs_dir, f"{run_id}_output.json")
                    write_json(out_path, out)

                    # Collect row
                    res = out["results"]
                    theta = input_rec["run_params"]["theta"]
                    row = {
                        "run_id": run_id,
                        "commit_hash": out["commit_hash"],
                        "N": N,
                        "M": M_DEFAULT,
                        "boundary": BOUNDARY_DEFAULT,
                        "kappa": kappa,
                        "lambda": lam,
                        "phase_delta": phase_delta,
                        "r_mean": res["bulk_r"]["r_mean"],
                        "r_count": res["bulk_r"]["count"],
                        "delta1": res["gap"]["delta1"],
                        "deltak": res["gap"]["deltak"],
                        "H1_pass_mean_only": res["acceptance"]["H1_pass_mean_only"],
                    }
                    rows.append(row)

                    # Progress
                    done += 1
                    if done % 5 == 0 or done == total_points:
                        dt = time.time() - t0
                        rate = done / dt if dt > 0 else 0.0
                        print(f"[{done}/{total_points}] kappa={kappa} lambda={lam} delta={phase_delta} N={N} | {rate:.2f} jobs/s")

    # Write CSV table
    csv_path = os.path.join(out_dir, "scan_table.csv")
    fieldnames = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Minimal scan report (aggregate)
    # Aggregate per (kappa, lambda, phase_delta): count of H1 passes across N ladder
    agg: Dict[str, Any] = {}
    for r in rows:
        key = f'kappa={r["kappa"]}|lambda={r["lambda"]}|phase_delta={r["phase_delta"]}'
        agg.setdefault(key, {"H1_pass_count": 0, "points": 0})
        agg[key]["points"] += 1
        if r["H1_pass_mean_only"]:
            agg[key]["H1_pass_count"] += 1

    report = {
        "probe_version": PROBE_VERSION,
        "operator_variant": OPERATOR_VARIANT,
        "generated_utc": stamp,
        "total_jobs": int(len(rows)),
        "config": scan_config,
        "aggregate": agg,
        "notes": [
            "This v1 scan uses dense diagonalization and mean-only H1 test.",
            "H2/H3 acceptance is not evaluated in this minimal scanner yet; add in v2 script or extend here.",
        ],
    }
    write_json(os.path.join(out_dir, "scan_report.json"), report)

    print("\nDONE")
    print("Output directory:", out_dir)
    print("Table:", csv_path)
    print("Report:", os.path.join(out_dir, "scan_report.json"))


if __name__ == "__main__":
    main()
