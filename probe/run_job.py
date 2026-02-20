# entry point for single job execution
import json
import sys
import numpy as np

from matrix_builder import build_matrix
from diagonalize import diagonalize_dense
from observables import compute_gap, spacing_ratios
from canonical_json import canonical_json
from hash_utils import sha256_str


def main(input_path):
    with open(input_path, "r") as f:
        input_data = json.load(f)

    run_id = sha256_str(canonical_json(input_data))

    H = build_matrix(input_data["run_params"])
    eigs = diagonalize_dense(H)

    delta1, deltak = compute_gap(eigs)
    r = spacing_ratios(eigs)

    output = {
        "probe_version": input_data["probe_version"],
        "run_id": run_id,
        "status": "ok",
        "results": {
            "eigs_low": eigs[:11].tolist(),
            "gap": {
                "delta1": float(delta1),
                "deltak": float(deltak) if deltak else None,
            },
            "bulk_r": {
                "r_mean": float(np.mean(r)),
                "count": len(r),
            },
        },
    }

    commit_hash = sha256_str(canonical_json(output))
    output["commit_hash"] = commit_hash

    with open("output.json", "w") as f:
        json.dump(output, f, indent=2)

    print("run_id:", run_id)
    print("commit_hash:", commit_hash)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_job.py input.json")
        sys.exit(1)

    main(sys.argv[1])
