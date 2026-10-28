from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PBMC.config import MGPConfig
from PBMC.data_processing import DATA_DIR, DEFAULT_BASELINE, load_baseline
from PBMC.experiment import run_gibbs, save_gibbs_run, subset_prepared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--leiden-resolution", type=float, default=0.5)
    parser.add_argument("--n-iter", type=int)
    parser.add_argument("--burn-in", type=int)
    parser.add_argument("--thin", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-downstream", action="store_true")
    parser.add_argument("--chain-path", type=Path, help="completed chain checkpoint path")
    parser.add_argument("--reuse-chain", action="store_true", help="skip sampling and reuse a completed chain checkpoint")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-source-check", action="store_true")
    args = parser.parse_args()
    if args.pilot and args.smoke:
        raise ValueError("choose only one of --pilot and --smoke")
    config = MGPConfig(
        precision="float32",
        leiden_resolution=args.leiden_resolution,
    )
    overrides = {
        key: value for key, value in {
            "n_iter": args.n_iter,
            "burn_in": args.burn_in,
            "thin": args.thin,
            "seed": args.seed,
        }.items() if value is not None
    }
    if overrides:
        config = replace(config, **overrides)
    if args.pilot:
        config = replace(config, n_iter=200, burn_in=100, thin=2)
    if args.smoke:
        config = replace(config, K=3, n_iter=20, burn_in=10, thin=2, n_columns=2, batch_size=2)
    prepared = load_baseline(args.baseline, verify_source=not args.skip_source_check)
    if args.smoke:
        prepared = subset_prepared(prepared, 40, 30)
    result = run_gibbs(
        prepared,
        config,
        pilot=args.pilot or args.smoke,
        downstream_enabled=not (args.no_downstream or args.pilot or args.smoke),
        chain_path=args.chain_path,
        reuse_chain=args.reuse_chain,
        checkpoint_overwrite=args.overwrite,
    )
    paths = save_gibbs_run(result, args.output_dir, overwrite=args.overwrite)
    print(json.dumps({
        "run_id": result.run_id,
        "fit": {k: v for k, v in result.diagnostics["fit"].items() if k != "retained_sweeps"},
        "rsp_status": result.diagnostics["rsp"]["status_code"],
        "active_factors": result.diagnostics["representation"]["active_factor_count"],
        "active_loading_ess": result.diagnostics["sampling"]["active_loading_bulk_ess"],
        "review_required": result.diagnostics["sampling"]["review_required"],
        "downstream": result.downstream,
        "h5ad": str(paths[0]),
        "model": str(paths[1]),
    }, default=str, indent=2))


if __name__ == "__main__":
    main()
