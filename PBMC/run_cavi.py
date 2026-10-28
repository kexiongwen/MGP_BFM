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
from PBMC.experiment import run_cavi, run_stem, save_cavi_run, subset_prepared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--max-iter", type=int)
    parser.add_argument("--leiden-resolution", type=float, default=0.425)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-downstream", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-source-check", action="store_true")
    args = parser.parse_args()
    if args.pilot and args.smoke:
        raise ValueError("choose only one of --pilot and --smoke")
    config = MGPConfig(
        precision="float64",
        leiden_resolution=args.leiden_resolution,
    )
    if args.max_iter is not None:
        config = replace(config, max_iter=args.max_iter)
    if args.pilot:
        config = replace(config, max_iter=20)
    if args.smoke:
        config = replace(config, K=3, max_iter=5, n_columns=2)
    prepared = load_baseline(args.baseline, verify_source=not args.skip_source_check)
    if args.smoke:
        prepared = subset_prepared(prepared, 40, 30)
    result = run_cavi(
        prepared,
        config,
        pilot=args.pilot or args.smoke,
        downstream_enabled=not (args.no_downstream or args.pilot or args.smoke),
    )
    paths = save_cavi_run(result, args.output_dir, overwrite=args.overwrite)
    print(json.dumps({
        "run_id": result.run_id,
        "selected_start": result.diagnostics["selected_start"],
        "fit": result.diagnostics["fit"],
        "rsp_status": result.diagnostics["rsp"]["status_code"],
        "active_factors": result.diagnostics["representation"]["active_factor_count"],
        "downstream": result.downstream,
        "h5ad": str(paths[0]),
        "model": str(paths[1]),
    }, default=str, indent=2))


if __name__ == "__main__":
    main()
