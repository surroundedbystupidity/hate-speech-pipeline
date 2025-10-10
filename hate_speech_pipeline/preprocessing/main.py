#!/usr/bin/env python3
"""
Small runner for the preprocessing pipeline.
Runs the set of scripts required to build the DB, label and prepare supervision CSVs.

Usage examples are included in the --help text.
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_ORDER = [
    "build_db_from_zst.py",
    "fast_prune_orphans_cte.py",
    "make_train_pool_from_lexicon.py",
    "label_pool_in_place.py",
    "build_supervision_train.py",
    "chrono_thread_split_80_10_10.py",
    "dump_supervision_threads.py",
]


def run_cmd(cmd, cwd=None):
    logger.info("Running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=cwd)
    if proc.returncode != 0:
        logger.error("Command failed: %s (exit %d)", " ".join(cmd), proc.returncode)
        sys.exit(proc.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Run the preprocessing pipeline scripts in order."
    )
    parser.add_argument(
        "--scripts-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Directory containing the preprocessing scripts (default: this folder)",
    )
    parser.add_argument(
        "--db", type=str, required=True, help="Path to sqlite DB to operate on"
    )
    parser.add_argument(
        "--lexicon-csv",
        type=str,
        required=False,
        help="Path to refined_ngram_dict.csv for make_train_pool_from_lexicon.py",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="prepared/csv",
        help="Output dir for dump_supervision_threads.py",
    )

    args = parser.parse_args()
    scripts_dir = args.scripts_dir

    # sanity checks
    for name in SCRIPT_ORDER:
        p = scripts_dir / name
        if not p.exists():
            logger.warning("Expected script %s not found in %s", name, scripts_dir)

    # 1: build_db_from_zst.py
    run_cmd([sys.executable, str(scripts_dir / "build_db_from_zst.py")])

    # 2: fast_prune_orphans_cte.py <db>
    run_cmd([sys.executable, str(scripts_dir / "fast_prune_orphans_cte.py"), args.db])

    # 3: make_train_pool_from_lexicon.py --db <db> --csv <lexicon_csv>
    if args.lexicon_csv:
        run_cmd(
            [
                sys.executable,
                str(scripts_dir / "make_train_pool_from_lexicon.py"),
                "--db",
                args.db,
                "--csv",
                args.lexicon_csv,
            ]
        )
    else:
        logger.info("Skipping make_train_pool_from_lexicon (no --lexicon-csv provided)")

    # 4: label_pool_in_place.py --db <db>
    run_cmd(
        [sys.executable, str(scripts_dir / "label_pool_in_place.py"), "--db", args.db]
    )

    # 5: build_supervision_train.py --db <db> --k 1 --make-val-test
    run_cmd(
        [
            sys.executable,
            str(scripts_dir / "build_supervision_train.py"),
            "--db",
            args.db,
            "--k",
            "1",
            "--make-val-test",
        ]
    )

    # 6: chrono_thread_split_80_10_10.py --db <db>
    run_cmd(
        [
            sys.executable,
            str(scripts_dir / "chrono_thread_split_80_10_10.py"),
            "--db",
            args.db,
        ]
    )

    # 7: dump_supervision_threads.py --db <db> --outdir <outdir>
    run_cmd(
        [
            sys.executable,
            str(scripts_dir / "dump_supervision_threads.py"),
            "--db",
            args.db,
            "--outdir",
            args.outdir,
        ]
    )

    logger.info("Preprocessing pipeline completed successfully.")


if __name__ == "__main__":
    main()
