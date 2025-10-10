import argparse
import sqlite3
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser(
        description="Export supervision-thread comments to CSV with labels."
    )
    ap.add_argument("--db", required=True, help="Path to SQLite DB")
    ap.add_argument(
        "--outdir", default="prepared/csv", help="Output directory for CSVs"
    )
    ap.add_argument("--comments-table", default="comments")
    ap.add_argument("--train-split", default="supervision_train80")
    ap.add_argument("--val-split", default="supervision_val10")
    ap.add_argument("--test-split", default="supervision_test10")
    ap.add_argument(
        "--cols",
        nargs="*",
        help="Optional explicit columns to select from comments table",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    splits = [
        ("train", args.train_split),
        ("val", args.val_split),
        ("test", args.test_split),
    ]

    select_cols = None
    if args.cols and len(args.cols) > 0:
        cols = [c.strip() for c in args.cols if c.strip()]
        select_cols = ", ".join(f"c.{c}" for c in cols)
    else:
        select_cols = "c.*"

    with sqlite3.connect(args.db) as con:
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=OFF;")
        con.execute("PRAGMA temp_store=MEMORY;")
        con.execute("PRAGMA cache_size=-2000000;")

        for name, split in splits:
            q = f"""
            WITH tt AS (
              SELECT DISTINCT c.link_id
              FROM {args.comments_table} c
              JOIN {split} s ON s.id = c.id
            )
            SELECT {select_cols},
                   s.label AS hate_label
            FROM {args.comments_table} c
            JOIN tt t ON t.link_id = c.link_id
            LEFT JOIN {split} s ON s.id = c.id
            ORDER BY c.link_id, c.created_utc
            """
            df = pd.read_sql_query(q, con)
            out = outdir / f"{split}_threads.csv"
            df.to_csv(out, index=False)
            print(f"[OK] {name}: wrote {len(df):,} rows → {out}")


if __name__ == "__main__":
    main()
