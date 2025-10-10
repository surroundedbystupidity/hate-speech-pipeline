import argparse
import math
import sqlite3


def main():
    ap = argparse.ArgumentParser(
        description="Chronological thread-aware 80/10/10 split."
    )
    ap.add_argument("--db", required=True, help="Path to SQLite DB")
    ap.add_argument(
        "--train-table",
        default="supervision_train",
        help="Source supervision table with (id,label). Default: supervision_train",
    )
    ap.add_argument(
        "--comments-table",
        default="comments",
        help="Comments table with (id, link_id, created_utc). Default: comments",
    )
    ap.add_argument(
        "--train-out",
        default="supervision_train80",
        help="Output table for 80% train (id,label)",
    )
    ap.add_argument(
        "--val-out",
        default="supervision_val10",
        help="Output table for 10% val (id,label)",
    )
    ap.add_argument(
        "--test-out",
        default="supervision_test10",
        help="Output table for 10% test (id,label)",
    )
    ap.add_argument(
        "--train-ratio", type=float, default=0.8, help="Train ratio (default 0.8)"
    )
    ap.add_argument(
        "--val-ratio", type=float, default=0.1, help="Val ratio (default 0.1)"
    )
    args = ap.parse_args()

    if not (0 < args.train_ratio < 1) or not (0 < args.val_ratio < 1):
        raise SystemExit("train-ratio and val-ratio must be in (0,1).")
    if args.train_ratio + args.val_ratio >= 1.0:
        raise SystemExit(
            "train-ratio + val-ratio must be < 1.0 (remainder goes to test)."
        )

    con = sqlite3.connect(args.db, timeout=300)
    cur = con.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA synchronous=OFF")
    cur.execute("PRAGMA temp_store=MEMORY")
    cur.execute("PRAGMA cache_size=-200000")
    con.commit()

    def has_cols(table, needed):
        cur.execute(f"PRAGMA table_info({table})")
        cols = {r[1] for r in cur.fetchall()}
        return all(c in cols for c in needed)

    if not has_cols(args.train_table, ["id", "label"]):
        raise SystemExit(f"ERROR: {args.train_table} must have columns (id,label).")
    if not has_cols(args.comments_table, ["id", "link_id", "created_utc"]):
        raise SystemExit(
            f"ERROR: {args.comments_table} must have (id,link_id,created_utc)."
        )

    print("[INFO] Computing thread start times from supervision_train …")
    cur.execute(
        f"""
        WITH sup AS (
            SELECT s.id, s.label, c.link_id, c.created_utc
            FROM {args.train_table} AS s
            JOIN {args.comments_table} AS c ON c.id = s.id
        )
        SELECT link_id, MIN(created_utc) AS start_ts
        FROM sup
        GROUP BY link_id
        ORDER BY start_ts ASC, link_id ASC
    """
    )
    rows = cur.fetchall()
    if not rows:
        raise SystemExit(
            "ERROR: No rows found when joining supervision_train to comments."
        )

    n_threads = len(rows)
    n_train_threads = max(1, math.floor(n_threads * args.train_ratio))
    n_val_threads = max(1, math.floor(n_threads * args.val_ratio))
    n_test_threads = n_threads - n_train_threads - n_val_threads
    if n_test_threads <= 0:
        n_test_threads = 1
        if n_val_threads > 1:
            n_val_threads -= 1
        else:
            n_train_threads = max(1, n_train_threads - 1)

    print(
        f"[INFO] Threads: total={n_threads:,} (train={n_train_threads:,}, val={n_val_threads:,}, test={n_test_threads:,})"
    )

    train_links = [link for link, _ in rows[:n_train_threads]]
    val_links = [
        link for link, _ in rows[n_train_threads : n_train_threads + n_val_threads]
    ]
    test_links = [link for link, _ in rows[n_train_threads + n_val_threads :]]

    cur.execute("DROP TABLE IF EXISTS _thread_split_bins")
    cur.execute(
        "CREATE TEMP TABLE _thread_split_bins (link_id TEXT PRIMARY KEY, split TEXT NOT NULL)"
    )
    cur.executemany(
        "INSERT OR IGNORE INTO _thread_split_bins(link_id, split) VALUES (?, 'train')",
        [(x,) for x in train_links],
    )
    cur.executemany(
        "INSERT OR IGNORE INTO _thread_split_bins(link_id, split) VALUES (?, 'val')",
        [(x,) for x in val_links],
    )
    cur.executemany(
        "INSERT OR IGNORE INTO _thread_split_bins(link_id, split) VALUES (?, 'test')",
        [(x,) for x in test_links],
    )
    con.commit()

    print(
        "[INFO] Writing supervision_train80 / supervision_val10 / supervision_test10 …"
    )
    for tbl in (args.train_out, args.val_out, args.test_out):
        cur.execute(f"DROP TABLE IF EXISTS {tbl}")
    con.commit()

    # Train
    cur.execute(
        f"""
        CREATE TABLE {args.train_out} AS
        SELECT s.id, s.label
        FROM {args.train_table} AS s
        JOIN {args.comments_table} AS c ON c.id = s.id
        JOIN _thread_split_bins b ON b.link_id = c.link_id
        WHERE b.split = 'train'
    """
    )
    # Val
    cur.execute(
        f"""
        CREATE TABLE {args.val_out} AS
        SELECT s.id, s.label
        FROM {args.train_table} AS s
        JOIN {args.comments_table} AS c ON c.id = s.id
        JOIN _thread_split_bins b ON b.link_id = c.link_id
        WHERE b.split = 'val'
    """
    )
    # Test
    cur.execute(
        f"""
        CREATE TABLE {args.test_out} AS
        SELECT s.id, s.label
        FROM {args.train_table} AS s
        JOIN {args.comments_table} AS c ON c.id = s.id
        JOIN _thread_split_bins b ON b.link_id = c.link_id
        WHERE b.split = 'test'
    """
    )
    con.commit()

    cur.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{args.train_out}_id ON {args.train_out}(id)"
    )
    cur.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{args.val_out}_id   ON {args.val_out}(id)"
    )
    cur.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{args.test_out}_id  ON {args.test_out}(id)"
    )
    con.commit()

    def counts(tbl):
        cur.execute(f"SELECT label, COUNT(*) FROM {tbl} GROUP BY label")
        d = dict(cur.fetchall())
        total = sum(d.values())
        return total, d.get(1, 0), d.get(0, 0)

    tr_tot, tr_pos, tr_neg = counts(args.train_out)
    va_tot, va_pos, va_neg = counts(args.val_out)
    te_tot, te_pos, te_neg = counts(args.test_out)

    print("[INFO] Split sizes (total / pos / neg):")
    print(f"  {args.train_out}: {tr_tot:,} / {tr_pos:,} / {tr_neg:,}")
    print(f"  {args.val_out}:   {va_tot:,} / {va_pos:,} / {va_neg:,}")
    print(f"  {args.test_out}:  {te_tot:,} / {te_pos:,} / {te_neg:,}")

    cur.execute(f"SELECT COUNT(*) FROM {args.train_table}")
    n_src = cur.fetchone()[0]
    if tr_tot + va_tot + te_tot != n_src:
        print(
            "[WARN] Split totals != supervision_train rows "
            f"({tr_tot + va_tot + te_tot:,} vs {n_src:,}). "
            "This would only happen if some supervision_train ids lacked link_id in comments."
        )
    else:
        print("[OK] All supervision_train rows assigned to exactly one split.")

    con.close()
    print("[OK] Done.")


if __name__ == "__main__":
    main()
