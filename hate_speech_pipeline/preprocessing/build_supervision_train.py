import argparse
import sqlite3
import sys


def execmany(cur, stmts):
    for s in stmts:
        cur.execute(s)


def main():
    ap = argparse.ArgumentParser(
        description="Create positives_train and balanced supervision_train (thread & time matched)."
    )
    ap.add_argument("--db", required=True, help="Path to SQLite DB")
    ap.add_argument(
        "--train-end",
        type=int,
        default=1479600000,
        help="Unix ts for train end (default 2016-11-20 00:00:00 UTC)",
    )
    ap.add_argument(
        "--val-end",
        type=int,
        default=1480118400,
        help="Unix ts for val end (default 2016-11-26 00:00:00 UTC)",
    )
    ap.add_argument(
        "--window-seconds",
        type=int,
        default=86400,
        help="Time window half-width for negatives (default 86400 = ±1 day)",
    )
    ap.add_argument("--k", type=int, default=1, help="Negatives per positive (k)")
    ap.add_argument(
        "--make-val-test",
        action="store_true",
        help="Also create supervision_val/test with natural prevalence",
    )
    args = ap.parse_args()

    conn = sqlite3.connect(args.db, timeout=300)
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA synchronous=OFF")
    cur.execute("PRAGMA temp_store=MEMORY")
    cur.execute("PRAGMA cache_size=-200000")
    conn.commit()

    print("[INFO] Creating positives_train …")
    cur.executescript(
        f"""
    DROP TABLE IF EXISTS positives_train;
    CREATE TABLE positives_train AS
    SELECT id, link_id, subreddit, created_utc
    FROM comments
    WHERE is_hate = 1
      AND created_utc < {args.train_end};

    CREATE INDEX IF NOT EXISTS idx_positives_train_id   ON positives_train(id);
    CREATE INDEX IF NOT EXISTS idx_positives_train_link ON positives_train(link_id);
    """
    )
    conn.commit()

    cur.execute("SELECT COUNT(*) FROM positives_train")
    n_pos = cur.fetchone()[0]
    print(f"[INFO] positives_train: {n_pos:,} rows")

    print("[INFO] Building supervision_train with thread & time-matched negatives …")
    cur.executescript(
        """
    DROP TABLE IF EXISTS supervision_train;
    CREATE TABLE supervision_train (id TEXT PRIMARY KEY, label INTEGER NOT NULL);
    """
    )
    # insert positives
    cur.execute(
        "INSERT INTO supervision_train(id,label) SELECT id, 1 FROM positives_train"
    )
    conn.commit()

    # Build candidate negatives and sample k * n_pos of them
    cur.executescript(
        f"""
    DROP TABLE IF EXISTS _neg_cand;
    CREATE TEMP TABLE _neg_cand (id TEXT PRIMARY KEY);

    INSERT OR IGNORE INTO _neg_cand(id)
    SELECT DISTINCT c.id
    FROM positives_train p
    JOIN comments c
      ON c.link_id = p.link_id
     AND c.is_hate = 0
     AND c.created_utc BETWEEN p.created_utc - {args.window_seconds} AND p.created_utc + {args.window_seconds}
     AND c.body NOT IN ('[deleted]','[removed]','');

    """
    )
    conn.commit()

    cur.execute("SELECT COUNT(*) FROM _neg_cand")
    n_cand = cur.fetchone()[0]
    print(f"[INFO] candidate negatives found: {n_cand:,}")

    n_to_take = args.k * n_pos
    cur.execute(
        f"""
        INSERT OR IGNORE INTO supervision_train(id, label)
        SELECT id, 0 FROM (
            SELECT id FROM _neg_cand
            ORDER BY RANDOM()
            LIMIT {n_to_take}
        )
    """
    )
    conn.commit()

    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_supervision_train_id ON supervision_train(id)"
    )
    conn.commit()

    cur.execute("SELECT label, COUNT(*) FROM supervision_train GROUP BY label")
    rows = cur.fetchall()
    print("[INFO] supervision_train label counts:", {int(k): v for k, v in rows})

    if args.make_val_test:
        print("[INFO] Creating supervision_val/test (natural prevalence) …")
        cur.executescript(
            f"""
        DROP TABLE IF EXISTS supervision_val;
        CREATE TABLE supervision_val AS
        SELECT id, is_hate AS label
        FROM comments
        WHERE created_utc >= {args.train_end} AND created_utc < {args.val_end};

        DROP TABLE IF EXISTS supervision_test;
        CREATE TABLE supervision_test AS
        SELECT id, is_hate AS label
        FROM comments
        WHERE created_utc >= {args.val_end};

        CREATE INDEX IF NOT EXISTS idx_supervision_val_id  ON supervision_val(id);
        CREATE INDEX IF NOT EXISTS idx_supervision_test_id ON supervision_test(id);
        """
        )
        conn.commit()
        for table in ("supervision_val", "supervision_test"):
            cur.execute(
                f"SELECT COUNT(*), SUM(CASE WHEN label=1 THEN 1 ELSE 0 END) FROM {table}"
            )
            total, pos = cur.fetchone()
            print(f"[INFO] {table}: total={total:,}, positives={pos or 0:,}")

    conn.close()
    print("[OK] Done.")


if __name__ == "__main__":
    sys.exit(main())
