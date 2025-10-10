import io
import json
import sqlite3
from pathlib import Path

import zstandard as zstd
from tqdm import tqdm

# -------------------- CONFIG --------------------
COMMENTS_ZSTS = [
    "/Users/simonloh/Downloads/reddit/comments/RC_2016-11.zst",
]
SUBS_ZSTS = [
    "/Users/simonloh/Downloads/reddit/submissions/RS_2016-11.zst",
]
DB_PATH = "reddit_2016_11.db"
BATCH_SIZE = 50_000
ZSTD_MAX_WINDOW = 2**31
# ------------------------------------------------


def apply_fast_pragmas(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA synchronous=OFF")
    cur.execute("PRAGMA temp_store=MEMORY")
    cur.execute("PRAGMA cache_size=-200000")
    cur.execute("PRAGMA foreign_keys=OFF")
    conn.commit()


def create_tables_no_indexes(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS submissions (
      id TEXT PRIMARY KEY,
      author TEXT,
      title TEXT,
      score INTEGER,
      subreddit TEXT,
      created_utc REAL
    )
    """
    )
    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS comments (
      id TEXT PRIMARY KEY,
      parent_id TEXT,
      link_id TEXT,
      author TEXT,
      body TEXT,
      score INTEGER,
      subreddit TEXT,
      created_utc REAL,
      is_hate INTEGER
    )
    """
    )
    conn.commit()


def create_indexes_after_load(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute("CREATE INDEX IF NOT EXISTS idx_comments_link     ON comments(link_id)")
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_comments_parent   ON comments(parent_id)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_comments_sub      ON comments(subreddit)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_comments_created  ON comments(created_utc)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_subs_sub          ON submissions(subreddit)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_subs_created      ON submissions(created_utc)"
    )
    conn.commit()
    try:
        cur.execute("ANALYZE")
        conn.commit()
    except sqlite3.OperationalError:
        pass
    try:
        cur.execute("VACUUM")
        conn.commit()
    except sqlite3.OperationalError:
        pass


def norm_pid(x):
    """Strip t1_/t3_ prefixes from parent_id/link_id."""
    if not isinstance(x, str):
        return x
    if x.startswith("t1_") or x.startswith("t3_"):
        return x.split("_", 1)[1]
    return x


def stream_zst_into_table(conn: sqlite3.Connection, zst_path: str, kind: str):
    """Stream a .zst file into submissions or comments table, committing per batch."""
    cur = conn.cursor()
    zst_path = str(zst_path)

    if kind == "subs":
        table = "submissions"
        fields = ["id", "author", "title", "score", "subreddit", "created_utc"]
    else:
        table = "comments"
        fields = [
            "id",
            "parent_id",
            "link_id",
            "author",
            "body",
            "score",
            "subreddit",
            "created_utc",
        ]

    placeholders = ",".join("?" * len(fields))
    sql = f"INSERT OR IGNORE INTO {table} ({','.join(fields)}) VALUES ({placeholders})"

    batch = []
    total = 0

    with open(zst_path, "rb") as fh:
        dctx = zstd.ZstdDecompressor(max_window_size=ZSTD_MAX_WINDOW)
        with dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(
                reader, encoding="utf-8", errors="ignore", newline=""
            )
            for line in tqdm(
                text_stream, desc=f"Streaming {Path(zst_path).name} -> {table}"
            ):
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except Exception:
                    continue

                if kind == "subs":
                    row = [
                        str(o.get("id") or o.get("name") or ""),
                        o.get("author"),
                        o.get("title"),
                        int(o.get("score") or 0),
                        o.get("subreddit"),
                        o.get("created_utc") or 0,
                    ]
                else:
                    row = [
                        str(o.get("id") or o.get("name") or ""),
                        norm_pid(o.get("parent_id") or o.get("parent") or ""),
                        norm_pid(o.get("link_id") or o.get("link") or ""),
                        o.get("author"),
                        o.get("body"),
                        int(o.get("score") or 0),
                        o.get("subreddit"),
                        o.get("created_utc") or 0,
                    ]

                batch.append(tuple(row))
                total += 1

                if len(batch) >= BATCH_SIZE:
                    cur.executemany(sql, batch)
                    conn.commit()
                    batch.clear()

    if batch:
        cur.executemany(sql, batch)
        conn.commit()

    return total


def main():
    conn = sqlite3.connect(DB_PATH)
    apply_fast_pragmas(conn)
    create_tables_no_indexes(conn)

    for z in SUBS_ZSTS:
        print(f"[INGEST] Submissions from: {z}")
        n = stream_zst_into_table(conn, z, "subs")
        print(f"[OK] {n} submission rows ingested from {z}")

    for z in COMMENTS_ZSTS:
        print(f"[INGEST] Comments from: {z}")
        n = stream_zst_into_table(conn, z, "comms")
        print(f"[OK] {n} comment rows ingested from {z}")

    print("[INDEX] Creating indexes …")
    create_indexes_after_load(conn)

    cur = conn.cursor()
    cur.execute("PRAGMA synchronous=NORMAL")
    conn.commit()

    conn.close()
    print(f"[DONE] Database built: {DB_PATH}")


if __name__ == "__main__":
    main()
