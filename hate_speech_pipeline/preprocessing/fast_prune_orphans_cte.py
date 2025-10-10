import sqlite3, sys

DB = sys.argv[1] if len(sys.argv) > 1 else "reddit_2016_11.db"

SQL = """
PRAGMA temp_store=MEMORY;
PRAGMA cache_size=-200000;

CREATE INDEX IF NOT EXISTS idx_comments_id     ON comments(id);
CREATE INDEX IF NOT EXISTS idx_comments_parent ON comments(parent_id);
CREATE INDEX IF NOT EXISTS idx_submissions_id  ON submissions(id);

CREATE TEMP TABLE keep_ids AS
WITH RECURSIVE
roots(id) AS (
  SELECT c.id
  FROM comments c
  WHERE c.parent_id IS NULL OR c.parent_id = ''
     OR c.parent_id IN (SELECT id FROM submissions)
),
chain(id) AS (
  SELECT id FROM roots
  UNION ALL
  SELECT c.id
  FROM comments c
  JOIN chain p ON c.parent_id = p.id
)
SELECT id FROM chain;

DELETE FROM comments
WHERE id NOT IN (SELECT id FROM keep_ids);

DROP TABLE keep_ids;
"""

def main():
    conn = sqlite3.connect(DB, timeout=300)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM comments")
    before = cur.fetchone()[0]
    print(f"[INFO] Comments before prune: {before}")

    cur.executescript(SQL)
    conn.commit()

    cur.execute("SELECT COUNT(*) FROM comments")
    after = cur.fetchone()[0]
    print(f"[INFO] Comments after prune:  {after}")
    print(f"[INFO] Deleted:               {before - after}")

    # Optional compact/defrag
    cur.execute("VACUUM")
    conn.commit()
    conn.close()
    print("[OK] One-shot prune complete.")

if __name__ == "__main__":
    main()
