import argparse
import csv
import re
import sqlite3

T_TRAIN_END = 1479600000  # 2016-11-20 00:00:00 UTC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--db", required=True, help="Path to sqlite DB (absolute path recommended)"
    )
    ap.add_argument(
        "--csv", required=True, help="refined_ngram_dict.csv (Davidson et al.)"
    )
    ap.add_argument("--batch", type=int, default=100_000)
    ap.add_argument(
        "--col",
        default=None,
        help="Column name in CSV containing the n-gram (auto-detect if omitted)",
    )
    args = ap.parse_args()

    # 1) Load lexicon terms
    terms = []
    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        col = args.col
        if col is None:
            # try common headers, else first column
            for cand in ("ngram", "term", "token", "phrase", "lexeme", "ngram_str"):
                if cand in reader.fieldnames:
                    col = cand
                    break
            if col is None:
                col = reader.fieldnames[0]
        for row in reader:
            t = (row.get(col) or "").strip()
            if t:
                terms.append(t)

    # normalize/dedupe
    terms = sorted(set(t.lower() for t in terms if t.strip()))

    # Build a single regex: word-boundary; flexible whitespace for multiword ngrams
    def term_to_re(t: str) -> str:
        parts = [re.escape(p) for p in t.split()]
        return r"\b" + r"\s+".join(parts) + r"\b"

    patterns = [term_to_re(t) for t in terms]
    big_re = re.compile("|".join(patterns), flags=re.IGNORECASE) if patterns else None
    print(
        f"[INFO] Loaded {len(terms)} terms; regex length={len(big_re.pattern) if big_re else 0:,}"
    )

    # 2) Scan TRAIN comments and fill train_lex_pool
    conn = sqlite3.connect(args.db, timeout=300)
    cur = conn.cursor()
    cur.execute("PRAGMA temp_store=MEMORY")
    cur.execute("PRAGMA cache_size=-200000")
    cur.execute("CREATE TABLE IF NOT EXISTS train_lex_pool (id TEXT PRIMARY KEY)")
    conn.commit()

    cur.execute(
        """
        SELECT id, body
        FROM comments
        WHERE created_utc < ? AND body NOT IN ('[deleted]','[removed]','')
    """,
        (T_TRAIN_END,),
    )

    to_insert, seen, inserted = [], 0, 0
    while True:
        rows = cur.fetchmany(args.batch)
        if not rows:
            break
        for cid, body in rows:
            text = body or ""
            if big_re and big_re.search(text):
                to_insert.append((cid,))
        if to_insert:
            conn.executemany(
                "INSERT OR IGNORE INTO train_lex_pool(id) VALUES (?)", to_insert
            )
            conn.commit()
            inserted += len(to_insert)
            to_insert.clear()
        seen += len(rows)
        if seen % 1_000_000 == 0:
            print(f"[INFO] scanned={seen:,}  matches_inserted={inserted:,}")

    print(f"[DONE] scanned={seen:,}  matches_inserted={inserted:,} into train_lex_pool")
    conn.close()


if __name__ == "__main__":
    main()
