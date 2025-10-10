import argparse
import csv
import gc
import hashlib
import os
import sqlite3
import time

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
torch.set_grad_enabled(False)

inputs = {
    "train": "mine/supervision_train80_threads.csv",
    "valid": "mine/supervision_validation10_threads.csv",
    "test": "supervision_test10_threads.csv",
}
outputs = {
    "train": "retrain_train80.csv",
    "valid": "retrain_validation10.csv",
    "test": "retrain_test10.csv",
}

os.makedirs("tmp_retrain", exist_ok=True)
cache_db = "tmp_retrain/hatexplain_cache.sqlite"


def log(msg, on):
    if on:
        print(msg, flush=True)


def md5(s):
    return hashlib.md5(s.encode("utf-8", "ignore")).hexdigest()


def db_init():
    conn = sqlite3.connect(cache_db)
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS pred_cache (k TEXT PRIMARY KEY, p REAL)")
    conn.commit()
    return conn


def db_get_probs_chunked(conn, keys, chunk=900):
    cur = conn.cursor()
    out = {}
    for i in range(0, len(keys), chunk):
        ks = keys[i : i + chunk]
        q = "SELECT k,p FROM pred_cache WHERE k IN ({})".format(",".join("?" * len(ks)))
        for k, p in cur.execute(q, ks).fetchall():
            out[k] = p
    return out


def db_put_probs(conn, kv):
    if not kv:
        return
    cur = conn.cursor()
    cur.executemany(
        "INSERT OR REPLACE INTO pred_cache (k,p) VALUES (?,?)", list(kv.items())
    )
    conn.commit()


tokenizer = AutoTokenizer.from_pretrained("tum-nlp/bert-hateXplain")
model = AutoModelForSequenceClassification.from_pretrained(
    "tum-nlp/bert-hateXplain"
).to(device)
clf = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=0 if device == "cuda" else -1,
    dtype=torch.float16 if device == "cuda" else None,
)


def tox_from_labels(lbls):
    s = 0.0
    for d in lbls:
        L = str(d["label"]).lower()
        # Skip non-toxic/non-hate labels
        if (
            ("normal" in L)
            or ("neutral" in L)
            or ("non-toxic" in L)
            or ("non_hate" in L)
            or ("clean" in L)
        ):
            continue
        s += float(d["score"])
    return 0.0 if s < 0 else (1.0 if s > 1 else s)


def batched_predict(texts, bs=192, max_len=256, timing=False):
    t0 = time.time()
    conn = db_init()
    keys = [md5(t) for t in texts]
    out = [None] * len(texts)
    cached = db_get_probs_chunked(conn, keys)
    miss = [i for i, k in enumerate(keys) if k not in cached]
    for i, k in enumerate(keys):
        if k in cached:
            out[i] = cached[k]
    i = 0
    cur_bs = bs
    to_store = {}
    while i < len(miss):
        try:
            idxs = miss[i : i + cur_bs]
            batch = [texts[j] for j in idxs]
            res = clf(
                batch,
                batch_size=cur_bs,
                truncation=True,
                padding=True,
                max_length=max_len,
                top_k=None,
            )
            for j, labels in zip(idxs, res):
                p = tox_from_labels(labels)
                out[j] = p
                to_store[keys[j]] = p
            i += cur_bs
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) and cur_bs > 1:
                torch.cuda.empty_cache()
                cur_bs = max(1, cur_bs // 2)
            else:
                raise
    db_put_probs(conn, to_store)
    conn.close()
    for i, v in enumerate(out):
        if v is None:
            out[i] = 0.0
    if timing:
        print(f"infer({len(texts)}): {time.time()-t0:.2f}s", flush=True)
    return out


def read_df_cp775(path, nrows=None, bad_lines_mode="skip"):
    # Try different encodings based on the file

     = ["utf-8", "cp775", "latin-1"]
    df = None

    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(
                path,
                encoding=encoding,
                engine="python",
                sep=",",
                quoting=csv.QUOTE_NONE,
                escapechar="\\",
                on_bad_lines=bad_lines_mode,
                dtype=str,
                keep_default_na=False,
                nrows=nrows,
            )
            print(f"Successfully read {path} with encoding: {encoding}")
            break
        except (UnicodeDecodeError, UnicodeError):
            continue

    if df is None:
        raise ValueError(
            f"Could not read {path} with any of the attempted encodings: {encodings_to_try}"
        )
    cols = [
        "id",
        "parent_id",
        "link_id",
        "author",
        "body",
        "score",
        "subreddit",
        "created_utc",
        "is_hate",
        "hate_label",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    df = df[cols]
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0).astype(float)
    df["created_utc"] = (
        pd.to_numeric(df["created_utc"], errors="coerce").fillna(0).astype(int)
    )
    df = df.replace(
        {
            "id": {"": np.nan},
            "author": {"": np.nan},
            "body": {"": np.nan},
            "subreddit": {"": np.nan},
            "link_id": {"": np.nan},
        }
    )
    df = df.dropna(subset=["id", "author", "body", "subreddit", "link_id"]).reset_index(
        drop=True
    )
    return df


def compute_depth(df):
    pid = df.set_index("id")["parent_id"].to_dict()
    depth = {}

    def dpt(x):
        if x in depth:
            return depth[x]
        seen = set()
        cur = x
        d = 1
        while True:
            p = pid.get(cur, None)
            if p is None or p == cur or p not in pid:
                break
            if p in seen:
                break
            seen.add(p)
            cur = p
            d += 1
        depth[x] = d
        return d

    return df["id"].apply(dpt)


def build_user_feats_binary(df, cls_col="class_self"):
    g = df.groupby("author", dropna=False)
    uniq = g["subreddit"].nunique().rename("user_unique_subreddits")
    total = g.size().rename("user_total_comments")
    hate = g[cls_col].apply(lambda s: (s == "toxic").sum()).rename("user_hate_comments")
    ratio = (hate / total.replace(0, np.nan)).fillna(0).rename("user_hate_ratio")

    def mean_interval(s):
        v = np.sort(s.values)
        if len(v) < 2:
            return 0.0
        return float(np.mean(np.diff(v)))

    avg_int = g["created_utc"].apply(mean_interval).rename("user_avg_posting_interval")
    avg_tod = (
        g["created_utc"]
        .apply(lambda s: float(np.mean(s.values % 86400)) if len(s) > 0 else 0.0)
        .rename("user_avg_comment_time_of_day")
    )
    return pd.concat([uniq, total, hate, ratio, avg_int, avg_tod], axis=1).reset_index()


def build_user_feats_ordinal(df):
    # Try hate_label first, then is_hate as fallback
    label_col = None
    if "hate_label" in df.columns:
        label_col = "hate_label"
    elif "is_hate" in df.columns:
        label_col = "is_hate"
    else:
        return None

    ih = pd.to_numeric(df[label_col], errors="coerce")
    if not ih.notna().any():
        return None

    df2 = df.copy()
    df2["hate_ord"] = ih

    # Check if we have ordinal data (0-4) or binary data (0-1)
    unique_vals = set(ih.dropna().astype(int).unique())
    if {3, 4}.intersection(unique_vals):
        # Ordinal case: 0,1,2 are not hate. 3 and 4 are hate
        df2["hate_ord_is_hate"] = df2["hate_ord"].isin([3, 4]).astype(int)
    elif {1}.intersection(unique_vals):
        # Binary case: 1 is hate, 0 is not hate
        df2["hate_ord_is_hate"] = df2["hate_ord"].isin([1]).astype(int)
    else:
        return None

    g = df2.groupby("author", dropna=False)
    total = g.size().rename("user_total_comments")
    hate = g["hate_ord_is_hate"].sum().rename("user_hate_comments_ord")
    ratio = (hate / total.replace(0, np.nan)).fillna(0).rename("user_hate_ratio_ord")
    return pd.concat([hate, ratio], axis=1).reset_index()


def process_split(
    in_path,
    out_path,
    tau=0.5,
    bs=192,
    mode="full",
    n=1000,
    bad_lines_mode="skip",
    parent_policy="reuse",
    timing=False,
):
    log(
        f"[START] {in_path} mode={mode} n={n} bs={bs} tau={tau} parent={parent_policy}",
        timing,
    )
    t0 = time.time()
    nrows = None if mode == "full" else int(n)
    df = read_df_cp775(in_path, nrows=nrows, bad_lines_mode=bad_lines_mode)
    log(f"read_df: {df.shape} in {time.time()-t0:.2f}s", timing)
    t0 = time.time()
    tox_self = batched_predict(
        df["body"].astype(str).tolist(), bs=bs, max_len=256, timing=timing
    )
    df["toxicity_probability_self"] = tox_self
    df["class_self"] = np.where(
        df["toxicity_probability_self"] >= tau, "toxic", "non-toxic"
    )
    if parent_policy == "zero":
        prob_parent = pd.Series(0.0, index=df.index)
    elif parent_policy == "reuse":
        self_map = dict(zip(df["id"], df["toxicity_probability_self"]))
        idx = df["parent_id"].isin(self_map.keys())
        prob_parent = pd.Series(0.0, index=df.index)
        prob_parent.loc[idx] = df.loc[idx, "parent_id"].map(self_map).values
    else:
        body_map = df.set_index("id")["body"].to_dict()
        pid_present = df["parent_id"].isin(df["id"])
        parent_text = (
            df.loc[pid_present, "parent_id"].map(body_map).astype(str).tolist()
        )
        parent_probs = (
            batched_predict(parent_text, bs=bs, max_len=256, timing=timing)
            if len(parent_text) > 0
            else []
        )
        prob_parent = pd.Series(0.0, index=df.index)
        if len(parent_probs) > 0:
            prob_parent.loc[pid_present] = parent_probs
    df["toxicity_probability_parent"] = prob_parent.values
    log(f"parent_done in {time.time()-t0:.2f}s", timing)
    t0 = time.time()
    df["thread_depth"] = compute_depth(df)
    # Delete threads with depth < 3
    link_max = df.groupby("link_id")["thread_depth"].max()
    keep_links = set(link_max[link_max >= 3].index)
    log(f"Before depth filtering: {len(df)} rows", timing)
    df = df[df["link_id"].isin(keep_links)].reset_index(drop=True)
    log(f"After depth filtering: {len(df)} rows", timing)

    # Delete comments from subreddits which appear less than 5 times across the dataset
    sr_counts = df["subreddit"].value_counts()
    keep_srs = set(sr_counts[sr_counts >= 5].index)
    log(f"Before subreddit filtering: {len(df)} rows", timing)
    df = df[df["subreddit"].isin(keep_srs)].reset_index(drop=True)
    log(f"After subreddit filtering: {len(df)} rows", timing)

    # Check if we have any data left after filtering
    if len(df) == 0:
        log("Warning: No data left after filtering. Relaxing constraints...", True)
        # Reload data and use more lenient filtering
        df = read_df_cp775(in_path, nrows=nrows, bad_lines_mode=bad_lines_mode)
        tox_self = batched_predict(
            df["body"].astype(str).tolist(), bs=bs, max_len=256, timing=timing
        )
        df["toxicity_probability_self"] = tox_self
        df["class_self"] = np.where(
            df["toxicity_probability_self"] >= tau, "toxic", "non-toxic"
        )

        # Recompute parent toxicity
        if parent_policy == "zero":
            prob_parent = pd.Series(0.0, index=df.index)
        elif parent_policy == "reuse":
            self_map = dict(zip(df["id"], df["toxicity_probability_self"]))
            idx = df["parent_id"].isin(self_map.keys())
            prob_parent = pd.Series(0.0, index=df.index)
            prob_parent.loc[idx] = df.loc[idx, "parent_id"].map(self_map).values
        else:
            body_map = df.set_index("id")["body"].to_dict()
            pid_present = df["parent_id"].isin(df["id"])
            parent_text = (
                df.loc[pid_present, "parent_id"].map(body_map).astype(str).tolist()
            )
            parent_probs = (
                batched_predict(parent_text, bs=bs, max_len=256, timing=timing)
                if len(parent_text) > 0
                else []
            )
            prob_parent = pd.Series(0.0, index=df.index)
            if len(parent_probs) > 0:
                prob_parent.loc[pid_present] = parent_probs
        df["toxicity_probability_parent"] = prob_parent.values

        df["thread_depth"] = compute_depth(df)
        # Use more lenient filtering: depth >= 2 and subreddit >= 2
        link_max = df.groupby("link_id")["thread_depth"].max()
        keep_links = set(link_max[link_max >= 2].index)
        df = df[df["link_id"].isin(keep_links)].reset_index(drop=True)
        sr_counts = df["subreddit"].value_counts()
        keep_srs = set(sr_counts[sr_counts >= 2].index)
        df = df[df["subreddit"].isin(keep_srs)].reset_index(drop=True)
        log(f"After lenient filtering: {len(df)} rows", True)

    log(f"filtering done in {time.time()-t0:.2f}s", timing)
    t0 = time.time()
    created_map = df.set_index("id")["created_utc"].to_dict()
    score_map = df.set_index("id")["score"].to_dict()
    df["response_time"] = df.apply(
        lambda r: r["created_utc"] - created_map.get(r["parent_id"], 0), axis=1
    )
    df["score_f"] = df["score"].astype(float)
    scaler = StandardScaler()
    df["score_z"] = scaler.fit_transform(df[["score_f"]])
    qs = df["score_f"].quantile([0.2, 0.4, 0.6, 0.8]).values

    def bin5(x):
        if x <= qs[0]:
            return 0
        if x <= qs[1]:
            return 1
        if x <= qs[2]:
            return 2
        if x <= qs[3]:
            return 3
        return 4

    df["score_bin5"] = df["score_f"].apply(bin5)
    df["score_parent"] = df["parent_id"].map(score_map).fillna(0).astype(float)
    # Comment: Hate_score (self) = toxicity_probability_self x score_self
    df["hate_score_self"] = df["toxicity_probability_self"] * df["score_f"]
    # Comment: Hate_score (context) = toxicity_probability_parent x score_parent
    df["hate_score_ctx"] = df["toxicity_probability_parent"] * df["score_parent"]

    # User features
    user_bin = build_user_feats_binary(df, "class_self")
    df = df.merge(user_bin, on="author", how="left")
    user_ord = build_user_feats_ordinal(df)
    if user_ord is not None:
        df = df.merge(user_ord, on="author", how="left")
    dmy = pd.get_dummies(df["score_bin5"], prefix="scorebin")
    df = pd.concat([df, dmy], axis=1)
    cols = [
        "id",
        "parent_id",
        "link_id",
        "author",
        "subreddit",
        "created_utc",
        "body",
        "toxicity_probability_self",
        "class_self",
        "toxicity_probability_parent",
        "thread_depth",
        "score_f",
        "score_z",
        "score_bin5",
        "response_time",
        "score_parent",
        "hate_score_self",
        "hate_score_ctx",
        "user_unique_subreddits",
        "user_total_comments",
        "user_hate_comments",
        "user_hate_ratio",
        "user_avg_posting_interval",
        "user_avg_comment_time_of_day",
    ]
    # Add ordinal hate features if available (for ordinal categories 0,1,2=not hate; 3,4=hate)
    if "user_hate_comments_ord" in df.columns and "user_hate_ratio_ord" in df.columns:
        cols += ["user_hate_comments_ord", "user_hate_ratio_ord"]
    # Add one-hot encoded score bins
    cols += [c for c in df.columns if c.startswith("scorebin_")]
    df[cols].to_csv(out_path, index=False)
    log(f"saved -> {out_path} ({df.shape}) in {time.time()-t0:.2f}s", timing)
    del df
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "sample"], default="full")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    ap.add_argument("--bs", type=int, default=192)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--badlines", choices=["skip", "warn"], default="skip")
    ap.add_argument("--parent", choices=["zero", "reuse", "compute"], default="reuse")
    ap.add_argument("--timing", action="store_true")
    args = ap.parse_args()
    print(
        (
            f"Device set to use {device}:0"
            if device == "cuda"
            else f"Device set to use CPU"
        ),
        flush=True,
    )
    for sp in args.splits:
        if os.path.exists(inputs.get(sp, "")):
            process_split(
                inputs[sp],
                outputs[sp],
                tau=args.tau,
                bs=args.bs,
                mode=args.mode,
                n=args.n,
                bad_lines_mode=args.badlines,
                parent_policy=args.parent,
                timing=args.timing,
            )


if __name__ == "__main__":
    main()
