import logging
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

logging.basicConfig(
    level="INFO",
    format="%(asctime)s %(levelname)s %(module)s(%(lineno)d): %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_temporal_graph_local_diffusion(
    df: pd.DataFrame,
    author2idx: Dict[str, int],
    subreddit2idx: Dict[str, int],
    num_subreddits: int,
    comment_feature_names: List[str],
    user_feature_names: List[str],
    tox_thresh: float = 0.5,
) -> tuple:

    # Sorting
    if "timestamp" not in df.columns:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["created_utc"], unit="s", errors="coerce")

    if "time_bin" not in df.columns:
        raise ValueError(
            "Missing 'time_bin'. Ensure you've windowed timestamps before calling this."
        )

    # Sort to ensure temporal order inside bins is stable
    df = df.sort_values(["time_bin", "timestamp", "id"])

    # Ordered unique bins
    bin_order = np.array(sorted(df["time_bin"].unique()))

    # Precompute child lists per parent per bin
    # parent_id -> { tbin : [child_toxicity_prob, ...] }
    children_by_parent_by_bin: Dict[str, Dict[pd.Timestamp, List[float]]] = {}

    child_rows = df.dropna(subset=["parent_id"])[
        ["id", "parent_id", "time_bin", "toxicity_probability_self"]
    ]
    for pid, grp in child_rows.groupby("parent_id"):
        inner: Dict[pd.Timestamp, List[float]] = {}
        for tbin, g in grp.groupby("time_bin"):
            inner[tbin] = g["toxicity_probability_self"].astype(float).tolist()
        children_by_parent_by_bin[str(pid)] = inner

    def next_bin(tbin: pd.Timestamp) -> pd.Timestamp | None:
        idx = np.searchsorted(bin_order, tbin)
        if idx >= len(bin_order) - 1:
            return None
        return bin_order[idx + 1]

    # Outputs
    edge_index_list: List[np.ndarray] = []
    edge_weight_list: List[np.ndarray] = []
    features_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []
    masks_list: List[np.ndarray] = []
    time_vs_count: Dict[str, int] = {}

    # Build per-bin snapshots
    for tbin, group in df.groupby("time_bin"):
        tbin_next = next_bin(tbin)
        logger.debug(
            "Processing bin %s with %s comments; next bin: %s",
            tbin,
            len(group),
            tbin_next,
        )
        time_vs_count[str(tbin)] = len(group)

        # Nodes in this snapshot: comments posted in t
        comment_ids: List[str] = group["id"].astype(str).tolist()
        node_list: List[str] = comment_ids

        local_node2idx = {f"comment_{cid}": i for i, cid in enumerate(node_list)}
        num_nodes = len(node_list)

        # Mask: only nodes with a valid next window will be trained/evaluated
        mask = np.zeros(num_nodes, dtype=bool)

        # Feature dimension mirrors your original builder
        feature_dim = (
            1  # author_id (index)
            + len(user_feature_names)
            + 1  # score_f
            + 1  # body length
            + 1  # class_self as numeric feature
            + len(comment_feature_names)
            + 384  # body_emb
            + num_subreddits  # subreddit one-hot
        )

        x = np.zeros((num_nodes, feature_dim), dtype=np.float32)
        y = np.zeros((num_nodes, 1), dtype=np.float32)

        # Quick access to current rows
        group_rows = {str(row.id): row for row in group.itertuples(index=False)}

        # --Fill features and diffusion labels --
        for cid in node_list:
            row = group_rows[cid]
            idx_local = local_node2idx[f"comment_{cid}"]

            feat_offset = 0
            # author id idx
            x[idx_local, feat_offset] = float(author2idx.get(f"author_{row.author}", 0))
            feat_offset += 1

            # user features
            for uf in user_feature_names:
                x[idx_local, feat_offset] = float(getattr(row, uf))
                feat_offset += 1

            # score_f
            x[idx_local, feat_offset] = float(row.score_f)
            feat_offset += 1

            # body length
            x[idx_local, feat_offset] = float(len(str(row.body)))
            feat_offset += 1

            # class_self as a feature (0/1)
            cls = 1.0 if str(row.class_self) == "toxic" else 0.0
            x[idx_local, feat_offset] = cls
            feat_offset += 1

            # comment features
            for cf in comment_feature_names:
                x[idx_local, feat_offset] = float(getattr(row, cf))
            feat_offset += len(comment_feature_names)

            # body embedding (384-d)
            emb = np.array(getattr(row, "body_emb"), dtype=np.float32)
            if emb.size == 384:
                x[idx_local, feat_offset : feat_offset + 384] = emb
            feat_offset += 384

            # subreddit one-hot
            sub_idx = subreddit2idx[row.subreddit]
            x[idx_local, feat_offset + sub_idx] = 1.0
            feat_offset += num_subreddits

            # -Label: any toxic child in t+1? -
            if tbin_next is not None:
                tox_list_next = children_by_parent_by_bin.get(cid, {}).get(
                    tbin_next, []
                )
                if len(tox_list_next) == 0:
                    y[idx_local, 0] = 0.0
                    logger.debug("Comment %s has no children in next bin.", cid)
                else:
                    y[idx_local, 0] = float(any(t > tox_thresh for t in tox_list_next))
                    logger.debug(
                        "Comment %s has %d children in next bin; label = %.3f",
                        cid,
                        len(tox_list_next),
                        y[idx_local, 0],
                    )
                mask[idx_local] = True  # valid label only when t+1 exists
            else:
                logger.debug("Comment %s has no next bin; no label assigned.", cid)
                mask[idx_local] = False  # last bin has no label

        # --Edges within t (child -> parent if both in node_list) --
        edges: List[List[int]] = []
        edge_weights: List[float] = []

        # Build edges from replies where both child and parent are present in this t-bin
        # (No future leakage; optional to include additional historical context nodes)
        for cid in node_list:
            row = group_rows[cid]
            p = getattr(row, "parent_id")
            if pd.notna(p):
                pkey = f"comment_{str(p)}"
                if pkey in local_node2idx:
                    edges.append(
                        [local_node2idx[f"comment_{cid}"], local_node2idx[pkey]]
                    )
                    edge_weights.append(1.0)

        if len(edges) == 0:
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_weight = np.zeros((0,), dtype=np.float32)
        else:
            edge_index = np.array(edges, dtype=np.int64).T.copy()
            edge_weight = np.array(edge_weights, dtype=np.float32).copy()

        edge_index_list.append(edge_index)
        edge_weight_list.append(edge_weight)
        features_list.append(x)
        labels_list.append(y)
        masks_list.append(mask)

    logger.info(
        "%s nodes and %s edges built.",
        len(edge_index_list),
        sum(len(e) for e in edge_index_list),
    )
    return (
        edge_index_list,
        edge_weight_list,
        features_list,
        labels_list,
        time_vs_count,
        masks_list,
    )


def generate_comment_embeddings(
    df: pd.DataFrame, col_name: str, batch_size=96
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    emb_model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    comments = df[col_name].to_list()
    loader = DataLoader(comments, batch_size=batch_size)  # type: ignore

    all_embeddings = []
    for batch in tqdm(loader, desc="Generating embeddings"):
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True
        ).to(DEVICE)
        with torch.no_grad():
            outputs = emb_model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1)
        all_embeddings.append(emb.cpu())
    return torch.cat(all_embeddings).numpy()


def load_and_prepare_data(
    csv_path, window_size_hours, generate_embeddings, subset_count=500
):
    df = pd.read_csv(csv_path)
    if subset_count is not None and subset_count > 0:
        df = df.head(subset_count)

    # Drop NAs and filter subreddit without regex overhead (regex is slower)
    mask_valid = df["subreddit"].notna() & ~df["subreddit"].str.contains(" ")
    df = df[mask_valid & df["created_utc"].notna()]

    # Vectorized datetime conversion
    df["timestamp"] = pd.to_datetime(df["created_utc"], unit="s", errors="coerce")
    df = df[df["timestamp"].dt.year >= 2015]

    # Precompute time bins once
    df["time_bin"] = df["timestamp"].dt.floor(f"{window_size_hours}h")

    # Convert embeddings string to vector.
    if generate_embeddings:
        df["body_emb"] = list(generate_comment_embeddings(df, "body"))
        df.to_csv(f"{csv_path.replace('.csv', '')}_with_embeddings.csv", index=False)
    else:
        translation = str.maketrans({"[": "", "]": "", "\n": " "})
        cleaned = (
            df["body_emb"]
            .astype(str)
            .str.translate(translation)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        df["body_emb"] = cleaned.map(
            lambda s: np.fromstring(s, dtype=np.float32, sep=" ")
        )
    return df


def build_node_mappings(df):
    authors = df["author"].unique()
    subreddits = df["subreddit"].unique()
    author2idx = {}
    idx = 0
    for author in authors:
        author2idx[f"author_{author}"] = idx
        idx += 1
    subreddit2idx = {sub: idx for idx, sub in enumerate(subreddits)}
    return author2idx, subreddit2idx, len(subreddits)
