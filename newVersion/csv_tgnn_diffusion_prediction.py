
#!/usr/bin/env python3
"""TGNN-enhanced diffusion predictor supporting sampled datasets and comment-level supervision."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score
from tqdm import tqdm

# Optional TGNN dependency
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "scripts"))
try:
    import importlib.util

    tgnn_path = Path(__file__).parent.parent / "scripts" / "03_tgnn_model.py"
    spec = importlib.util.spec_from_file_location("tgnn_model", str(tgnn_path))
    tgnn_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tgnn_module)
    TGNNModel = tgnn_module.TGNNModel
    print("    [OK] Successfully imported TGNN model")
except Exception as exc:  # pragma: no cover - logging purpose only
    print(f"    [Warning] TGNN model import failed: {exc}")
    TGNNModel = None


class TGNNDiffusionPredictor:
    """CSV-based hate speech diffusion predictor with TGNN integration."""

    def __init__(
        self,
        data_dir: str | Path = "./",
        sample_config: Optional[Dict[str, float | int]] = None,
        window_size_comments: int = 8000,
        window_hours: int = 12,
        min_hate_per_window: int = 1,
        recall_topk: int = 200,
        hard_neg_per_pos: int = 5,
        tgnn_hidden: int = 128,
        tgnn_epochs: int = 8,
        mlp_hidden: int = 128,
        mlp_epochs: int = 3,
        mlp_batch_size: int = 2048,
        precision_half: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.precision_half = bool(precision_half and self.device.type == "cuda")
        torch.set_default_dtype(torch.float16 if self.precision_half else torch.float32)
        self.model_dtype = torch.float16 if self.precision_half else torch.float32
        self._cpu_dtype = torch.float32
        self.k_values = [1, 5, 10, 20]
        self.tgnn_model = None
        self.use_tgnn = TGNNModel is not None
        self.tgnn_module = None
        self.sample_config = sample_config or {}
        self.window_size_comments = window_size_comments
        self.window_hours = window_hours
        self.min_hate_per_window = min_hate_per_window
        self.recall_topk = recall_topk
        self.hard_neg_per_pos = hard_neg_per_pos
        self.tgnn_hidden = int(max(16, tgnn_hidden))
        self.tgnn_epochs = int(max(1, tgnn_epochs))
        self.mlp_hidden = int(max(32, mlp_hidden))
        self.mlp_epochs = int(max(1, mlp_epochs))
        self.mlp_batch_size = int(max(128, min(mlp_batch_size, 1024)))
        self.node_embedding_dim: Optional[int] = None
        self.text_vectorizer = None
        self.text_svd = None
        self.text_embedding_dim: Optional[int] = None
        self._text_fallback_ready = False
        self._text_cache: Dict[str, np.ndarray] = {}
        self._needs_refresh = False
        self.temperature_scale = 1.0
        self.author_hate_stats: Dict[str, List[int]] = {}
        self.subreddit_to_idx: Dict[str, int] = {}
        self.thread_to_idx: Dict[str, int] = {}
        self.edge_feature_dim: Optional[int] = None
        self._node_embeddings_tensor: Optional[torch.Tensor] = None
        self.validation_threshold: Optional[float] = None
        self._rng = np.random.default_rng(42)
        self._neighbor_fanouts: Optional[List[int]] = None
        self._neighbor_batch_size: Optional[int] = None
        self.cache_dir = self.data_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.user_history: Dict[str, deque] = defaultdict(deque)
        self.history_window_7 = pd.Timedelta(days=7)
        self.history_window_30 = pd.Timedelta(days=30)

        if self.use_tgnn:
            try:  # pragma: no cover
                import importlib.util

                tgnn_path = Path(__file__).parent.parent / "scripts" / "03_tgnn_model.py"
                spec = importlib.util.spec_from_file_location("tgnn_module", str(tgnn_path))
                self.tgnn_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(self.tgnn_module)
            except Exception:
                self.tgnn_module = None
                self.use_tgnn = False

        print(f"Using device: {self.device}")
        if self.precision_half and self.device.type == "cuda":
            print("    [OK] Half precision enabled (float16 model tensors)")
        if self.use_tgnn:
            print("    [OK] TGNN model available - will use TGNN for embeddings and prediction")
        else:
            print("    [Warning] TGNN model not available, using traditional methods")

    def _log_cuda_memory(self, tag: str) -> None:
        if torch.cuda.is_available():
            mem_mb = torch.cuda.max_memory_allocated() // (1024 ** 2)
            print(f"    [CUDA] {tag} max memory {mem_mb} MB")

    @staticmethod
    def _reset_cuda_peak() -> None:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def save_tgnn_cache(self) -> None:
        if self.node_embeddings is None:
            return
        chunk_size = 50000
        embeddings = self.node_embeddings
        dtype_str = str(embeddings.dtype)
        meta_path = self.cache_dir / "tgnn_embeddings_meta.json"
        # Clean previous chunks
        for path in self.cache_dir.glob("tgnn_embeddings_*.npy"):
            try:
                path.unlink()
            except OSError:
                pass
        chunk_files: List[str] = []
        for offset in range(0, embeddings.shape[0], chunk_size):
            chunk = embeddings[offset : offset + chunk_size]
            chunk_name = f"tgnn_embeddings_{offset // chunk_size:03d}.npy"
            np.save(self.cache_dir / chunk_name, chunk)
            chunk_files.append(chunk_name)
        node_order = [self.id_to_node[idx] for idx in range(len(self.id_to_node))]
        meta = {
            "node_embedding_dim": int(self.node_embedding_dim or 0),
            "dtype": dtype_str,
            "chunk_size": chunk_size,
            "chunk_files": chunk_files,
            "total_nodes": len(node_order),
        }
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump({"meta": meta, "nodes": node_order}, f)

    def load_tgnn_cache(self) -> bool:
        meta_path = self.cache_dir / "tgnn_embeddings_meta.json"
        if not meta_path.exists():
            return False
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"    [Warning] Failed to read TGNN cache metadata: {exc}")
            return False
        meta = data.get("meta", {})
        nodes = data.get("nodes")
        if not nodes:
            return False
        chunk_files = meta.get("chunk_files", [])
        arrays = []
        for chunk_name in chunk_files:
            path = self.cache_dir / chunk_name
            if not path.exists():
                print(f"    [Warning] Missing TGNN cache chunk {chunk_name}")
                return False
            arrays.append(np.load(path))
        if not arrays:
            return False
        embeddings = np.vstack(arrays)
        if embeddings.shape[0] != len(nodes):
            print("    [Warning] TGNN cache size mismatch")
            return False
        dtype = embeddings.dtype
        self.node_embeddings = embeddings
        self.node_embedding_dim = int(meta.get("node_embedding_dim") or embeddings.shape[1])
        self.node_to_id = {node: idx for idx, node in enumerate(nodes)}
        self.id_to_node = {idx: node for idx, node in enumerate(nodes)}
        self._invalidate_embedding_cache()
        print(
            f"    Loaded cached TGNN embeddings: {embeddings.shape}, dtype={dtype}, chunks={len(chunk_files)}"
        )
        return True

    def load_data(self) -> None:
        print("    Loading data from CSV files...")

        def _read_csv(path: Path) -> pd.DataFrame:
            print(f"    Loading {path.name} with utf-8...")
            try:
                return pd.read_csv(path, encoding="utf-8", low_memory=False)
            except UnicodeDecodeError:
                print("    UTF-8 failed, trying latin-1 encoding...")
                try:
                    return pd.read_csv(path, encoding="latin-1", low_memory=False)
                except Exception as exc:
                    print(f"    latin-1 standard read failed: {exc}")
            except pd.errors.ParserError as exc:
                print(f"    C-engine parse error: {exc}; retrying with python engine and skipping bad lines...")
            print("    Fallback: engine='python', on_bad_lines='skip', escapechar='\'")
            return pd.read_csv(
                path,
                encoding="latin-1",
                engine="python",
                on_bad_lines="skip",
                sep=",",
                quotechar='"',
                escapechar='\\',
                doublequote=True,
            )



        train_path = self.data_dir / "supervision_train80_threads.csv"
        raw_train = _read_csv(train_path)
        self.train_df = self._apply_sample(raw_train, "train")
        print(f"    Loaded training data: {len(self.train_df)} rows (raw {len(raw_train)})")

        val_path = self.data_dir / "supervision_validation10_threads.csv"
        raw_val = _read_csv(val_path)
        self.val_df = self._apply_sample(raw_val, "val")
        print(f"    Loaded validation data: {len(self.val_df)} rows (raw {len(raw_val)})")

        test_path = self.data_dir / "supervision_test10_threads.csv"
        raw_test = _read_csv(test_path)
        self.test_df = self._apply_sample(raw_test, "test")
        print(f"    Loaded test data: {len(self.test_df)} rows (raw {len(raw_test)})")

        self.all_df = pd.concat([self.train_df, self.val_df, self.test_df], ignore_index=True)
        print(f"    Total data: {len(self.all_df)} comments")

        self.preprocess_data()
        self.prepare_text_encoders()

    def _apply_sample(self, df: pd.DataFrame, split: str) -> pd.DataFrame:
        cfg = None
        if isinstance(self.sample_config, dict):
            cfg = self.sample_config.get(split)
        if cfg in (None, 0):
            return df
        df_sorted = df.sort_values("created_utc") if "created_utc" in df.columns else df
        try:
            if isinstance(cfg, (int, float)):
                if isinstance(cfg, float):
                    if cfg <= 0:
                        return df_sorted
                    count = max(1, int(len(df_sorted) * min(cfg, 1.0)))
                else:
                    count = max(1, min(int(cfg), len(df_sorted)))
                return df_sorted.iloc[:count].copy()
            if isinstance(cfg, dict):
                count = cfg.get("count")
                frac = cfg.get("frac")
                if count is not None:
                    count = max(1, min(int(count), len(df_sorted)))
                    return df_sorted.iloc[:count].copy()
                if frac is not None:
                    frac = max(0.0, min(float(frac), 1.0))
                    count = max(1, int(len(df_sorted) * frac))
                    return df_sorted.iloc[:count].copy()
        except Exception as exc:
            print(f"    [Warning] Sampling configuration for {split} failed: {exc}")
        return df_sorted

    def preprocess_data(self) -> None:
        df = self.all_df
        df["body"] = df["body"].fillna("")
        df["author"] = df["author"].fillna("[deleted]")
        df["hate_label"] = pd.to_numeric(df["hate_label"], errors="coerce").fillna(0)
        df["created_utc"] = pd.to_numeric(df["created_utc"], errors="coerce")
        before = len(df)
        df = df.dropna(subset=["created_utc"])
        if len(df) != before:
            print(f"    Dropped {before - len(df)} rows with invalid timestamps")
        df["created_utc"] = pd.to_datetime(df["created_utc"], unit="s")
        min_time = pd.Timestamp("2005-01-01")
        before = len(df)
        df = df[df["created_utc"] >= min_time]
        if len(df) != before:
            print(f"    Dropped {before - len(df)} rows earlier than {min_time.date()}")
        valid_mask = (
            (df["body"] != "[removed]")
            & (df["body"] != "[deleted]")
            & (df["body"] != "")
            & (df["author"] != "[deleted]")
        )
        df = df[valid_mask].reset_index(drop=True)
        print(f"    Selected {len(df)} valid comments")
        print(f"    Found {int(df['hate_label'].sum())} hate speech comments")
        df["id_norm"] = df["id"].apply(self._norm_cid)
        df["parent_norm"] = df["parent_id"].apply(self._norm_cid)
        df["link_norm"] = df["link_id"].apply(self._norm_cid)
        self.all_df = df

    def prepare_text_encoders(self) -> None:
        if hasattr(self, "text_vectorizer") and self.text_vectorizer is not None:
            return
        vectorizer_path = self.cache_dir / "tfidf_vectorizer.pkl"
        svd_path = self.cache_dir / "tfidf_svd.pkl"
        fitted = False
        if vectorizer_path.exists() and svd_path.exists():
            try:
                self.text_vectorizer = joblib.load(vectorizer_path)
                self.text_svd = joblib.load(svd_path)
                fitted = True
            except Exception as exc:
                print(f"    [Warning] Failed to load cached TF-IDF encoders: {exc}")
                self.text_vectorizer = None
                self.text_svd = None
        if not fitted:
            comments = self.all_df.get("body", pd.Series(dtype=str)).astype(str).tolist()
            if not comments:
                self.text_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), dtype=np.float32)
                self.text_svd = TruncatedSVD(n_components=256, random_state=42)
            else:
                print("    Fitting TF-IDF (max_features=10000, ngram_range=(1,2)) ...")
                self.text_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), dtype=np.float32)
                tfidf_matrix = self.text_vectorizer.fit_transform(comments)
                print("    Applying TruncatedSVD to 256 dimensions ...")
                self.text_svd = TruncatedSVD(n_components=256, random_state=42)
                self.text_svd.fit(tfidf_matrix)
                try:
                    joblib.dump(self.text_vectorizer, vectorizer_path)
                    joblib.dump(self.text_svd, svd_path)
                except Exception as exc:
                    print(f"    [Warning] Failed to cache text encoders: {exc}")
        self.text_embedding_dim = 256
        self._text_fallback_ready = True

    @staticmethod
    def _norm_cid(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        s = str(x)
        s = s.replace("comment_", "")
        if s.startswith("t1_") or s.startswith("t3_"):
            s = s.split("_", 1)[1]
        return s

    @staticmethod
    def _comment_key(x):
        norm = TGNNDiffusionPredictor._norm_cid(x)
        return None if norm is None else f"comment_{norm}"

    def build_graph(self) -> None:
        self.graph = nx.Graph()
        user_nodes = set()
        for _, row in self.all_df.iterrows():
            uid = f"user_{row['author']}"
            if uid not in user_nodes:
                self.graph.add_node(uid, node_type="user", name=row["author"])
                user_nodes.add(uid)
            cidk = self._comment_key(row["id_norm"])
            if cidk is None:
                continue
            self.graph.add_node(
                cidk,
                node_type="comment",
                text=row["body"],
                hate_label=int(row["hate_label"]),
                subreddit=row["subreddit"],
                created_utc=row["created_utc"],
            )
            self.graph.add_edge(uid, cidk, edge_type="authored")
        for _, row in self.all_df.iterrows():
            pc = self._comment_key(row["parent_norm"])
            cc = self._comment_key(row["id_norm"])
            lc = self._comment_key(row["link_norm"])
            if cc is None or pc is None or pc == lc:
                continue
            if self.graph.has_node(pc) and self.graph.has_node(cc):
                self.graph.add_edge(pc, cc, edge_type="reply")
        self.node_to_id = {node: idx for idx, node in enumerate(self.graph.nodes())}
        self.id_to_node = {idx: node for node, idx in self.node_to_id.items()}
        print(f"Graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

    def generate_embeddings(self) -> None:
        print("Generating node embeddings...")
        if self.use_tgnn:
            temporal_data = self.build_temporal_graph_data()
            if temporal_data is not None:
                tgnn_metrics = self.train_tgnn_model(temporal_data)
                if self.tgnn_model is not None:
                    embeddings = self.get_tgnn_embeddings(temporal_data)
                    if embeddings is not None:
                        dtype = np.float16 if self.device.type == "cuda" else np.float32
                        self.node_embeddings = embeddings.astype(dtype, copy=False)
                        self.tgnn_embeddings = self.node_embeddings
                        self.node_embedding_dim = self.node_embeddings.shape[1]
                        self._invalidate_embedding_cache()
                        self.temporal_data = temporal_data
                        self.tgnn_metrics = tgnn_metrics
                        self._text_fallback_ready = True
                        self.save_tgnn_cache()
                        self._log_cuda_memory("TGNN embeddings")
                        print(f"    Using TGNN embeddings: {self.node_embeddings.shape}")
                        return
        print("    Using TF-IDF embeddings as fallback...")
        self.generate_tfidf_embeddings()

    def generate_tfidf_embeddings(self) -> None:
        comment_texts: List[str] = []
        node_labels: List[int] = []
        for node in self.graph.nodes():
            data = self.graph.nodes[node]
            if data["node_type"] == "comment":
                comment_texts.append(data.get("text", ""))
                node_labels.append(data.get("hate_label", 0))
            else:
                comment_texts.append("")
                node_labels.append(0)
        tfidf = TfidfVectorizer(max_features=5000, stop_words="english", min_df=5)
        processed_texts = [txt if txt.strip() else "empty" for txt in comment_texts]
        tfidf_matrix = tfidf.fit_transform(processed_texts)
        user_features = []
        for node in self.graph.nodes():
            data = self.graph.nodes[node]
            if data["node_type"] == "user":
                user_comments = [n for n in self.graph.neighbors(node) if self.graph.nodes[n]["node_type"] == "comment"]
                hate_ratio = (
                    np.mean([self.graph.nodes[c].get("hate_label", 0) for c in user_comments])
                    if user_comments
                    else 0
                )
                features = [len(user_comments), hate_ratio, self.graph.degree(node)]
            else:
                features = [0, 0, 0]
            user_features.append(features)
        user_features_array = np.array(user_features)
        combined = np.hstack([tfidf_matrix.toarray(), user_features_array])
        pca = PCA(n_components=min(128, combined.shape[1]))
        self.node_embeddings = pca.fit_transform(combined)
        dtype = np.float16 if self.device.type == "cuda" else np.float32
        self.node_embeddings = self.node_embeddings.astype(dtype, copy=False)
        self.node_labels = np.array(node_labels)
        self.node_embedding_dim = self.node_embeddings.shape[1]
        self._invalidate_embedding_cache()
        self.text_vectorizer = tfidf
        self.text_svd = None
        self.text_embedding_dim = self.node_embedding_dim
        self._text_fallback_ready = True

    def _fit_text_fallback(self, texts: List[str]) -> None:
        self.prepare_text_encoders()

    def _embed_comment_text(self, text: str) -> Optional[torch.Tensor]:
        if not self._text_fallback_ready:
            self.prepare_text_encoders()
        norm_text = text if isinstance(text, str) and text.strip() else "empty"
        if norm_text in self._text_cache:
            dense = self._text_cache[norm_text]
        else:
            vec = self.text_vectorizer.transform([norm_text])
            if self.text_svd is not None:
                dense = self.text_svd.transform(vec)[0]
            else:
                dense = vec.toarray()[0]
            dense = dense.astype(np.float32, copy=False)
            if len(self._text_cache) < 50000:
                self._text_cache[norm_text] = dense
        target_dtype = self.model_dtype if self.precision_half and self.device.type == "cuda" else torch.float32
        embedding = torch.tensor(dense, dtype=torch.float32, device=self.device)
        if embedding.dtype != target_dtype:
            embedding = embedding.to(dtype=target_dtype)
        return embedding

    def _invalidate_embedding_cache(self) -> None:
        self._node_embeddings_tensor = None

    def _ensure_embedding_tensor(self) -> Optional[torch.Tensor]:
        if not hasattr(self, "node_embeddings") or self.node_embeddings is None:
            return None
        tensor = self._node_embeddings_tensor
        if tensor is not None:
            return tensor
        if isinstance(self.node_embeddings, torch.Tensor):
            tensor = self.node_embeddings.detach().to(self.device)
        else:
            base_dtype = self.model_dtype if self.precision_half and self.device.type == "cuda" else torch.float32
            tensor = torch.tensor(self.node_embeddings, dtype=base_dtype, device=self.device)
        target_dtype = self.model_dtype if self.precision_half and self.device.type == "cuda" else torch.float32
        if tensor.dtype != target_dtype:
            tensor = tensor.to(dtype=target_dtype)
        self._node_embeddings_tensor = tensor
        if self.node_embedding_dim is None and tensor is not None:
            self.node_embedding_dim = tensor.shape[1]
        return self._node_embeddings_tensor

    def _to_device_half(self, data):
        data = data.to(self.device)
        if not self.precision_half or self.device.type != "cuda":
            return data
        if hasattr(data, 'items'):
            for key, value in list(data.items()):
                if isinstance(value, torch.Tensor) and value.is_floating_point():
                    data[key] = value.half()
        elif isinstance(data, torch.Tensor) and data.is_floating_point():
            data = data.half()
        return data

    def _fallback_node_embedding(self, text: Optional[str] = None) -> np.ndarray:
        dim = self.node_embedding_dim or self.text_embedding_dim or 128
        embedding_tensor: Optional[torch.Tensor] = None
        if text and self._text_fallback_ready:
            embedding_tensor = self._embed_comment_text(text)
        target_dtype = self.model_dtype if self.precision_half and self.device.type == "cuda" else torch.float32
        if embedding_tensor is None or embedding_tensor.numel() == 0:
            embedding_tensor = torch.zeros(dim, dtype=target_dtype, device=self.device)
        if embedding_tensor.shape[0] < dim:
            pad = torch.zeros(
                dim - embedding_tensor.shape[0],
                dtype=embedding_tensor.dtype,
                device=embedding_tensor.device,
            )
            embedding_tensor = torch.cat([embedding_tensor, pad])
        elif embedding_tensor.shape[0] > dim:
            embedding_tensor = embedding_tensor[:dim]
        if embedding_tensor.dtype != target_dtype:
            embedding_tensor = embedding_tensor.to(dtype=target_dtype)
        arr = embedding_tensor.detach().cpu().numpy()
        return arr

    def _append_node_embedding_np(self, vector: np.ndarray) -> None:
        dtype = np.float16 if (self.precision_half and self.device.type == "cuda") else np.float32
        vector = vector.astype(dtype, copy=False).reshape(1, -1)
        if self.node_embedding_dim is None:
            self.node_embedding_dim = vector.shape[1]
        if self.node_embeddings is None or not isinstance(self.node_embeddings, np.ndarray):
            self.node_embeddings = vector
        else:
            self.node_embeddings = np.vstack([self.node_embeddings, vector])
        self._invalidate_embedding_cache()

    def _record_user_history(self, author: str, created_time, hate_label: int) -> None:
        if not isinstance(created_time, pd.Timestamp):
            created_time = pd.to_datetime(created_time)
        history = self.user_history[author]
        cutoff_30 = created_time - self.history_window_30
        while history and history[0][0] < cutoff_30:
            history.popleft()
        history.append((created_time, int(hate_label)))

    def _get_user_history_stats(self, author: str, current_time) -> Dict[str, float]:
        if not isinstance(current_time, pd.Timestamp):
            current_time = pd.to_datetime(current_time)
        history = self.user_history.get(author)
        if history is None:
            return {
                "count_7": 0.0,
                "hate_rate_7": 0.0,
                "count_30": 0.0,
                "hate_rate_30": 0.0,
            }
        cutoff_30 = current_time - self.history_window_30
        cutoff_7 = current_time - self.history_window_7
        count_30 = hate_30 = 0
        count_7 = hate_7 = 0
        while history and history[0][0] < cutoff_30:
            history.popleft()
        for ts, hate in history:
            count_30 += 1
            hate_30 += hate
            if ts >= cutoff_7:
                count_7 += 1
                hate_7 += hate
        hate_rate_7 = (hate_7 / count_7) if count_7 else 0.0
        hate_rate_30 = (hate_30 / count_30) if count_30 else 0.0
        return {
            "count_7": float(np.log1p(count_7)),
            "hate_rate_7": float(hate_rate_7),
            "count_30": float(np.log1p(count_30)),
            "hate_rate_30": float(hate_rate_30),
        }

    def _get_node_embedding(self, node_key: Optional[str], *, text: Optional[str] = None) -> torch.Tensor:
        tensor = self._ensure_embedding_tensor()
        target_dim = None
        if tensor is not None:
            target_dim = tensor.shape[1]
        elif self.node_embedding_dim is not None:
            target_dim = self.node_embedding_dim
        elif self.text_embedding_dim is not None:
            target_dim = self.text_embedding_dim
        else:
            target_dim = 128
            self.node_embedding_dim = target_dim
        if node_key is not None and tensor is not None:
            idx = self.node_to_id.get(node_key)
            if idx is not None and idx < tensor.shape[0]:
                return tensor[idx].float()
        if text is not None:
            fallback = self._embed_comment_text(text)
            if fallback is not None:
                if fallback.shape[0] < target_dim:
                    pad = torch.zeros(
                        target_dim - fallback.shape[0],
                        device=self.device,
                        dtype=fallback.dtype,
                    )
                    fallback = torch.cat([fallback, pad])
                elif fallback.shape[0] > target_dim:
                    fallback = fallback[:target_dim]
                return fallback.to(self.device)
        target_dtype = self.model_dtype if self.precision_half and self.device.type == "cuda" else torch.float32
        return torch.zeros(target_dim, dtype=target_dtype, device=self.device)

    def _compute_parent_similarity_feature(
        self,
        comment_embedding: torch.Tensor,
        parent_key: Optional[str],
        parent_text: Optional[str],
    ) -> float:
        if not parent_key:
            return 0.0
        parent_embedding = self._get_node_embedding(parent_key, text=parent_text)
        if parent_embedding is None:
            return 0.0
        if torch.sum(torch.abs(parent_embedding)).item() == 0 or torch.sum(torch.abs(comment_embedding)).item() == 0:
            return 0.0
        similarity = F.cosine_similarity(
            comment_embedding.unsqueeze(0), parent_embedding.unsqueeze(0), dim=1
        )
        return float(torch.clamp(similarity, min=-1.0, max=1.0).item())

    def _author_hate_rate(self, author: str) -> float:
        stats = self.author_hate_stats.get(author)
        if not stats:
            return 0.0
        total, hate = stats
        if total <= 0:
            return 0.0
        return float(hate) / float(total)

    def _update_author_hate_stats(self, author: str, hate_label: int) -> None:
        stats = self.author_hate_stats.setdefault(author, [0, 0])
        stats[0] += 1
        stats[1] += int(hate_label)

    def _build_edge_feature(
        self,
        *,
        author: str,
        comment_key: Optional[str],
        comment_text: str,
        created_utc,
        parent_key: Optional[str],
        window_start,
        subreddit: Optional[str],
        link_norm: Optional[str],
        is_reply: bool,
        user_stats: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        user_node = f"user_{author}"
        target_dtype = self.model_dtype if self.precision_half and self.device.type == "cuda" else torch.float32
        user_embedding = self._get_node_embedding(user_node).to(dtype=target_dtype)
        comment_tgnn = self._get_node_embedding(comment_key, text=comment_text).to(dtype=target_dtype)
        text_embedding = self._embed_comment_text(comment_text).to(dtype=target_dtype)
        tgnn_dim = min(self.node_embedding_dim or 128, 128)
        text_dim = self.text_embedding_dim or 256

        def _resize(vec: torch.Tensor, dim: int) -> torch.Tensor:
            target_dtype = vec.dtype if vec.is_floating_point() else (self.model_dtype if self.precision_half and self.device.type == "cuda" else torch.float32)
            if vec.shape[0] < dim:
                pad = torch.zeros(dim - vec.shape[0], dtype=target_dtype, device=self.device)
                vec = torch.cat([vec, pad])
            elif vec.shape[0] > dim:
                vec = vec[:dim]
            if vec.dtype != target_dtype:
                vec = vec.to(dtype=target_dtype)
            return vec

        user_embedding = _resize(user_embedding, tgnn_dim)
        comment_tgnn = _resize(comment_tgnn, tgnn_dim)
        text_embedding = _resize(text_embedding, text_dim)
        diff = torch.abs(user_embedding - comment_tgnn)
        prod = user_embedding * comment_tgnn
        parent_created = None
        if parent_key and parent_key in self.graph:
            parent_created = self.graph.nodes[parent_key].get("created_utc")
        created_dt = pd.to_datetime(created_utc)
        time_dtype = text_embedding.dtype if text_embedding.is_floating_point() else (self.model_dtype if self.precision_half and self.device.type == "cuda" else torch.float32)
        time_feats = torch.tensor(
            self._compute_time_features(
                user_node,
                created_utc,
                parent_created,
                window_start,
            ),
            dtype=time_dtype,
            device=self.device,
        )
        parent_text = None
        if parent_key and parent_key in self.graph:
            parent_text = self.graph.nodes[parent_key].get("text")
        parent_similarity = self._compute_parent_similarity_feature(comment_tgnn, parent_key, parent_text)
        parent_subreddit = None
        parent_link = None
        if parent_key and parent_key in self.graph:
            parent_data = self.graph.nodes[parent_key]
            parent_subreddit = parent_data.get("subreddit")
            parent_link = parent_data.get("link_norm")
        same_subreddit = 1.0 if parent_key is None or parent_subreddit is None or parent_subreddit == subreddit else 0.0
        same_thread = 1.0 if parent_key and parent_link and link_norm and parent_link == link_norm else 0.0
        delta_hours = 0.0
        if parent_created is not None:
            delta_seconds = max((created_dt - parent_created).total_seconds(), 0.0)
            delta_hours = delta_seconds / 3600.0
        meta_features = np.concatenate(
            [
                time_feats.cpu().numpy(),
                np.array(
                    [
                        float(is_reply),
                        same_subreddit,
                        same_thread,
                        float(np.log1p(delta_hours)),
                        parent_similarity,
                    ],
                    dtype=np.float32,
                ),
            ]
        )
        stats = user_stats or self._get_user_history_stats(author, created_dt)
        stats_vec = np.array(
            [
                stats.get("count_7", 0.0),
                stats.get("hate_rate_7", 0.0),
                stats.get("count_30", 0.0),
                stats.get("hate_rate_30", 0.0),
            ],
            dtype=np.float32,
        )
        meta_full = np.concatenate([meta_features, stats_vec])
        meta_tensor = torch.tensor(meta_full, dtype=torch.float32, device=self.device)
        if meta_tensor.dtype != text_embedding.dtype:
            meta_tensor = meta_tensor.to(dtype=text_embedding.dtype)
        subreddit_vec = torch.zeros(
            len(self.subreddit_to_idx), dtype=text_embedding.dtype, device=self.device
        )
        if subreddit in self.subreddit_to_idx:
            subreddit_vec[self.subreddit_to_idx[subreddit]] = 1.0
        thread_vec = torch.zeros(
            len(self.thread_to_idx), dtype=text_embedding.dtype, device=self.device
        )
        if link_norm in self.thread_to_idx:
            thread_vec[self.thread_to_idx[link_norm]] = 1.0
        comment_embedding = torch.cat([comment_tgnn, text_embedding, meta_tensor])
        feature_vector = torch.cat(
            [
                user_embedding,
                comment_embedding,
                diff,
                prod,
                subreddit_vec,
                thread_vec,
            ]
        )
        if self.edge_feature_dim is not None:
            if feature_vector.shape[0] < self.edge_feature_dim:
                pad = torch.zeros(
                    self.edge_feature_dim - feature_vector.shape[0],
                    device=self.device,
                    dtype=feature_vector.dtype,
                )
                feature_vector = torch.cat([feature_vector, pad])
            elif feature_vector.shape[0] > self.edge_feature_dim:
                feature_vector = feature_vector[: self.edge_feature_dim]
        target_dtype = self.model_dtype if self.precision_half and self.device.type == "cuda" else torch.float32
        if feature_vector.dtype != target_dtype:
            feature_vector = feature_vector.to(dtype=target_dtype)
        feature_vector = feature_vector.detach()
        if feature_vector.device.type != "cpu":
            feature_vector = feature_vector.cpu()
        return feature_vector

    def _prepare_edge_events(self, windows: List[Dict[str, Any]], split: str) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        for window in windows:
            window_start = window.get("start_time")
            if window_start is None:
                continue
            window_df = window["data"].sort_values("created_utc")
            for _, row in window_df.iterrows():
                author = row.get("author", "[unknown]")
                if pd.isna(author):
                    author = "[unknown]"
                comment_key = self._comment_key(row.get("id_norm"))
                parent_norm = row.get("parent_norm")
                if pd.isna(parent_norm):
                    parent_norm = None
                parent_key = self._comment_key(parent_norm)
                link_norm = row.get("link_norm")
                if pd.isna(link_norm):
                    link_norm = None
                subreddit = row.get("subreddit")
                if pd.isna(subreddit):
                    subreddit = None
                created = row.get("created_utc")
                is_reply = bool(parent_norm and parent_key)
                stats = self._get_user_history_stats(author, created)
                feature_vec = self._build_edge_feature(
                    author=author,
                    comment_key=comment_key,
                    comment_text=row.get("body", ""),
                    created_utc=created,
                    parent_key=parent_key,
                    window_start=window_start,
                    subreddit=subreddit,
                    link_norm=link_norm,
                    is_reply=is_reply,
                    user_stats=stats,
                )
                events.append(
                    {
                        "features": feature_vec,
                        "label": float(row.get("hate_label", 0)),
                        "author": author,
                        "subreddit": subreddit,
                        "link_norm": link_norm,
                        "is_reply": is_reply,
                        "window_id": window.get("window_id"),
                        "created_utc": created,
                        "split": split,
                    }
                )
                self._record_user_history(author, created, int(row.get("hate_label", 0)))
        return events

    def _select_hard_negative_indices(
        self,
        events: List[Dict[str, Any]],
        ratio: Optional[int] = None,
    ) -> List[int]:
        if ratio is None:
            ratio = self.hard_neg_per_pos or 5
        pos_indices = [idx for idx, ev in enumerate(events) if ev["label"] >= 1.0]
        neg_indices = [idx for idx, ev in enumerate(events) if ev["label"] < 1.0]
        if not pos_indices or not neg_indices:
            return list(range(len(events)))
        neg_by_thread: Dict[str, List[int]] = defaultdict(list)
        neg_by_sub: Dict[str, List[int]] = defaultdict(list)
        neg_by_author: Dict[str, List[int]] = defaultdict(list)
        for idx in neg_indices:
            ev = events[idx]
            if ev.get("link_norm"):
                neg_by_thread[str(ev["link_norm"])].append(idx)
            if ev.get("subreddit"):
                neg_by_sub[str(ev["subreddit"])].append(idx)
            if ev.get("author"):
                neg_by_author[str(ev["author"])].append(idx)
        selected: List[int] = []
        for pos_idx in pos_indices:
            selected.append(pos_idx)
            pos_event = events[pos_idx]
            bucket: List[int] = []
            if pos_event.get("link_norm"):
                bucket.extend(neg_by_thread.get(str(pos_event["link_norm"]), []))
            if pos_event.get("subreddit"):
                bucket.extend(neg_by_sub.get(str(pos_event["subreddit"]), []))
            if pos_event.get("author"):
                bucket.extend(neg_by_author.get(str(pos_event["author"]), []))
            if not bucket:
                bucket = neg_indices
            bucket = list(dict.fromkeys(bucket))  # preserve order, remove duplicates
            if not bucket:
                continue
            need = ratio
            replace = len(bucket) < need
            sampled = self._rng.choice(bucket, size=min(need, len(bucket)), replace=replace)
            for idx in np.atleast_1d(sampled):
                selected.append(int(idx))
        if not selected:
            return list(range(len(events)))
        self._rng.shuffle(selected)
        return selected


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits_fp32 = logits.float()
        targets_fp32 = targets.float()
        probs = torch.sigmoid(logits_fp32)
        ce_loss = F.binary_cross_entropy_with_logits(logits_fp32, targets_fp32, reduction="none")
        p_t = probs * targets_fp32 + (1 - probs) * (1 - targets_fp32)
        alpha_factor = self.alpha * targets_fp32 + (1 - self.alpha) * (1 - targets_fp32)
        focal_weight = alpha_factor * torch.pow(1 - p_t, self.gamma)
        loss = focal_weight * ce_loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

    def _recall_topk(self, candidates: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if not candidates:
            return [], {"total": 0, "selected": 0, "positives": 0, "positives_selected": 0}
        scored: List[Tuple[float, Dict[str, Any]]] = []
        total_pos = 0
        for cand in candidates:
            hate_label = int(cand.get("hate_label", 0))
            total_pos += hate_label
            author = cand.get("author", "[unknown]")
            created = cand.get("created_utc")
            user_node = f"user_{author}"
            h_user = self._get_node_embedding(user_node).float()
            parent_key = self._comment_key(cand.get("parent_norm"))
            parent_author_node = None
            parent_created = None
            parent_subreddit = None
            parent_thread = None
            if parent_key and parent_key in self.graph:
                parent_data = self.graph.nodes[parent_key]
                parent_author_node = parent_data.get("author_node")
                parent_created = parent_data.get("created_utc")
                parent_subreddit = parent_data.get("subreddit")
                parent_thread = parent_data.get("link_norm")
            if parent_author_node is None and parent_key and parent_key in self.graph:
                # fall back by inspecting neighbors
                neighbors = list(self.graph.neighbors(parent_key))
                for nbr in neighbors:
                    if self.graph.nodes[nbr].get("node_type") == "user":
                        parent_author_node = nbr
                        break
            if parent_author_node is None:
                parent_author_node = user_node
            h_parent = self._get_node_embedding(parent_author_node).float()
            if h_user.norm().item() == 0 or h_parent.norm().item() == 0:
                cosine = 0.0
            else:
                cosine = float(
                    F.cosine_similarity(h_user.unsqueeze(0), h_parent.unsqueeze(0), dim=1)
                    .clamp(min=-1.0, max=1.0)
                    .item()
                )
            stats = self._get_user_history_stats(author, created)
            hate_rate7 = stats.get("hate_rate_7", 0.0)
            subreddit = cand.get("subreddit")
            same_subreddit = 1.0 if parent_subreddit is None or parent_subreddit == subreddit else 0.0
            link_norm = cand.get("link_norm")
            same_thread = 1.0 if parent_thread and link_norm and parent_thread == link_norm else 0.0
            if parent_created is not None:
                delta_seconds = max((pd.to_datetime(created) - parent_created).total_seconds(), 0.0)
                delta_hours = delta_seconds / 3600.0
                time_decay = 1.0 / (1.0 + delta_hours)
            else:
                time_decay = 1.0
            base = max(cosine, 0.0)
            score = (
                base
                * (1.0 + hate_rate7)
                * time_decay
                * (same_subreddit if same_subreddit > 0 else 0.5)
                * (0.5 + 0.5 * same_thread)
            )
            cand["recall_score"] = float(score)
            scored.append((score, cand))
        scored.sort(key=lambda x: x[0], reverse=True)
        k = min(self.recall_topk, len(scored))
        selected_pairs = scored[:k]
        selected_candidates = [c for _, c in selected_pairs]
        positives_selected = sum(int(c.get("hate_label", 0)) for c in selected_candidates)
        print(
            "    Recall stage: total={} pos={} selected={} pos_selected={} coverage={:.2f}".format(
                len(candidates), total_pos, len(selected_candidates), positives_selected,
                (positives_selected / total_pos) if total_pos else 0.0,
            )
        )
        info = {
            "total": len(candidates),
            "selected": len(selected_candidates),
            "positives": total_pos,
            "positives_selected": positives_selected,
            "coverage": (positives_selected / total_pos) if total_pos else 0.0,
        }
        return selected_candidates, info

    def _fit_temperature_scale(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        if logits.numel() == 0 or labels is None or labels.numel() == 0:
            return 1.0
        if labels.sum().item() == 0:
            return 1.0
        logits_fp32 = logits.float()
        labels_fp32 = labels.float()
        log_temperature = torch.zeros(1, device=self.device, dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.Adam([log_temperature], lr=0.05)
        for _ in range(200):
            optimizer.zero_grad()
            temperature = torch.exp(log_temperature)
            loss = F.binary_cross_entropy_with_logits(logits_fp32 / temperature, labels_fp32)
            loss.backward()
            optimizer.step()
            log_temperature.data.clamp_(-5.0, 5.0)
        temperature = float(torch.exp(log_temperature).item())
        return float(np.clip(temperature, 0.05, 10.0))

    @staticmethod
    def _search_optimal_threshold(probs: np.ndarray, labels: np.ndarray) -> float:
        if probs.size == 0:
            return 0.5
        thresholds = np.linspace(0.05, 0.95, num=19)
        best_threshold = 0.5
        best_score = -1.0
        positives = float(np.sum(labels >= 1.0))
        for thresh in thresholds:
            preds = probs >= thresh
            tp = float(np.sum(preds & (labels >= 1.0)))
            fp = float(np.sum(preds & (labels < 1.0)))
            fn = positives - tp
            precision = tp / max(tp + fp, 1.0)
            recall = tp / max(positives, 1.0)
            if precision + recall == 0:
                score = 0.0
            else:
                score = 2 * precision * recall / (precision + recall)
            if score > best_score:
                best_score = score
                best_threshold = float(thresh)
        return best_threshold

    def build_temporal_graph_data(self):
        if not self.use_tgnn:
            return None
        node_features = []
        node_labels = []
        comment_texts = []
        for node in self.graph.nodes():
            data = self.graph.nodes[node]
            if data["node_type"] == "comment":
                comment_texts.append(data.get("text", ""))
                node_labels.append(min(int(data.get("hate_label", 0)), 1))
            else:
                comment_texts.append("")
                node_labels.append(0)
        tfidf = TfidfVectorizer(max_features=50, stop_words="english", min_df=1)
        processed = [txt if txt.strip() else "empty" for txt in comment_texts]
        matrix = tfidf.fit_transform(processed).toarray()
        for idx, node in enumerate(self.graph.nodes()):
            data = self.graph.nodes[node]
            features = [
                1.0 if data["node_type"] == "user" else 0.0,
                1.0 if data["node_type"] == "comment" else 0.0,
                float(self.graph.degree(node)),
                float(data.get("hate_label", 0)),
            ]
            node_features.append(np.concatenate([matrix[idx], features]))
        edges = []
        edge_time = []
        edge_attr = []
        for u, v in self.graph.edges():
            u_idx = self.node_to_id[u]
            v_idx = self.node_to_id[v]
            ts = None
            if self.graph.nodes[u]["node_type"] == "comment":
                ts = self.graph.nodes[u].get("created_utc")
            elif self.graph.nodes[v]["node_type"] == "comment":
                ts = self.graph.nodes[v].get("created_utc")
            timestamp = ts.timestamp() if isinstance(ts, pd.Timestamp) else pd.Timestamp.now().timestamp()
            edges.append([u_idx, v_idx])
            edge_time.append(timestamp)
            edge_attr.append([1.0])
        if not edges:
            return None
        from torch_geometric.data import TemporalData

        x = torch.tensor(node_features, dtype=torch.float32)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        edge_time = torch.tensor(edge_time, dtype=torch.float32)
        y = torch.tensor(node_labels, dtype=torch.long)
        temporal = TemporalData(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_time=edge_time, y=y)
        print(f"    Created temporal graph: {x.shape[0]} nodes, {edge_index.shape[1]} edges")
        return temporal

    @staticmethod
    def _temporal_to_static_data(temporal_data):
        """Convert TemporalData to a static Data object for NeighborLoader sampling."""
        try:
            from torch_geometric.data import Data
        except Exception as exc:
            raise RuntimeError(f"torch_geometric is required for TGNN neighbor sampling: {exc}")

        data_kwargs = {
            "x": temporal_data.x.cpu(),
            "edge_index": temporal_data.edge_index.cpu(),
            "y": temporal_data.y.cpu() if hasattr(temporal_data, "y") else None,
        }
        if hasattr(temporal_data, "edge_attr") and temporal_data.edge_attr is not None:
            data_kwargs["edge_attr"] = temporal_data.edge_attr.cpu()
        static_data = Data(**data_kwargs)
        if hasattr(temporal_data, "edge_time") and temporal_data.edge_time is not None:
            static_data.edge_time = temporal_data.edge_time.cpu()
        static_data.num_nodes = temporal_data.x.shape[0]
        return static_data

    def train_tgnn_model(self, temporal_data):
        if not self.use_tgnn or temporal_data is None:
            return None
        if TGNNModel is None:
            return None
        config = {
            "tgnn": {
                "input_dim": int(temporal_data.x.shape[1]),
                "hidden_dim": int(min(self.tgnn_hidden, 128)),
                "num_layers": 2,
                "num_classes": 2,
                "model_type": "TGAT",
                "dropout": 0.2,
                "learning_rate": 0.001,
            }
        }
        self.tgnn_model = TGNNModel(config).to(self.device)
        if self.precision_half and self.device.type == "cuda":
            self.tgnn_model.half()
        else:
            self.tgnn_model.float()

        try:
            from torch_geometric.loader import NeighborLoader
        except Exception as exc:
            print(
                "    [Warning] NeighborLoader unavailable ({}); falling back to full-batch TGNN training.".format(
                    exc
                )
            )
            return self._train_tgnn_fullbatch(temporal_data)

        self._reset_cuda_peak()
        graph_data = self._temporal_to_static_data(temporal_data)
        comment_entries = []
        for node, data in self.graph.nodes(data=True):
            if data.get("node_type") != "comment":
                continue
            node_idx = self.node_to_id.get(node)
            if node_idx is None:
                continue
            ts = data.get("created_utc")
            if isinstance(ts, pd.Timestamp):
                ts_float = ts.timestamp()
            elif ts is None:
                ts_float = 0.0
            else:
                ts_float = float(ts)
            comment_entries.append((node_idx, ts_float, ts))
        if not comment_entries:
            return None
        comment_entries.sort(key=lambda x: x[1])
        n_comments = len(comment_entries)
        train_end = max(1, int(n_comments * 0.7))
        val_end = max(train_end + 1, int(n_comments * 0.85))
        train_end = min(train_end, n_comments)
        val_end = min(val_end, n_comments)
        train_idx = [idx for idx, _, _ in comment_entries[:train_end]]
        val_idx = [idx for idx, _, _ in comment_entries[train_end:val_end]]
        test_idx = [idx for idx, _, _ in comment_entries[val_end:]]
        if not val_idx and len(test_idx) > 1:
            val_idx.append(test_idx.pop(0))
        if not test_idx and len(val_idx) > 1:
            test_idx.append(val_idx.pop())
        train_span = (
            comment_entries[0][2],
            comment_entries[min(train_end, n_comments) - 1][2],
        )
        val_span = None
        if val_idx:
            val_start_idx = train_end
            val_end_idx = train_end + len(val_idx) - 1
            val_span = (
                comment_entries[val_start_idx][2],
                comment_entries[val_end_idx][2],
            )
        test_span = None
        if test_idx:
            test_start_idx = val_end
            test_span = (
                comment_entries[test_start_idx][2],
                comment_entries[-1][2],
            )
        print(
            "    TGNN node split -> train: {} val: {} test: {}".format(
                len(train_idx), len(val_idx), len(test_idx)
            )
        )
        if train_span[0] is not None and train_span[1] is not None:
            print(f"        Train span {train_span[0]} -> {train_span[1]}")
        if val_span is not None and all(v is not None for v in val_span):
            print(f"        Val span   {val_span[0]} -> {val_span[1]}")
        if test_span is not None and all(v is not None for v in test_span):
            print(f"        Test span  {test_span[0]} -> {test_span[1]}")

        graph_data.y = graph_data.y.to(torch.long) if graph_data.y is not None else torch.zeros(graph_data.num_nodes, dtype=torch.long)
        fanout = 10
        num_layers = max(1, len(self.tgnn_model.tgnn_layers))
        fanouts = [fanout] * num_layers
        train_nodes = torch.tensor(train_idx, dtype=torch.long)
        val_nodes = torch.tensor(val_idx, dtype=torch.long) if val_idx else None

        batch_size = min(1024, max(256, len(train_idx) // 20))
        if len(train_idx) < 256:
            batch_size = max(64, len(train_idx))
        train_loader = NeighborLoader(
            graph_data,
            num_neighbors=fanouts,
            input_nodes=train_nodes,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
        )
        val_loader = None
        if val_nodes is not None and val_nodes.numel() > 0:
            val_loader = NeighborLoader(
                graph_data,
                num_neighbors=fanouts,
                input_nodes=val_nodes,
                batch_size=min(batch_size, max(128, val_nodes.numel())),
                shuffle=False,
                pin_memory=True,
            )

        optimizer = torch.optim.AdamW(self.tgnn_model.parameters(), lr=1e-3, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()
        best_loss = float("inf")
        patience = 0
        for epoch in range(self.tgnn_epochs):
            self.tgnn_model.train()
            optimizer.zero_grad()
            out = self.tgnn_model(temporal_cpu, task="classification")
            loss = criterion(out[train_mask].float(), temporal_cpu.y[train_mask])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.tgnn_model.parameters(), 1.0)
            optimizer.step()
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience = 0
            else:
                patience += 1
                if patience >= 5:
                    break
        print(f"    TGNN training completed with best loss: {best_loss:.4f}")
        try:
            self.graph_data = self._temporal_to_static_data(temporal_data.cpu())
        except Exception:
            self.graph_data = None
        self._log_cuda_memory("TGNN training (full batch)")
        return {"best_loss": float(best_loss)}
    def _train_tgnn_fullbatch(self, temporal_data):
        self._reset_cuda_peak()
        temporal_gpu = self._to_device_half(temporal_data)
        comment_entries = []
        for node, data in self.graph.nodes(data=True):
            if data.get("node_type") != "comment":
                continue
            node_idx = self.node_to_id.get(node)
            if node_idx is None:
                continue
            ts = data.get("created_utc")
            if isinstance(ts, pd.Timestamp):
                ts_float = ts.timestamp()
            elif ts is None:
                ts_float = 0.0
            else:
                ts_float = float(ts)
            comment_entries.append((node_idx, ts_float, ts))
        if not comment_entries:
            return None
        comment_entries.sort(key=lambda x: x[1])
        n_comments = len(comment_entries)
        train_end = max(1, int(n_comments * 0.7))
        val_end = max(train_end + 1, int(n_comments * 0.85))
        train_end = min(train_end, n_comments)
        val_end = min(val_end, n_comments)
        train_idx = [idx for idx, _, _ in comment_entries[:train_end]]
        val_idx = [idx for idx, _, _ in comment_entries[train_end:val_end]]
        test_idx = [idx for idx, _, _ in comment_entries[val_end:]]
        num_nodes = temporal_gpu.x.shape[0]
        device = self.device
        train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True
        temporal_gpu.train_mask = train_mask
        temporal_gpu.val_mask = val_mask
        temporal_gpu.test_mask = test_mask
        optimizer = torch.optim.AdamW(self.tgnn_model.parameters(), lr=1e-3, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()
        best_loss = float("inf")
        patience = 0
        for epoch in range(self.tgnn_epochs):
            self.tgnn_model.train()
            optimizer.zero_grad()
            out = self.tgnn_model(temporal_gpu, task="classification")
            loss = criterion(out[train_mask].float(), temporal_gpu.y[train_mask])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.tgnn_model.parameters(), 1.0)
            optimizer.step()
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience = 0
            else:
                patience += 1
                if patience >= 5:
                    break
        print(f"    TGNN training completed with best loss: {best_loss:.4f}")
        try:
            self.graph_data = self._temporal_to_static_data(temporal_data.cpu())
        except Exception:
            self.graph_data = None
        self._log_cuda_memory("TGNN training (full batch)")
        return {"best_loss": float(best_loss)}


    def _evaluate_tgnn_loader(self, loader, criterion):
        self.tgnn_model.eval()
        total_loss = 0.0
        total_examples = 0
        with torch.no_grad():
            for batch in loader:
                batch = self._to_device_half(batch)
                logits = self.tgnn_model(batch, task="classification")
                target = batch.y[: batch.batch_size].to(self.device)
                loss = criterion(logits[: batch.batch_size].float(), target)
                total_loss += loss.item() * target.size(0)
                total_examples += target.size(0)
        self.tgnn_model.train()
        return total_loss / max(1, total_examples)


    def get_tgnn_embeddings(self, temporal_data):
        if self.tgnn_model is None or temporal_data is None:
            return None
        target_dtype = (
            torch.float16 if self.precision_half and self.device.type == "cuda" else torch.float32
        )
        try:
            from torch_geometric.loader import NeighborLoader
        except Exception:
            self.tgnn_model.eval()
            with torch.no_grad():
                x = self.tgnn_model.input_projection(temporal_data.x.to(self.device))
                edge_index = temporal_data.edge_index.to(self.device)
                edge_attr = temporal_data.edge_attr.to(self.device) if temporal_data.edge_attr is not None else None
                edge_time = temporal_data.edge_time.to(self.device) if temporal_data.edge_time is not None else None
                for layer in self.tgnn_model.tgnn_layers:
                    x = layer(x, edge_index, edge_attr, edge_time)
                if x.dtype != target_dtype:
                    x = x.to(dtype=target_dtype)
                embeddings = x.cpu().numpy()
            print(f"    Generated TGNN embeddings shape: {embeddings.shape}")
            self._invalidate_embedding_cache()
            return embeddings

        graph_data = getattr(self, "graph_data", None)
        if graph_data is None:
            graph_data = self._temporal_to_static_data(temporal_data)
            self.graph_data = graph_data
        fanouts = self._neighbor_fanouts
        if fanouts is None:
            num_layers = max(1, len(self.tgnn_model.tgnn_layers))
            fanouts = [10] * num_layers
        batch_size = self._neighbor_batch_size or min(1024, max(256, graph_data.num_nodes // 20))
        seed_nodes = torch.arange(graph_data.num_nodes, dtype=torch.long)
        loader = NeighborLoader(
            graph_data,
            num_neighbors=fanouts,
            input_nodes=seed_nodes,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
        )
        embeddings: List[torch.Tensor] = []
        self.tgnn_model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = self._to_device_half(batch)
                node_repr = self.tgnn_model(batch, task=None)
                chunk = node_repr[: batch.batch_size]
                if chunk.dtype != target_dtype:
                    chunk = chunk.to(dtype=target_dtype)
                embeddings.append(chunk.cpu())
        if not embeddings:
            return None
        stacked = torch.cat(embeddings, dim=0)
        print(f"    Generated TGNN embeddings shape: {stacked.shape}")
        self._invalidate_embedding_cache()
        return stacked.numpy()


    def create_time_windows(
        self,
        window_hours: Optional[int] = None,
        min_hate_per_window: int = 1,
    ):
        print("Creating fixed-duration windows for global hate prediction...")
        if len(self.all_df) == 0:
            return [], [], []
        df = self.all_df.sort_values("created_utc").reset_index(drop=True)
        total = len(df)
        print(f"    Total comments: {total}")
        window_hours = window_hours or self.window_hours or 12
        window_delta = pd.Timedelta(hours=max(1, int(window_hours)))
        start_time = df["created_utc"].min().floor("H")
        end_time = start_time + window_delta
        windows: List[Dict[str, Any]] = []
        window_id = 0
        max_time = df["created_utc"].max()
        while start_time <= max_time:
            mask = (df["created_utc"] >= start_time) & (df["created_utc"] < end_time)
            subset = df.loc[mask]
            if not subset.empty:
                num_hate = int(subset["hate_label"].sum())
                window = {
                    "window_id": window_id,
                    "start_time": subset["created_utc"].min(),
                    "end_time": subset["created_utc"].max(),
                    "data": subset.copy(),
                    "num_comments": len(subset),
                    "num_hate": num_hate,
                }
                windows.append(window)
            window_id += 1
            start_time = end_time
            end_time = start_time + window_delta

        if not windows:
            print("    [Warning] No non-empty windows were generated")
            return [], [], []

        for w in windows:
            print(
                f"      Window {w['window_id']:02d}: "
                f"{w['start_time']} -> {w['end_time']} | comments={w['num_comments']} | hate={w['num_hate']}"
            )

        usable_windows = [w for w in windows if w["num_hate"] >= min_hate_per_window]
        dropped = len(windows) - len(usable_windows)
        if dropped:
            print(f"    Skipped {dropped} windows with fewer than {min_hate_per_window} hate comments")
        if len(usable_windows) < len(windows):
            if not usable_windows:
                print("    [Warning] All windows filtered out after hate threshold")
                return [], [], []
        else:
            usable_windows = windows

        total_windows = len(usable_windows)
        if total_windows < 3:
            print("    [Warning] Not enough windows for train/val/test split")
            return usable_windows, [], []

        val_count = 2 if total_windows >= 4 else 1
        test_count = min(4, max(2, total_windows - val_count - 1)) if total_windows >= val_count + 3 else max(1, total_windows - val_count - 1)
        train_count = total_windows - val_count - test_count
        if train_count < 1:
            deficit = 1 - train_count
            if test_count - deficit >= 2:
                test_count -= deficit
            else:
                val_count = max(1, val_count - deficit)
            train_count = total_windows - val_count - test_count
        if train_count < 1:
            train_count = max(1, total_windows - val_count - test_count)
            if train_count < 1:
                train_count = 1
                if val_count > 1:
                    val_count -= 1
                elif test_count > 1:
                    test_count -= 1

        train_windows = usable_windows[:train_count]
        val_windows = usable_windows[train_count : train_count + val_count]
        test_windows = usable_windows[train_count + val_count : train_count + val_count + test_count]
        self.validation_windows = val_windows

        def _summarize(windows: List[Dict[str, Any]], name: str) -> None:
            if not windows:
                print(f"    {name.capitalize()} windows: 0")
                return
            hate_list = [w["num_hate"] for w in windows]
            print(
                f"    {name.capitalize()} windows: {len(windows)} | avg comments {np.mean([w['num_comments'] for w in windows]):.1f} | "
                f"hate per window {hate_list}"
            )

        _summarize(train_windows, "train")
        _summarize(val_windows, "val")
        _summarize(test_windows, "test")
        # 贪心合并：确保val/test窗口有足够正样本
        def merge_windows_for_hate(windows, min_hate):
            if not windows:
                return []
            merged = []
            current = windows[0].copy()

            for i in range(1, len(windows)):
                if current['num_hate'] >= min_hate:
                    merged.append(current)
                    current = windows[i].copy()
                else:
                    # 合并下一个窗口
                    next_window = windows[i]
                    current['data'] = pd.concat([current['data'], next_window['data']], ignore_index=True)
                    current['num_comments'] += next_window['num_comments']
                    current['num_hate'] += next_window['num_hate']
                    current['end_time'] = next_window['end_time']

            # 处理最后一个窗口
            if current['num_hate'] >= min_hate:
                merged.append(current)
            elif merged:  # 合并到最后一个已有窗口
                merged[-1]['data'] = pd.concat([merged[-1]['data'], current['data']], ignore_index=True)
                merged[-1]['num_comments'] += current['num_comments']
                merged[-1]['num_hate'] += current['num_hate']
                merged[-1]['end_time'] = current['end_time']

            return merged

        val_windows = merge_windows_for_hate(val_windows, min_hate_per_window)
        test_windows = merge_windows_for_hate(test_windows, min_hate_per_window)

        # 确保至少有1个窗口
        if not train_windows or not val_windows or not test_windows:
            print("    Insufficient windows after merging, using fallback split...")
            if len(usable_windows) >= 3:
                train_windows = usable_windows[:-2]
                val_windows = [usable_windows[-2]]
                test_windows = [usable_windows[-1]]
            else:
                train_windows = usable_windows
                val_windows = []
                test_windows = []

        return train_windows, val_windows, test_windows

    def build_network_state(self, train_windows) -> None:
        print("    Building initial network state from training windows...")
        train_data = pd.concat([w["data"] for w in train_windows], ignore_index=True)
        train_data = train_data.sort_values("created_utc").reset_index(drop=True)
        self.graph = nx.Graph()
        self.author_hate_stats = {}
        self.user_history = defaultdict(deque)
        for _, row in train_data.iterrows():
            uid = f"user_{row['author']}"
            if not self.graph.has_node(uid):
                self.graph.add_node(uid, node_type="user", name=row["author"])
            cidk = self._comment_key(row["id_norm"])
            if cidk is None:
                continue
            self.graph.add_node(
                cidk,
                node_type="comment",
                text=row["body"],
                hate_label=int(row["hate_label"]),
                subreddit=row["subreddit"],
                created_utc=row["created_utc"],
                author=uid,
                author_node=uid,
                link_norm=row.get("link_norm"),
            )
            self.graph.add_edge(uid, cidk, edge_type="authored")
            self._update_author_hate_stats(row["author"], int(row["hate_label"]))
        for _, row in train_data.iterrows():
            pc = self._comment_key(row["parent_norm"])
            cc = self._comment_key(row["id_norm"])
            lc = self._comment_key(row["link_norm"])
            if cc is None or pc is None or pc == lc:
                continue
            if self.graph.has_node(pc) and self.graph.has_node(cc):
                self.graph.add_edge(pc, cc, edge_type="reply")
        self.node_to_id = {node: idx for idx, node in enumerate(self.graph.nodes())}
        self.id_to_node = {idx: node for node, idx in self.node_to_id.items()}
        self._reset_cuda_peak()
        self.generate_embeddings()
        self._needs_refresh = False

    def train_edge_mlp_from_windows(self, train_windows) -> None:
        print("    Training Edge-MLP from window data...")
        if self.tgnn_model is None and not self._text_fallback_ready:
            print("    No embeddings available, skipping Edge-MLP training")
            return
        subreddit_counts: Counter = Counter()
        thread_counts: Counter = Counter()
        for window in train_windows:
            data = window["data"].fillna({"subreddit": "", "link_norm": ""})
            subreddit_counts.update([str(x) for x in data["subreddit"] if str(x)])
            thread_counts.update([str(x) for x in data["link_norm"] if str(x)])
        max_subreddits = 32
        max_threads = 64
        self.subreddit_to_idx = {
            subreddit: idx for idx, (subreddit, _) in enumerate(subreddit_counts.most_common(max_subreddits))
        }
        self.thread_to_idx = {
            thread: idx for idx, (thread, _) in enumerate(thread_counts.most_common(max_threads))
        }
        self.edge_feature_dim = None
        train_events = self._prepare_edge_events(train_windows, split="train")
        if not train_events:
            print("    No valid training events for Edge-MLP")
            return
        max_events = 200_000
        if len(train_events) > max_events:
            positives = [ev for ev in train_events if ev["label"] >= 1.0]
            negatives = [ev for ev in train_events if ev["label"] < 1.0]
            keep_neg = max(max_events - len(positives), 0)
            if keep_neg < len(negatives):
                indices = self._rng.choice(len(negatives), size=keep_neg, replace=False)
                selected_negatives = [negatives[i] for i in indices]
            else:
                selected_negatives = negatives
            train_events = positives + selected_negatives
            self._rng.shuffle(train_events)
        self.edge_feature_dim = train_events[0]["features"].shape[0]
        val_windows = getattr(self, "validation_windows", []) or []
        val_events = self._prepare_edge_events(val_windows, split="val") if val_windows else []
        print(
            f"    Edge-MLP training events: {len(train_events)}, positives: "
            f"{int(sum(ev['label'] for ev in train_events))}"
        )
        if val_events:
            print(
                f"    Validation events: {len(val_events)}, positives: "
                f"{int(sum(ev['label'] for ev in val_events))}"
            )
        self._train_edge_mlp_direct(
            train_events, val_events, batch_size=self.mlp_batch_size, max_epochs=self.mlp_epochs
        )

    def _train_edge_mlp_direct(
        self,
        train_events: List[Dict[str, Any]],
        val_events: List[Dict[str, Any]],
        batch_size: Optional[int] = None,
        max_epochs: Optional[int] = None,
    ):
        self._reset_cuda_peak()
        if not train_events:
            return {"trained": False}
        target_dtype = (
            self.model_dtype if self.precision_half and self.device.type == "cuda" else torch.float32
        )
        train_features = torch.stack([ev["features"] for ev in train_events]).to(self.device)
        if train_features.dtype != target_dtype:
            train_features = train_features.to(dtype=target_dtype)
        train_labels = torch.tensor(
            [ev["label"] for ev in train_events], dtype=torch.float32, device=self.device
        )
        sampled_indices = self._select_hard_negative_indices(train_events, ratio=self.hard_neg_per_pos)
        index_tensor = torch.tensor(sampled_indices, dtype=torch.long, device=self.device)
        train_features = train_features[index_tensor]
        train_labels = train_labels[index_tensor]
        input_dim = train_features.shape[1]
        self.edge_feature_dim = input_dim
        self._init_edge_mlp(input_dim)
        criterion = FocalLoss(alpha=0.25, gamma=2.0).to(self.device)
        optimizer = torch.optim.AdamW(self.edge_mlp.parameters(), lr=1e-3, weight_decay=0.01)
        val_features = None
        val_labels = None
        if val_events:
            val_features = torch.stack([ev["features"] for ev in val_events]).to(self.device)
            if val_features.dtype != target_dtype:
                val_features = val_features.to(dtype=target_dtype)
            val_labels = torch.tensor(
                [ev["label"] for ev in val_events], dtype=torch.float32, device=self.device
            )
        best_auc = 0.0
        patience = 0
        dataset_size = train_features.shape[0]
        batch_size = batch_size or self.mlp_batch_size
        effective_batch = min(batch_size, dataset_size)
        max_epochs = max_epochs or self.mlp_epochs
        for epoch in range(max_epochs):
            self.edge_mlp.train()
            perm = torch.randperm(dataset_size, device=self.device)
            total_loss = 0.0
            for i in range(0, dataset_size, effective_batch):
                batch_idx = perm[i : i + effective_batch]
                feats = train_features[batch_idx]
                labels = train_labels[batch_idx].to(dtype=feats.dtype)
                logits = self.edge_mlp(feats).squeeze(-1)
                loss = criterion(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.edge_mlp.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item() * len(batch_idx)
            val_auc = None
            if val_features is not None and val_features.shape[0] > 0:
                self.edge_mlp.eval()
                with torch.no_grad():
                    val_logits = self.edge_mlp(val_features).squeeze(-1)
                    val_probs = torch.sigmoid(val_logits).float().cpu().numpy().astype(np.float32)
                    val_truth = val_labels.cpu().numpy().astype(np.float32)
                try:
                    val_auc = average_precision_score(val_truth, val_probs)
                except Exception:
                    val_auc = 0.0
                print(
                    f"      Edge-MLP epoch {epoch}: loss={total_loss / dataset_size:.4f}, valAP={val_auc:.3f}"
                )
                if val_auc > best_auc + 1e-3:
                    best_auc = val_auc
                    patience = 0
                else:
                    patience += 1
            else:
                print(f"      Edge-MLP epoch {epoch}: loss={total_loss / dataset_size:.4f}")
                patience = 0
            if patience >= 3:
                break
        if val_features is not None and val_features.shape[0] > 0:
            self.edge_mlp.eval()
            with torch.no_grad():
                val_logits = self.edge_mlp(val_features).squeeze(-1)
            self.temperature_scale = self._fit_temperature_scale(val_logits, val_labels)
            with torch.no_grad():
                calibrated = torch.sigmoid(val_logits / max(self.temperature_scale, 1e-6)).cpu().numpy().astype(np.float32)
            try:
                self.validation_threshold = self._search_optimal_threshold(
                    calibrated, val_labels.cpu().numpy().astype(np.float32)
                )
                print(
                    f"      Calibrated temperature={self.temperature_scale:.3f}, "
                    f"validation threshold={self.validation_threshold:.3f}"
                )
            except Exception:
                self.validation_threshold = 0.5
        else:
            self.temperature_scale = 1.0
            self.validation_threshold = 0.5
        self._log_cuda_memory("Edge-MLP training")
        return {"trained": True, "best_valAP": float(best_auc)}
    def _init_edge_mlp(self, in_dim: int) -> None:
        hidden = int(min(self.mlp_hidden, 128))
        layers = [
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 1),
        ]
        self.edge_mlp = nn.Sequential(*layers).to(self.device)
        if self.precision_half and self.device.type == "cuda":
            self.edge_mlp = self.edge_mlp.half()
        else:
            self.edge_mlp = self.edge_mlp.float()

    def _compute_time_features(self, user_node, current_time, parent_time=None, window_start=None):
        current_dt = pd.to_datetime(current_time)
        features: List[float] = []
        if parent_time is not None:
            delta = (current_dt - parent_time).total_seconds()
            features.append(np.log1p(max(delta, 0.0)))
        else:
            features.append(0.0)
        if window_start is not None:
            progress = (current_dt - window_start).total_seconds() / (24 * 3600)
            features.append(float(np.clip(progress, 0.0, 1.0)))
        else:
            features.append(0.5)
        if user_node in self.graph:
            timestamps = []
            for neighbor in self.graph.neighbors(user_node):
                node_data = self.graph.nodes[neighbor]
                if node_data.get("node_type") == "comment":
                    ts = node_data.get("created_utc")
                    if ts is not None and ts < current_dt:
                        timestamps.append(ts)
            if len(timestamps) >= 2:
                timestamps.sort()
                gaps = [
                    (timestamps[i] - timestamps[i - 1]).total_seconds()
                    for i in range(1, min(len(timestamps), 5))
                ]
                features.append(np.log1p(np.mean(gaps)))
            else:
                features.append(10.0)
        else:
            features.append(10.0)
        features.append(current_dt.hour / 24.0)
        features.append(float(current_dt.weekday() >= 5))
        return np.array(features, dtype=np.float32)


    def predict_window_hate(self, candidates):
        if not hasattr(self, "edge_mlp") or self.edge_mlp is None:
            print("    Edge-MLP not available, using fallback scoring...")
            return self._fallback_hate_scoring(candidates)
        if getattr(self, "_needs_refresh", False):
            self.refresh_temporal_state()
            self._needs_refresh = False
        if self.edge_feature_dim is None:
            print("    Edge feature dimension unknown, falling back")
            return self._fallback_hate_scoring(candidates)
        feature_batches: List[torch.Tensor] = []
        for candidate in candidates:
            comment_key = self._comment_key(candidate.get("id_norm"))
            parent_key = self._comment_key(candidate.get("parent_norm"))
            feature_vec = self._build_edge_feature(
                author=candidate.get("author", "[unknown]"),
                comment_key=comment_key,
                comment_text=candidate.get("body", ""),
                created_utc=candidate.get("created_utc"),
                parent_key=parent_key,
                window_start=candidate.get("window_start"),
                subreddit=candidate.get("subreddit"),
                link_norm=candidate.get("link_norm"),
                is_reply=bool(candidate.get("is_reply", False)),
            ).unsqueeze(0)
            feature_batches.append(feature_vec)
        if not feature_batches:
            return []
        batch = torch.cat(feature_batches, dim=0)
        self.edge_mlp.eval()
        with torch.no_grad():
            logits = self.edge_mlp(batch).squeeze(-1)
            temperature = self.temperature_scale if self.temperature_scale > 0 else 1.0
            probs = torch.sigmoid(logits / temperature).detach().cpu().numpy()
        return [float(p) for p in probs]

    def _fallback_hate_scoring(self, candidates):
        preds = []
        for cand in candidates:
            user_node = f"user_{cand['author']}"
            if user_node not in self.node_to_id or not hasattr(self, "node_embeddings"):
                preds.append(0.1)
                continue
            uid = self.node_to_id[user_node]
            emb = self.node_embeddings[uid]
            preds.append(float(np.tanh(emb.mean()) * 0.5 + 0.5))
        return preds

    def update_network_state(self, window):
        for _, row in window["data"].iterrows():
            uid = f"user_{row['author']}"
            cidk = self._comment_key(row["id_norm"])
            if uid not in self.graph:
                self.graph.add_node(uid, node_type="user", name=row["author"])
            if uid not in self.node_to_id:
                new_idx = len(self.node_to_id)
                self.node_to_id[uid] = new_idx
                self.id_to_node[new_idx] = uid
                vector = self._fallback_node_embedding()
                self._append_node_embedding_np(vector)
            if cidk and cidk not in self.graph:
                self.graph.add_node(
                    cidk,
                    node_type="comment",
                    text=row["body"],
                    hate_label=int(row["hate_label"]),
                    subreddit=row["subreddit"],
                    created_utc=row["created_utc"],
                    author=uid,
                    author_node=uid,
                    link_norm=row.get("link_norm"),
                )
            if cidk and cidk not in self.node_to_id:
                new_idx = len(self.node_to_id)
                self.node_to_id[cidk] = new_idx
                self.id_to_node[new_idx] = cidk
                vector = self._fallback_node_embedding(row.get("body", ""))
                self._append_node_embedding_np(vector)
                self.graph.add_edge(uid, cidk, edge_type="authored")
                self._update_author_hate_stats(row["author"], int(row["hate_label"]))
            parent = self._comment_key(row["parent_norm"])
            if cidk and parent and self.graph.has_node(parent):
                self.graph.add_edge(parent, cidk, edge_type="reply")
            self._record_user_history(row.get("author", "[unknown]"), row.get("created_utc"), int(row.get("hate_label", 0)))
        self._needs_refresh = False
        self._invalidate_embedding_cache()

    def refresh_temporal_state(self) -> None:
        self._invalidate_embedding_cache()

    def extract_candidate_comments(self, window):
        candidates: List[Dict[str, Any]] = []
        replies = 0
        window_start = window.get("start_time")
        for _, row in window["data"].iterrows():
            parent_norm = row.get("parent_norm")
            if pd.isna(parent_norm):
                parent_norm = None
            link_norm = row.get("link_norm")
            if pd.isna(link_norm):
                link_norm = None
            parent_id = row.get("parent_id")
            is_reply = bool(parent_norm and parent_norm != link_norm and pd.notna(parent_id))
            subreddit = row.get("subreddit")
            if pd.isna(subreddit):
                subreddit = None
            author = row.get("author")
            if pd.isna(author):
                author = "[unknown]"
            body = row.get("body", "")
            if pd.isna(body):
                body = ""
            if is_reply:
                replies += 1
            candidates.append(
                {
                    "id": row.get("id"),
                    "id_norm": row.get("id_norm"),
                    "parent_id": parent_id,
                    "parent_norm": parent_norm,
                    "author": author,
                    "body": body,
                    "hate_label": int(row.get("hate_label", 0)),
                    "subreddit": subreddit,
                    "created_utc": row.get("created_utc"),
                    "link_norm": link_norm,
                    "is_reply": is_reply,
                    "window_start": window_start,
                }
            )
        window["num_candidates"] = len(candidates)
        print(
            f"    Candidates: {len(candidates)} total | replies {replies} | top-level {len(candidates) - replies}"
        )
        return candidates

    def evaluate_global_hate_prediction(self, window_records: List[Dict[str, Any]]):
        if not window_records:
            empty_hit = {k: 0.0 for k in [1, 5, 10, 20]}
        return {
                "global_hate": {
                    "pr_auc": 0.0,
                    "window_pr_auc": 0.0,
                    "hit": empty_hit,
                    "total_predictions": 0,
                    "total_positives": 0,
                    "positive_rate": 0.0,
                    "temperature_scale": float(self.temperature_scale),
                    "validation_threshold": float(self.validation_threshold) if self.validation_threshold is not None else None,
                }
            }
        all_predictions = []
        all_truth = []
        window_pr_aucs: List[float] = []
        hit_results = {k: [] for k in [1, 5, 10, 20]}
        window_f1_scores: List[float] = []
        recall_coverages: List[float] = []
        for record in window_records:
            preds = np.array(record["predictions"], dtype=np.float32)
            truth = np.array(record["truth"], dtype=np.int32)
            if preds.size == 0:
                continue
            all_predictions.append(preds)
            all_truth.append(truth)
            positives = truth.sum()
            if positives > 0:
                try:
                    window_auc = average_precision_score(truth, preds)
                except Exception:
                    window_auc = 0.0
                window_pr_aucs.append(window_auc)
            order = np.argsort(-preds)
            for k in [1, 5, 10, 20]:
                topk = order[: min(k, order.size)]
                if positives > 0:
                    hits = truth[topk].sum()
                    hit_results[k].append(float(hits / positives))
                else:
                    hit_results[k].append(0.0)
            K = max(5, int(2 * positives)) if positives > 0 else 5
            if order.size > 0:
                topk_f1 = order[: min(K, order.size)]
                tp = truth[topk_f1].sum()
                fp = len(topk_f1) - tp
                fn = positives - tp
                denom = 2 * tp + fp + fn
                f1 = float((2 * tp) / denom) if denom > 0 else 0.0
            else:
                f1 = 0.0
            window_f1_scores.append(f1)
            recall_info = record.get("recall", {}) or {}
            if "coverage" in recall_info:
                recall_coverages.append(float(recall_info.get("coverage", 0.0)))
        if not all_predictions:
            empty_hit = {k: 0.0 for k in [1, 5, 10, 20]}
        return {
                "global_hate": {
                    "pr_auc": 0.0,
                    "window_pr_auc": 0.0,
                    "hit": empty_hit,
                    "total_predictions": 0,
                    "total_positives": 0,
                    "positive_rate": 0.0,
                    "temperature_scale": float(self.temperature_scale),
                    "validation_threshold": float(self.validation_threshold) if self.validation_threshold is not None else None,
                }
            }
        global_preds = np.concatenate(all_predictions)
        global_preds = np.nan_to_num(global_preds, nan=0.0, posinf=1.0, neginf=0.0)
        global_truth = np.concatenate(all_truth)
        positives = int(global_truth.sum())
        pr_auc = average_precision_score(global_truth, global_preds) if positives > 0 else 0.0
        hit_avg = {k: float(np.mean(values)) if values else 0.0 for k, values in hit_results.items()}
        avg_window_pr_auc = float(np.mean(window_pr_aucs)) if window_pr_aucs else 0.0
        avg_f1 = float(np.mean(window_f1_scores)) if window_f1_scores else 0.0
        avg_recall_cov = float(np.mean(recall_coverages)) if recall_coverages else 0.0
        return {
            "global_hate": {
                "pr_auc": float(pr_auc),
                "window_pr_auc": avg_window_pr_auc,
                "hit": hit_avg,
                "total_predictions": int(global_preds.size),
                "total_positives": int(positives),
                "positive_rate": float(positives / global_preds.size) if global_preds.size else 0.0,
                "temperature_scale": float(self.temperature_scale),
                "validation_threshold": float(self.validation_threshold) if self.validation_threshold is not None else None,
                "avg_window_f1": avg_f1,
                "avg_recall_coverage": avg_recall_cov,
                "recall_topk_k": int(self.recall_topk),
                "focal_loss": True,
            }
        }

    def _forward_node_embeddings(self, requires_grad: bool = False) -> torch.Tensor:
        if self.tgnn_model is None:
            raise RuntimeError("TGNN model is not ready")
        graph_data = getattr(self, "graph_data", None)
        if graph_data is None:
            if hasattr(self, "temporal_data") and self.temporal_data is not None:
                graph_data = self._temporal_to_static_data(self.temporal_data)
                self.graph_data = graph_data
            else:
                raise RuntimeError("Static graph data unavailable for embedding computation")
        target_dtype = (
            torch.float16 if self.precision_half and self.device.type == "cuda" else torch.float32
        )
        try:
            from torch_geometric.loader import NeighborLoader
        except Exception:
            data = self._to_device_half(self.temporal_data)
            model = self.tgnn_model
            if requires_grad:
                model.train()
                x = model.input_projection(data.x)
                for layer in model.tgnn_layers:
                    x = layer(x, data.edge_index, data.edge_attr, data.edge_time)
                return x
            model.eval()
            with torch.no_grad():
                x = model.input_projection(data.x)
                for layer in model.tgnn_layers:
                    x = layer(x, data.edge_index, data.edge_attr, data.edge_time)
                if x.dtype != target_dtype:
                    x = x.to(dtype=target_dtype)
                return x

        fanouts = self._neighbor_fanouts
        if fanouts is None:
            num_layers = max(1, len(self.tgnn_model.tgnn_layers))
            fanouts = [10] * num_layers
        batch_size = self._neighbor_batch_size or min(1024, max(256, graph_data.num_nodes // 20))
        seed_nodes = torch.arange(graph_data.num_nodes, dtype=torch.long)
        loader = NeighborLoader(
            graph_data,
            num_neighbors=fanouts,
            input_nodes=seed_nodes,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
        )
        outputs: List[torch.Tensor] = []
        if requires_grad:
            self.tgnn_model.train()
            for batch in loader:
                batch = self._to_device_half(batch)
                node_repr = self.tgnn_model(batch, task=None)
                outputs.append(node_repr[: batch.batch_size])
            return torch.cat(outputs, dim=0)
        self.tgnn_model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = self._to_device_half(batch)
                node_repr = self.tgnn_model(batch, task=None)
                chunk = node_repr[: batch.batch_size]
                if chunk.dtype != target_dtype:
                    chunk = chunk.to(dtype=target_dtype)
                outputs.append(chunk.to(self.device))
        return torch.cat(outputs, dim=0)


    def run_global_hate_prediction(self):
        print("    Running global hate speech prediction...")
        train_windows, val_windows, test_windows = self.create_time_windows(
            window_hours=self.window_hours,
            min_hate_per_window=self.min_hate_per_window,
        )
        if not train_windows or not test_windows:
            print("    Insufficient data for temporal windows")
            return {}
        self.build_network_state(train_windows)
        self.train_edge_mlp_from_windows(train_windows)
        self._reset_cuda_peak()
        window_records: List[Dict[str, Any]] = []
        for idx, window in enumerate(test_windows):
            print(
                f"    Predicting window {idx + 1}/{len(test_windows)}: "
                f"{window['num_comments']} comments, {window['num_hate']} hate"
            )
            candidates = self.extract_candidate_comments(window)
            if not candidates:
                continue
            selected_candidates, recall_info = self._recall_topk(candidates)
            if not selected_candidates:
                continue
            predictions = self.predict_window_hate(selected_candidates)
            truth = [int(c["hate_label"]) for c in selected_candidates]
            window_records.append(
                {
                    "predictions": predictions,
                    "truth": truth,
                    "window": window,
                    "recall": recall_info,
                }
            )
            self.update_network_state(window)
        self._log_cuda_memory("Prediction stage")
        return self.evaluate_global_hate_prediction(window_records)


def _parse_sample_value(value: Optional[str]):
    if value is None:
        return None
    try:
        if value.strip() == "":
            return None
        if any(ch in value for ch in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid sample value '{value}': {exc}")


def _str2bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def main():
    parser = argparse.ArgumentParser(description="TGNN-enhanced diffusion prediction on CSV data")
    parser.add_argument("--sample-train", type=str, default=None, help="Sample size or fraction for training data")
    parser.add_argument("--sample-val", type=str, default=None, help="Sample size or fraction for validation data")
    parser.add_argument("--sample-test", type=str, default=None, help="Sample size or fraction for test data")
    parser.add_argument("--window-size-comments", type=int, default=8000, help="Number of comments per window (legacy)")
    parser.add_argument("--window-hours", type=int, default=12, help="Temporal window size in hours")
    parser.add_argument("--min-hate-per-window", type=int, default=1, help="Minimum hate comments per window")
    parser.add_argument("--recall-topk", type=int, default=200, help="Recall stage top-k candidates")
    parser.add_argument("--hard-neg-per-pos", type=int, default=5, help="Hard negatives per positive sample")
    parser.add_argument("--tgnn-hidden", type=int, default=128, help="TGNN hidden dimension (<=128)")
    parser.add_argument("--tgnn-epochs", type=int, default=8, help="TGNN training epochs")
    parser.add_argument("--mlp-hidden", type=int, default=128, help="Edge-MLP hidden dimension")
    parser.add_argument("--mlp-epochs", type=int, default=3, help="Edge-MLP training epochs")
    parser.add_argument("--mlp-batch-size", type=int, default=2048, help="Edge-MLP batch size")
    parser.add_argument("--precision-half", type=str, default="true", help="Use float16 on GPU (true/false)")
    args = parser.parse_args()

    sample_config = {}
    for key, value in [("train", args.sample_train), ("val", args.sample_val), ("test", args.sample_test)]:
        parsed = _parse_sample_value(value)
        if parsed is not None:
            sample_config[key] = parsed
    if not sample_config:
        sample_config = None

    print("=== TGNN-Enhanced CSV Diffusion Prediction ===")
    print("    [OK] Using CSV supervision data with TGNN integration")
    print("        Enhanced with temporal graph neural networks")

    script_dir = Path(__file__).parent
    predictor = TGNNDiffusionPredictor(
        data_dir=script_dir,
        sample_config=sample_config,
        window_size_comments=args.window_size_comments,
        window_hours=args.window_hours,
        min_hate_per_window=args.min_hate_per_window,
        recall_topk=args.recall_topk,
        hard_neg_per_pos=args.hard_neg_per_pos,
        tgnn_hidden=args.tgnn_hidden,
        tgnn_epochs=args.tgnn_epochs,
        mlp_hidden=args.mlp_hidden,
        mlp_epochs=args.mlp_epochs,
        mlp_batch_size=args.mlp_batch_size,
        precision_half=_str2bool(args.precision_half),
    )
    predictor.load_data()
    results = predictor.run_global_hate_prediction()

    print()
    print("=== Global Hate Speech Prediction Results ===")
    if "global_hate" in results:
        gh = results["global_hate"]
        print(f"    Total predictions: {gh['total_predictions']:,}")
        print(f"    Total positives: {gh['total_positives']:,} ({gh['positive_rate']*100:.2f}%)")
        print(f"    Global PR-AUC: {gh['pr_auc']:.3f}")
        print(f"    Avg Window PR-AUC: {gh['window_pr_auc']:.3f}")
        print("    Hit@k Results (per window):")
        for k in [1, 5, 10, 20]:
            print(f"      Hit@{k}: {gh['hit'].get(k, 0.0):.3f}")
        hit1 = gh["hit"].get(1, 0.0)
        hit5 = gh["hit"].get(5, 0.0)
        hit10 = gh["hit"].get(10, 0.0)
        mrr = (hit1 + hit5 / 5 + hit10 / 10) / 3
        print("    Overall Metrics:")
        print(f"      MRR: {mrr:.3f}")
        print(f"      Precision@1: {hit1:.3f}")

    with open("global_hate_prediction_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("    Results saved to global_hate_prediction_results.json")


if __name__ == "__main__":
    main()

