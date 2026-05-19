"""
全局可调测试参数。

约定：所有脚本都从这里读默认值，命令行参数 / 环境变量可覆盖。
覆盖优先级：CLI args > env > defaults。

`.env` 会在 import 时自动加载（如果文件存在且装了 python-dotenv）。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Tuple

# 静默尝试加载 .env：找项目根（脚本上溯找含 .env 的目录），找不到也不报错。
try:
    from dotenv import load_dotenv  # type: ignore

    _here = Path(__file__).resolve()
    for parent in [_here.parent, *_here.parents]:
        env_file = parent / ".env"
        if env_file.is_file():
            load_dotenv(env_file, override=False)
            break
except ImportError:
    pass


def _env(key: str, default, cast=str):
    v = os.environ.get(key)
    if v is None or v == "":
        return default
    try:
        if cast is bool:
            return v.lower() in ("1", "true", "yes", "y", "on")
        return cast(v)
    except Exception:
        return default


@dataclass
class TestConfig:
    # ---------- 连接 ----------
    uri: str = _env("MILVUS_URI", "http://localhost:19530")
    token: str = _env("MILVUS_TOKEN", "")
    db_name: str = _env("MILVUS_DB", "default")

    # ---------- Collection ----------
    collection_name: str = _env("COLLECTION_NAME", "scalar_filter_test")
    dim: int = _env("DIM", 3072, int)
    # 是否启用 mmap（3072 维 10M 行场景几乎必开）
    mmap_enabled: bool = _env("MMAP_ENABLED", False, bool)

    # ---------- 数据规模 ----------
    num_rows: int = _env("NUM_ROWS", 10_000_000, int)
    insert_batch_size: int = _env("INSERT_BATCH", 50_000, int)
    flush_every_batches: int = _env("FLUSH_EVERY", 20, int)

    # ---------- 文档切片模型 ----------
    # 每个文档随机切成 [min, max] 个 chunk（均匀分布）。同一文档的 chunk 共享
    # documentId / kbId / study / site / patient / visit。
    chunks_per_doc_min: int = _env("CHUNKS_PER_DOC_MIN", 20, int)
    chunks_per_doc_max: int = _env("CHUNKS_PER_DOC_MAX", 500, int)

    # ---------- 标量字段基数 ----------
    # kbId：知识库 ID（业务侧 ~10 个）。同时作为 partition_key（见 SCALAR_PLAN=S5）
    kb_cardinality: int = _env("KB_CARD", 10, int)
    # 用户场景：study/site/patient/visit 都是几十级。可单独调
    study_cardinality: int = _env("STUDY_CARD", 30, int)
    site_cardinality: int = _env("SITE_CARD", 30, int)
    patient_cardinality: int = _env("PATIENT_CARD", 30, int)
    visit_cardinality: int = _env("VISIT_CARD", 30, int)
    # 分布偏斜：0=均匀，>0=Zipfian（典型 1.0~1.5）。同时作用于 kbId 与其它低基数字段。
    zipf_alpha: float = _env("ZIPF_ALPHA", 1.0, float)

    # content 占位串长度（仅 output，不参与过滤）
    content_length: int = _env("CONTENT_LEN", 256, int)

    # ---------- 索引 ----------
    # 向量索引：HNSW / IVF_FLAT
    vector_index_type: str = _env("VEC_INDEX", "HNSW")
    vector_metric: str = _env("VEC_METRIC", "IP")  # IP / L2 / COSINE
    hnsw_M: int = _env("HNSW_M", 16, int)
    hnsw_efConstruction: int = _env("HNSW_EFC", 200, int)
    ivf_nlist: int = _env("IVF_NLIST", 4096, int)

    # 标量索引方案：S0..S5
    #   S0: 都不建；S1: documentId(INVERTED) + kbId(BITMAP)
    #   S2: 低基数(BITMAP) + kbId(BITMAP)；S3: documentId(INVERTED) + 低基数(BITMAP) + kbId(BITMAP)
    #   S4: 全 INVERTED；S5: 与 S3 同 + kbId 作为 partition_key（生产首选）
    scalar_index_plan: str = _env("SCALAR_PLAN", "S5")

    # partition_key 分区数（仅 S5 生效）。10 个 kbId hash 到 16 分区基本能 1KB ≈ 1 分区。
    num_partitions: int = _env("NUM_PARTITIONS", 16, int)

    # ---------- 稀疏向量（多路召回） ----------
    # 开关：True 时 collection 增加 sparse_embedding 字段并建 SPARSE_INVERTED_INDEX
    enable_sparse: bool = _env("ENABLE_SPARSE", True, bool)
    # 词表大小（max token id 范围）。SPLADE/BERT 级别 30k 是常见值
    sparse_vocab_size: int = _env("SPARSE_VOCAB", 30000, int)
    # 每行非零项数。BM25 通常几十~几百，SPLADE 蒸馏后 100~300
    sparse_nnz_per_row: int = _env("SPARSE_NNZ", 64, int)
    sparse_index_type: str = _env("SPARSE_INDEX", "SPARSE_INVERTED_INDEX")
    # 查询稀疏向量的非零数（一般比插入侧小）
    sparse_query_nnz: int = _env("SPARSE_QUERY_NNZ", 32, int)
    # Rank 融合方式：RRF / Weighted
    hybrid_ranker: str = _env("HYBRID_RANKER", "RRF")
    rrf_k: int = _env("RRF_K", 60, int)
    weighted_dense: float = _env("WEIGHT_DENSE", 1.0, float)
    weighted_sparse: float = _env("WEIGHT_SPARSE", 1.0, float)

    # ---------- 查询 ----------
    top_k: int = _env("TOPK", 10, int)
    nq: int = _env("NQ", 1, int)
    warmup_iters: int = _env("WARMUP", 100, int)
    serial_iters: int = _env("SERIAL_ITERS", 1000, int)
    search_ef: int = _env("SEARCH_EF", 64, int)  # HNSW 搜索时 ef
    search_nprobe: int = _env("NPROBE", 16, int)  # IVF 搜索时 nprobe
    concurrency_levels: Tuple[int, ...] = (1, 4, 16, 32)
    concurrency_duration_s: int = _env("CONC_DUR", 60, int)

    # ---------- 测试一（Collection 上限） ----------
    coll_limit_target: int = _env("COLL_LIMIT_TARGET", 5000, int)
    coll_limit_sample_every: int = _env("COLL_LIMIT_SAMPLE", 50, int)
    coll_limit_min_dim: int = _env("COLL_LIMIT_MIN_DIM", 8, int)

    # ---------- 输出 ----------
    results_dir: str = _env("RESULTS_DIR", "./results")
    log_level: str = _env("LOG_LEVEL", "INFO")

    # ---------- 随机种子 ----------
    seed: int = _env("SEED", 42, int)

    def to_dict(self) -> dict:
        return asdict(self)


# 全局单例（按需 import）
CFG = TestConfig()
