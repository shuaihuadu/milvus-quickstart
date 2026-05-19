"""
Collection schema 与索引构造。

把 schema、索引方案的所有变量都集中在这里，方便后面替换不同方案。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from pymilvus import DataType, MilvusClient

from .config import TestConfig


# 标量字段名集合（低基数）
LOW_CARD_FIELDS: tuple[str, ...] = ("study", "site", "patient", "visit")
# 高基数字段
HIGH_CARD_FIELDS: tuple[str, ...] = ("documentId",)
# kbId：知识库隔离主过滤字段，同时是 partition_key 候选字段
PARTITION_KEY_FIELD: str = "kbId"


def build_schema(client: MilvusClient, cfg: TestConfig):
    """构造 collection schema。

    使用 MilvusClient 风格 schema：
    - id INT64 PK, auto_id=False（外部控制 ID，便于结果可复现）
    - kbId VARCHAR(16)；在 SCALAR_PLAN=S5 时作为 partition_key
    - documentId VARCHAR(64)
    - embedding FLOAT_VECTOR(dim)
    - content VARCHAR(2048)
    - study/site/patient/visit VARCHAR
    - sparse_embedding SPARSE_FLOAT_VECTOR（cfg.enable_sparse=True 时）
    """
    plan = SCALAR_INDEX_PLANS[cfg.scalar_index_plan]
    schema = client.create_schema(
        auto_id=False,
        enable_dynamic_field=False,
        partition_key_field=PARTITION_KEY_FIELD if plan.use_partition_key else None,
        num_partitions=cfg.num_partitions if plan.use_partition_key else None,
        description="Milvus 2.5.10 scalar-filter performance test",
    )
    schema.add_field("id", DataType.INT64, is_primary=True)
    # kbId 字段：必须在创建时标记 is_partition_key=True（S5 plan）
    schema.add_field(
        "kbId",
        DataType.VARCHAR,
        max_length=16,
        is_partition_key=plan.use_partition_key,
    )
    schema.add_field("documentId", DataType.VARCHAR, max_length=64)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=cfg.dim)
    schema.add_field("content", DataType.VARCHAR, max_length=2048)
    schema.add_field("study", DataType.VARCHAR, max_length=32)
    schema.add_field("site", DataType.VARCHAR, max_length=32)
    schema.add_field("patient", DataType.VARCHAR, max_length=64)
    schema.add_field("visit", DataType.VARCHAR, max_length=32)
    if cfg.enable_sparse:
        # 稀疏向量字段不指定 dim，pymilvus 用 max-index 自动推断
        schema.add_field("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR)
    return schema


def _vector_index_params(cfg: TestConfig) -> dict:
    if cfg.vector_index_type.upper() == "HNSW":
        return {"M": cfg.hnsw_M, "efConstruction": cfg.hnsw_efConstruction}
    if cfg.vector_index_type.upper() == "IVF_FLAT":
        return {"nlist": cfg.ivf_nlist}
    return {}


@dataclass(frozen=True)
class ScalarIndexPlan:
    """标量索引方案：定义哪个字段建什么类型的索引。

    fields 中的 key 包括业务字段（documentId / kbId / study / site / patient / visit）；
    use_partition_key 表示是否把 kbId 设为 partition_key（只能创建 collection 时设）。
    """

    name: str
    # field_name -> index_type ("BITMAP" / "INVERTED" / None)
    fields: dict[str, str | None]
    # 是否启用 kbId 作为 partition_key
    use_partition_key: bool = False


# 6 个方案，与 docs/test-plan.md §2.3 对应。kbId 在 S1~S5 都默认带 BITMAP（低基数最佳）。
SCALAR_INDEX_PLANS: dict[str, ScalarIndexPlan] = {
    "S0": ScalarIndexPlan(
        name="S0_no_scalar_index",
        fields={
            "kbId": None,
            **{f: None for f in HIGH_CARD_FIELDS + LOW_CARD_FIELDS},
        },
    ),
    "S1": ScalarIndexPlan(
        name="S1_kbId_bitmap_plus_documentId_inverted",
        fields={
            "kbId": "BITMAP",
            **{f: "INVERTED" for f in HIGH_CARD_FIELDS},
            **{f: None for f in LOW_CARD_FIELDS},
        },
    ),
    "S2": ScalarIndexPlan(
        name="S2_kbId_bitmap_plus_lowcard_bitmap",
        fields={
            "kbId": "BITMAP",
            **{f: None for f in HIGH_CARD_FIELDS},
            **{f: "BITMAP" for f in LOW_CARD_FIELDS},
        },
    ),
    "S3": ScalarIndexPlan(
        name="S3_recommended_no_partition_key",
        fields={
            "kbId": "BITMAP",
            **{f: "INVERTED" for f in HIGH_CARD_FIELDS},
            **{f: "BITMAP" for f in LOW_CARD_FIELDS},
        },
    ),
    "S4": ScalarIndexPlan(
        name="S4_all_inverted",
        fields={
            "kbId": "INVERTED",
            **{f: "INVERTED" for f in HIGH_CARD_FIELDS},
            **{f: "INVERTED" for f in LOW_CARD_FIELDS},
        },
    ),
    "S5": ScalarIndexPlan(
        name="S5_production_kbId_partition_key",
        fields={
            "kbId": "BITMAP",
            **{f: "INVERTED" for f in HIGH_CARD_FIELDS},
            **{f: "BITMAP" for f in LOW_CARD_FIELDS},
        },
        use_partition_key=True,
    ),
}


def build_index_params(client: MilvusClient, cfg: TestConfig):
    """根据 cfg.scalar_index_plan 与向量索引参数组装 IndexParams 对象。"""
    plan = SCALAR_INDEX_PLANS[cfg.scalar_index_plan]
    ip = client.prepare_index_params()

    # 向量字段索引
    ip.add_index(
        field_name="embedding",
        index_type=cfg.vector_index_type.upper(),
        metric_type=cfg.vector_metric.upper(),
        params=_vector_index_params(cfg),
    )
    # 稀疏向量索引（仅支持 IP）
    if cfg.enable_sparse:
        ip.add_index(
            field_name="sparse_embedding",
            index_type=cfg.sparse_index_type.upper(),
            metric_type="IP",
        )
    # 标量字段索引
    for field_name, index_type in plan.fields.items():
        if index_type is None:
            continue
        ip.add_index(field_name=field_name, index_type=index_type.upper())
    return ip


def index_summary(cfg: TestConfig) -> dict:
    plan = SCALAR_INDEX_PLANS[cfg.scalar_index_plan]
    return {
        "scalar_plan": plan.name,
        "vector_index": cfg.vector_index_type.upper(),
        "vector_metric": cfg.vector_metric.upper(),
        "vector_params": _vector_index_params(cfg),
        "scalar_fields": plan.fields,
        "partition_key": PARTITION_KEY_FIELD if plan.use_partition_key else None,
        "num_partitions": cfg.num_partitions if plan.use_partition_key else None,
        "sparse_enabled": cfg.enable_sparse,
        "sparse_index": cfg.sparse_index_type.upper() if cfg.enable_sparse else None,
    }
