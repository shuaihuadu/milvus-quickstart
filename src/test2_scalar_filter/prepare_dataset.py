"""
测试二 · 数据集准备：

1. 创建 collection（schema 见 src/common/schema.py）
2. 构建索引（向量 + 标量，方案由 cfg.scalar_index_plan 控制）
3. 按 batch 流式 insert 共 cfg.num_rows 行
4. flush + load

用法：
    # 小规模 smoke：5 万行，128 维
    NUM_ROWS=50000 DIM=128 SCALAR_PLAN=S3 python -m src.test2_scalar_filter.prepare_dataset

    # 生产规模：10M 行，3072 维（耗时较长）
    NUM_ROWS=10000000 DIM=3072 SCALAR_PLAN=S3 python -m src.test2_scalar_filter.prepare_dataset

参数全部走 src/common/config.py 的环境变量。
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from tqdm import tqdm

from src.common.config import CFG, TestConfig
from src.common.data_gen import BatchDataGenerator, make_scalar_pools
from src.common.logging_utils import get_logger
from src.common.metrics import timer, write_json
from src.common.milvus_client import drop_collection_if_exists, get_client
from src.common.schema import build_index_params, build_schema, index_summary

log = get_logger("prepare_dataset")


def parse_args(cfg: TestConfig) -> TestConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--collection", default=cfg.collection_name)
    p.add_argument("--num-rows", type=int, default=cfg.num_rows)
    p.add_argument("--dim", type=int, default=cfg.dim)
    p.add_argument("--batch", type=int, default=cfg.insert_batch_size)
    p.add_argument("--scalar-plan", default=cfg.scalar_index_plan,
                   choices=["S0", "S1", "S2", "S3", "S4", "S5"])
    p.add_argument("--vec-index", default=cfg.vector_index_type)
    p.add_argument("--no-sparse", action="store_true",
                   help="不创建稀疏向量字段（关闭多路召回能力）")
    p.add_argument("--sparse-nnz", type=int, default=cfg.sparse_nnz_per_row,
                   help="每行稀疏向量的非零项数")
    p.add_argument("--sparse-vocab", type=int, default=cfg.sparse_vocab_size,
                   help="稀疏向量词表大小")
    p.add_argument("--kb-card", type=int, default=cfg.kb_cardinality,
                   help="kbId 取值个数（业务侧 KB 数量）")
    p.add_argument("--num-partitions", type=int, default=cfg.num_partitions,
                   help="S5 方案下的 partition_key 分区数量")
    p.add_argument("--chunks-min", type=int, default=cfg.chunks_per_doc_min,
                   help="单文档最小 chunk 数")
    p.add_argument("--chunks-max", type=int, default=cfg.chunks_per_doc_max,
                   help="单文档最大 chunk 数")
    p.add_argument("--drop-existing", action="store_true",
                   help="如果同名 collection 已存在则先 drop")
    p.add_argument("--skip-load", action="store_true",
                   help="只插数据，不 load（适合超大规模数据先入库再 load）")
    args = p.parse_args()

    cfg.collection_name = args.collection
    cfg.num_rows = args.num_rows
    cfg.dim = args.dim
    cfg.insert_batch_size = args.batch
    cfg.scalar_index_plan = args.scalar_plan
    cfg.vector_index_type = args.vec_index
    if args.no_sparse:
        cfg.enable_sparse = False
    cfg.sparse_nnz_per_row = args.sparse_nnz
    cfg.sparse_vocab_size = args.sparse_vocab
    cfg.kb_cardinality = args.kb_card
    cfg.num_partitions = args.num_partitions
    cfg.chunks_per_doc_min = args.chunks_min
    cfg.chunks_per_doc_max = args.chunks_max
    cfg._cli_drop_existing = args.drop_existing  # type: ignore[attr-defined]
    cfg._cli_skip_load = args.skip_load  # type: ignore[attr-defined]
    return cfg


def main() -> None:
    cfg = parse_args(CFG)
    log.info("Config: %s", cfg.to_dict())

    client = get_client(cfg)

    # 1. 处理同名 collection
    if client.has_collection(cfg.collection_name):
        if getattr(cfg, "_cli_drop_existing", False):
            drop_collection_if_exists(client, cfg.collection_name)
        else:
            raise SystemExit(
                f"Collection '{cfg.collection_name}' already exists. "
                f"Use --drop-existing to recreate."
            )

    # 2. 创建 collection（含索引）
    log.info("Building schema (dim=%d)", cfg.dim)
    schema = build_schema(client, cfg)
    log.info("Index summary: %s", index_summary(cfg))
    index_params = build_index_params(client, cfg)

    with timer() as t:
        client.create_collection(
            collection_name=cfg.collection_name,
            schema=schema,
            index_params=index_params,
        )
    log.info("create_collection done in %.1f ms", t.ms)

    # 3. 流式 insert
    pools = make_scalar_pools(
        cfg.kb_cardinality,
        cfg.study_cardinality,
        cfg.site_cardinality,
        cfg.patient_cardinality,
        cfg.visit_cardinality,
    )
    gen = BatchDataGenerator(
        total_rows=cfg.num_rows,
        batch_size=cfg.insert_batch_size,
        dim=cfg.dim,
        pools=pools,
        zipf_alpha=cfg.zipf_alpha,
        content_length=cfg.content_length,
        seed=cfg.seed,
        chunks_per_doc_min=cfg.chunks_per_doc_min,
        chunks_per_doc_max=cfg.chunks_per_doc_max,
        enable_sparse=cfg.enable_sparse,
        sparse_vocab_size=cfg.sparse_vocab_size,
        sparse_nnz_per_row=cfg.sparse_nnz_per_row,
    )

    log.info(
        "Inserting %d rows in %d batches (batch_size=%d)",
        cfg.num_rows, gen.num_batches, cfg.insert_batch_size,
    )
    t_total_start = time.perf_counter()
    insert_ms_total = 0.0
    rows_inserted = 0
    pbar = tqdm(total=cfg.num_rows, unit="row", desc="insert", dynamic_ncols=True)
    for bi, rows in enumerate(gen.batches()):
        with timer() as t:
            client.insert(collection_name=cfg.collection_name, data=rows)
        insert_ms_total += t.ms
        rows_inserted += len(rows)
        pbar.update(len(rows))
        if (bi + 1) % cfg.flush_every_batches == 0:
            log.debug("Flushing at batch %d (rows=%d)", bi + 1, rows_inserted)
            client.flush(collection_name=cfg.collection_name)
    pbar.close()

    log.info("Final flush ...")
    with timer() as t_flush:
        client.flush(collection_name=cfg.collection_name)
    log.info("Final flush done in %.1f s", t_flush.ms / 1000.0)

    total_s = time.perf_counter() - t_total_start
    avg_throughput = rows_inserted / max(total_s, 1e-9)
    log.info(
        "Insert done: rows=%d, insert_ms_total=%.1f, wall_s=%.1f, throughput=%.0f rows/s",
        rows_inserted, insert_ms_total, total_s, avg_throughput,
    )

    # 4. load（索引在创建时已声明，create_collection 会触发 build；load 时如果索引未完成可能阻塞）
    if not getattr(cfg, "_cli_skip_load", False):
        log.info("Loading collection (this may block until indexes are built)...")
        with timer() as t_load:
            client.load_collection(collection_name=cfg.collection_name)
        log.info("load_collection done in %.1f s", t_load.ms / 1000.0)
    else:
        log.info("--skip-load specified, leaving collection unloaded")

    # 5. 落 manifest，方便后续脚本读取
    out = Path(cfg.results_dir) / f"{cfg.collection_name}.manifest.json"
    write_json(
        out,
        {
            "collection": cfg.collection_name,
            "config": cfg.to_dict(),
            "index_summary": index_summary(cfg),
            "rows_inserted": rows_inserted,
            "insert_wall_seconds": total_s,
            "avg_throughput_rows_per_s": avg_throughput,
            "scalar_pools": {
                "kb": pools.kb,
                "study": pools.study,
                "site": pools.site,
                "patient": pools.patient,
                "visit": pools.visit,
            },
            "docs_generated": gen.total_docs_generated,
        },
    )
    log.info("Wrote manifest -> %s", out)


if __name__ == "__main__":
    main()
