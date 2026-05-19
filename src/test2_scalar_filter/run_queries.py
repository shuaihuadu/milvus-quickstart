"""
测试二 · 查询性能：

读取 prepare_dataset.py 写出的 manifest 拿到 collection 名与标量值池，
然后跑 Q1..Q7 的串行延迟 + 并发 QPS。所有 expr 都以 kbId == "kb_xx" 作为前置过滤。

  Q1 kbId + 单字段等值 + dense 搜索
  Q2 kbId + 多字段 AND + dense 搜索（生产核心）
  Q3 kbId + documentId IN list + dense 搜索
  Q5 纯标量 query（复用 Q2 expr）
  Q6 hybrid_search (dense + sparse) + Q2 expr，多路召回
  Q7 热点分区：Q7a/Q7b/Q7c 分别打热门/中等/冷门 KB。指标差异只能在 S5 下出现
用法：
    python -m src.test2_scalar_filter.run_queries \
        --collection scalar_filter_test \
        --queries Q1 Q2 Q3 Q5 Q6 Q7 \
        --serial 1000 --warmup 100
"""
from __future__ import annotations

import argparse
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pymilvus import AnnSearchRequest, RRFRanker, WeightedRanker

from src.common.config import CFG, TestConfig
from src.common.data_gen import BatchDataGenerator, make_scalar_pools
from src.common.logging_utils import get_logger
from src.common.metrics import LatencyStats, append_csv, timer, write_json
from src.common.milvus_client import get_client

log = get_logger("run_queries")


# ============== Query expression builders ==============

def _pick_kb(pools_dict: dict, rng: random.Random) -> str:
    """从 kb 池里随机采一个 kbId（查询侧均匀采样，并不遵从 zipfian）。"""
    return rng.choice(pools_dict["kb"])


def expr_q1(pools_dict: dict, rng: random.Random) -> tuple[str, str]:
    """kbId + 单字段等值过滤。"""
    kb = _pick_kb(pools_dict, rng)
    val = rng.choice(pools_dict["study"])
    return f'kbId == "{kb}" && study == "{val}"', "Q1_kbId_plus_single_field"


def expr_q2(pools_dict: dict, rng: random.Random) -> tuple[str, str]:
    """kbId + 多字段 AND（生产核心场景）。"""
    kb = _pick_kb(pools_dict, rng)
    s = rng.choice(pools_dict["study"])
    si = rng.choice(pools_dict["site"])
    p = rng.choice(pools_dict["patient"])
    v = rng.choice(pools_dict["visit"])
    return (
        f'kbId == "{kb}" && study == "{s}" && site == "{si}" '
        f'&& patient == "{p}" && visit == "{v}"',
        "Q2_kbId_plus_multi_field_and",
    )


def expr_q3(pools_dict: dict, total_docs: int, list_size: int,
            rng: random.Random) -> tuple[str, str]:
    """kbId + 高基数 documentId IN list。

    documentId 是文档级字段，所以在 [0, total_docs) 里采样。
    """
    kb = _pick_kb(pools_dict, rng)
    n = min(list_size, max(total_docs, 1))
    # rng.sample 要求 k <= population size；total_docs 为 0 时退化到 [0]
    population = range(1, max(total_docs + 1, 2))
    ids = rng.sample(population, k=n)
    items = ", ".join(f'"doc_{i:08d}"' for i in ids)
    return (
        f'kbId == "{kb}" && documentId in [{items}]',
        f"Q3_kbId_plus_documentId_in_{list_size}",
    )


def expr_q5_pure(pools_dict: dict, rng: random.Random) -> tuple[str, str]:
    """纯标量 query，复用 Q2 表达式。"""
    expr, _ = expr_q2(pools_dict, rng)
    return expr, "Q5_pure_scalar_query"


def _expr_for_kb(pools_dict: dict, kb: str, rng: random.Random) -> str:
    """给定具体 kbId，拼出与 Q2 类似的多字段表达式。用于 Q7 热点实验。"""
    s = rng.choice(pools_dict["study"])
    si = rng.choice(pools_dict["site"])
    p = rng.choice(pools_dict["patient"])
    v = rng.choice(pools_dict["visit"])
    return (
        f'kbId == "{kb}" && study == "{s}" && site == "{si}" '
        f'&& patient == "{p}" && visit == "{v}"'
    )


def expr_q7a(pools_dict: dict, rng: random.Random) -> tuple[str, str]:
    """热门 KB：Zipfian 下第一名（~25% 数据量）。"""
    kb = pools_dict["kb"][0]
    return _expr_for_kb(pools_dict, kb, rng), "Q7a_hot_kb"


def expr_q7b(pools_dict: dict, rng: random.Random) -> tuple[str, str]:
    """中等 KB：取中位（~10% 数据量）。"""
    kbs = pools_dict["kb"]
    kb = kbs[len(kbs) // 2] if kbs else kbs[0]
    return _expr_for_kb(pools_dict, kb, rng), "Q7b_mid_kb"


def expr_q7c(pools_dict: dict, rng: random.Random) -> tuple[str, str]:
    """冷门 KB：最后一位（~2% 数据量）。"""
    kb = pools_dict["kb"][-1]
    return _expr_for_kb(pools_dict, kb, rng), "Q7c_cold_kb"


# ============== Search params ==============

def build_search_params(cfg: TestConfig) -> dict:
    if cfg.vector_index_type.upper() == "HNSW":
        return {"metric_type": cfg.vector_metric.upper(),
                "params": {"ef": cfg.search_ef}}
    if cfg.vector_index_type.upper() == "IVF_FLAT":
        return {"metric_type": cfg.vector_metric.upper(),
                "params": {"nprobe": cfg.search_nprobe}}
    return {"metric_type": cfg.vector_metric.upper(), "params": {}}


def build_sparse_search_params() -> dict:
    # SPARSE_INVERTED_INDEX 仅支持 IP
    return {"metric_type": "IP", "params": {}}


def build_ranker(cfg: TestConfig):
    if cfg.hybrid_ranker.upper() == "WEIGHTED":
        return WeightedRanker(cfg.weighted_dense, cfg.weighted_sparse)
    return RRFRanker(cfg.rrf_k)


# ============== Runners ==============

def run_serial_search(client, cfg: TestConfig, expr: str, q_vec, n: int) -> list[float]:
    samples = []
    sp = build_search_params(cfg)
    for _ in range(n):
        with timer() as t:
            client.search(
                collection_name=cfg.collection_name,
                data=[q_vec],
                anns_field="embedding",
                filter=expr,
                limit=cfg.top_k,
                output_fields=["documentId", "study", "site", "patient", "visit"],
                search_params=sp,
            )
        samples.append(t.ms)
    return samples


def run_serial_hybrid(
    client,
    cfg: TestConfig,
    expr: str,
    q_dense,
    q_sparse: dict,
    n: int,
) -> list[float]:
    """多路召回：dense + sparse 同时 ANN，各自 filter，然后 fusion。"""
    samples: list[float] = []
    sp_dense = build_search_params(cfg)
    sp_sparse = build_sparse_search_params()
    ranker = build_ranker(cfg)
    # AnnSearchRequest 在不同 pymilvus 小版本间采用 expr/filter 或仅位置参数，
    # 2.5.10 推荐用 expr。
    for _ in range(n):
        req_dense = AnnSearchRequest(
            data=[q_dense],
            anns_field="embedding",
            param=sp_dense,
            limit=cfg.top_k,
            expr=expr,
        )
        req_sparse = AnnSearchRequest(
            data=[q_sparse],
            anns_field="sparse_embedding",
            param=sp_sparse,
            limit=cfg.top_k,
            expr=expr,
        )
        with timer() as t:
            client.hybrid_search(
                collection_name=cfg.collection_name,
                reqs=[req_dense, req_sparse],
                ranker=ranker,
                limit=cfg.top_k,
                output_fields=["documentId", "study", "site", "patient", "visit"],
            )
        samples.append(t.ms)
    return samples


def run_serial_query(client, cfg: TestConfig, expr: str, n: int) -> list[float]:
    samples = []
    for _ in range(n):
        with timer() as t:
            client.query(
                collection_name=cfg.collection_name,
                filter=expr,
                limit=100,
                output_fields=["id", "documentId"],
            )
        samples.append(t.ms)
    return samples


def run_concurrent_search(
    client, cfg: TestConfig, expr_factory, q_vecs: list, concurrency: int, duration_s: int,
) -> tuple[int, list[float]]:
    """concurrency 个线程在 duration_s 内不停发 search，返回 (总请求数, 单请求 ms 列表)。"""
    sp = build_search_params(cfg)
    stop = threading.Event()
    samples: list[float] = []
    lock = threading.Lock()

    def worker(seed: int):
        rng = random.Random(seed)
        local_samples = []
        while not stop.is_set():
            expr, _ = expr_factory(rng)
            qv = rng.choice(q_vecs)
            with timer() as t:
                client.search(
                    collection_name=cfg.collection_name,
                    data=[qv],
                    anns_field="embedding",
                    filter=expr,
                    limit=cfg.top_k,
                    output_fields=["documentId"],
                    search_params=sp,
                )
            local_samples.append(t.ms)
        with lock:
            samples.extend(local_samples)

    threads = []
    start = time.perf_counter()
    for i in range(concurrency):
        th = threading.Thread(target=worker, args=(cfg.seed + i,), daemon=True)
        th.start()
        threads.append(th)
    time.sleep(duration_s)
    stop.set()
    for th in threads:
        th.join()
    wall_s = time.perf_counter() - start
    return len(samples), samples


# ============== Main ==============

def parse_args(cfg: TestConfig) -> tuple[TestConfig, argparse.Namespace]:
    p = argparse.ArgumentParser()
    p.add_argument("--collection", default=cfg.collection_name)
    p.add_argument("--queries", nargs="+",
                   default=["Q1", "Q2", "Q3", "Q5"],
                   choices=["Q1", "Q2", "Q3", "Q5", "Q6", "Q7"],
                   help="要跑的查询场景（Q4 在 run_hit_rate_scan.py 里专门做；Q6 需 sparse 启用；Q7 含 a/b/c 三个热点子场景）")
    p.add_argument("--serial", type=int, default=cfg.serial_iters)
    p.add_argument("--warmup", type=int, default=cfg.warmup_iters)
    p.add_argument("--q3-list-size", type=int, default=100,
                   help="Q3 中 documentId IN 列表的长度")
    p.add_argument("--ranker", default=cfg.hybrid_ranker, choices=["RRF", "WEIGHTED"],
                   help="Q6 hybrid_search 的排名融合方式")
    p.add_argument("--rrf-k", type=int, default=cfg.rrf_k)
    p.add_argument("--w-dense", type=float, default=cfg.weighted_dense)
    p.add_argument("--w-sparse", type=float, default=cfg.weighted_sparse)
    p.add_argument("--concurrency", nargs="+", type=int, default=list(cfg.concurrency_levels),
                   help="并发级别；设为 0 跳过并发测试")
    p.add_argument("--conc-dur", type=int, default=cfg.concurrency_duration_s)
    p.add_argument("--no-concurrent", action="store_true",
                   help="跳过并发测试，只跑串行延迟")
    args = p.parse_args()
    cfg.collection_name = args.collection
    cfg.serial_iters = args.serial
    cfg.warmup_iters = args.warmup
    cfg.concurrency_duration_s = args.conc_dur
    cfg.hybrid_ranker = args.ranker
    cfg.rrf_k = args.rrf_k
    cfg.weighted_dense = args.w_dense
    cfg.weighted_sparse = args.w_sparse
    return cfg, args


def load_manifest(cfg: TestConfig) -> dict:
    p = Path(cfg.results_dir) / f"{cfg.collection_name}.manifest.json"
    if not p.exists():
        raise SystemExit(f"manifest not found: {p}. Run prepare_dataset first.")
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    cfg, args = parse_args(CFG)
    manifest = load_manifest(cfg)
    pools_dict = manifest["scalar_pools"]
    num_rows = int(manifest.get("rows_inserted", cfg.num_rows))
    # 文档总数：优先读 manifest 中 docs_generated；没有则估算
    total_docs = int(manifest.get("docs_generated", 0))
    if total_docs <= 0:
        # 按默认 chunks_per_doc 中位 260 估算
        total_docs = max(num_rows // 260, 1)
    # 同步配置（manifest 里记录的是数据集构造时的实际参数）
    saved = manifest["config"]
    cfg.dim = int(saved["dim"])
    cfg.vector_index_type = saved["vector_index_type"]
    cfg.vector_metric = saved["vector_metric"]
    cfg.scalar_index_plan = saved["scalar_index_plan"]
    cfg.enable_sparse = bool(saved.get("enable_sparse", False))
    cfg.sparse_vocab_size = int(saved.get("sparse_vocab_size", cfg.sparse_vocab_size))
    cfg.sparse_nnz_per_row = int(saved.get("sparse_nnz_per_row", cfg.sparse_nnz_per_row))
    cfg.sparse_query_nnz = int(saved.get("sparse_query_nnz", cfg.sparse_query_nnz))
    log.info(
        "Loaded manifest: rows=%d, index=%s, plan=%s, sparse=%s",
        num_rows, cfg.vector_index_type, cfg.scalar_index_plan, cfg.enable_sparse,
    )

    if "Q6" in args.queries and not cfg.enable_sparse:
        raise SystemExit(
            "Q6 (hybrid_search) requested but this collection was created with "
            "enable_sparse=False. Re-run prepare_dataset without --no-sparse."
        )

    client = get_client(cfg)
    # 确保 collection 已 load
    client.load_collection(collection_name=cfg.collection_name)

    # 准备查询向量池（dense + 可选 sparse），各 1000 条，用于 warmup + 随机抽取
    pools = make_scalar_pools(
        # query 侧不需要 zipfian，kb 池仅用于 Q7；这里传业务参数只为了保持 BatchDataGenerator 可运行
        cfg.kb_cardinality, cfg.study_cardinality, cfg.site_cardinality,
        cfg.patient_cardinality, cfg.visit_cardinality,
    )
    qvec_gen = BatchDataGenerator(
        total_rows=1000, batch_size=1000, dim=cfg.dim,
        pools=pools, seed=cfg.seed + 1,
        enable_sparse=cfg.enable_sparse,
        sparse_vocab_size=cfg.sparse_vocab_size,
        sparse_nnz_per_row=cfg.sparse_nnz_per_row,
    )
    q_vecs = qvec_gen.sample_query_vectors(1000)
    q_sparse: list[dict] = []
    if cfg.enable_sparse:
        q_sparse = qvec_gen.sample_query_sparse(1000, nnz=cfg.sparse_query_nnz)
        log.info(
            "Prepared %d dense (dim=%d) + %d sparse (vocab=%d, nnz=%d) query vectors",
            len(q_vecs), cfg.dim, len(q_sparse), cfg.sparse_vocab_size, cfg.sparse_query_nnz,
        )
    else:
        log.info("Prepared %d query vectors (dim=%d, sparse disabled)", len(q_vecs), cfg.dim)

    rng = random.Random(cfg.seed + 2)

    # 输出文件
    csv_path = Path(cfg.results_dir) / "test2_serial_latency.csv"
    conc_csv = Path(cfg.results_dir) / "test2_concurrent_qps.csv"

    # ---------------- Q1..Q6 串行延迟 ----------------
    for q in args.queries:
        if q == "Q1":
            expr, label = expr_q1(pools_dict, rng)
            log.info("[%s] expr=%s", label, expr)
            # warmup
            run_serial_search(client, cfg, expr, q_vecs[0], cfg.warmup_iters)
            samples = run_serial_search(client, cfg, expr, q_vecs[0], cfg.serial_iters)
            stats = LatencyStats.from_samples(label, samples)
        elif q == "Q2":
            expr, label = expr_q2(pools_dict, rng)
            log.info("[%s] expr=%s", label, expr)
            run_serial_search(client, cfg, expr, q_vecs[0], cfg.warmup_iters)
            samples = run_serial_search(client, cfg, expr, q_vecs[0], cfg.serial_iters)
            stats = LatencyStats.from_samples(label, samples)
        elif q == "Q3":
            expr, label = expr_q3(pools_dict, total_docs, args.q3_list_size, rng)
            log.info("[%s] expr len=%d chars", label, len(expr))
            run_serial_search(client, cfg, expr, q_vecs[0], cfg.warmup_iters)
            samples = run_serial_search(client, cfg, expr, q_vecs[0], cfg.serial_iters)
            stats = LatencyStats.from_samples(label, samples)
        elif q == "Q5":
            expr, label = expr_q5_pure(pools_dict, rng)
            log.info("[%s] (pure query) expr=%s", label, expr)
            run_serial_query(client, cfg, expr, cfg.warmup_iters)
            samples = run_serial_query(client, cfg, expr, cfg.serial_iters)
            stats = LatencyStats.from_samples(label, samples)
        elif q == "Q6":
            expr, _ = expr_q2(pools_dict, rng)
            label = f"Q6_hybrid_{cfg.hybrid_ranker.lower()}"
            log.info("[%s] expr=%s ranker=%s", label, expr, cfg.hybrid_ranker)
            run_serial_hybrid(client, cfg, expr, q_vecs[0], q_sparse[0], cfg.warmup_iters)
            samples = run_serial_hybrid(
                client, cfg, expr, q_vecs[0], q_sparse[0], cfg.serial_iters,
            )
            stats = LatencyStats.from_samples(label, samples)
        elif q == "Q7":
            # Q7 一次跑三个子场景：热门/中等/冷门 KB
            for sub_factory in (expr_q7a, expr_q7b, expr_q7c):
                expr_sub, label_sub = sub_factory(pools_dict, rng)
                log.info("[%s] expr=%s", label_sub, expr_sub)
                run_serial_search(client, cfg, expr_sub, q_vecs[0], cfg.warmup_iters)
                samples_sub = run_serial_search(
                    client, cfg, expr_sub, q_vecs[0], cfg.serial_iters,
                )
                stats_sub = LatencyStats.from_samples(label_sub, samples_sub)
                row_sub = {
                    "plan": cfg.scalar_index_plan,
                    "vec_index": cfg.vector_index_type,
                    "query": label_sub,
                    "expr": expr_sub,
                    **stats_sub.to_row(),
                }
                append_csv(csv_path, row_sub)
                log.info(
                    "%s | n=%d p50=%.2fms p95=%.2fms p99=%.2fms mean=%.2fms",
                    label_sub, stats_sub.n, stats_sub.p50_ms, stats_sub.p95_ms,
                    stats_sub.p99_ms, stats_sub.mean_ms,
                )
            continue
        else:
            continue

        row = {
            "plan": cfg.scalar_index_plan,
            "vec_index": cfg.vector_index_type,
            "query": label,
            "expr": expr,
            **stats.to_row(),
        }
        append_csv(csv_path, row)
        log.info(
            "%s | n=%d p50=%.2fms p95=%.2fms p99=%.2fms mean=%.2fms",
            label, stats.n, stats.p50_ms, stats.p95_ms, stats.p99_ms, stats.mean_ms,
        )

    # ---------------- 并发 QPS（仅 Q2 场景，最贴近生产） ----------------
    if not args.no_concurrent and args.concurrency and args.concurrency != [0]:
        def q2_factory(local_rng: random.Random):
            return expr_q2(pools_dict, local_rng)

        for c in args.concurrency:
            log.info("Concurrency=%d, duration=%ds (Q2 expr factory)",
                     c, cfg.concurrency_duration_s)
            total, samples = run_concurrent_search(
                client, cfg, q2_factory, q_vecs, c, cfg.concurrency_duration_s,
            )
            qps = total / max(cfg.concurrency_duration_s, 1)
            stats = LatencyStats.from_samples(f"Q2_conc_{c}", samples)
            row = {
                "plan": cfg.scalar_index_plan,
                "vec_index": cfg.vector_index_type,
                "concurrency": c,
                "duration_s": cfg.concurrency_duration_s,
                "total_reqs": total,
                "qps": round(qps, 2),
                **stats.to_row(),
            }
            append_csv(conc_csv, row)
            log.info("  -> total=%d qps=%.1f p99=%.2fms", total, qps, stats.p99_ms)

    log.info("Done. Results saved under %s/", cfg.results_dir)


if __name__ == "__main__":
    main()
