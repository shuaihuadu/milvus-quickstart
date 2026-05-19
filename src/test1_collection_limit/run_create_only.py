"""
测试一 · Collection 数量上限 · 阶段 A：纯创建 + 极小 schema。

不 insert、不 load。目的：找出 standalone 在仅创建场景下的硬上限与失败模式。

用法：
    python -m src.test1_collection_limit.run_create_only --target 5000 --sample-every 50

中途 Ctrl+C 安全：会先把已有的采样落盘再退出。
"""
from __future__ import annotations

import argparse
import signal
import sys
import time
import traceback
from pathlib import Path

from pymilvus import DataType

from src.common.config import CFG, TestConfig
from src.common.logging_utils import get_logger
from src.common.metrics import append_csv, timer
from src.common.milvus_client import get_client

log = get_logger("test1_create_only")

_STOP = False


def _on_signal(signum, frame):
    global _STOP
    log.warning("Caught signal %s, will stop after current iteration ...", signum)
    _STOP = True


def parse_args(cfg: TestConfig) -> tuple[TestConfig, argparse.Namespace]:
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", default="t1c", help="批量创建的 collection 名前缀")
    p.add_argument("--target", type=int, default=cfg.coll_limit_target,
                   help="目标 collection 数（达到即停）")
    p.add_argument("--sample-every", type=int, default=cfg.coll_limit_sample_every,
                   help="每隔多少个 collection 采样一次")
    p.add_argument("--dim", type=int, default=cfg.coll_limit_min_dim,
                   help="极小向量维度（默认 8，省内存）")
    p.add_argument("--cleanup", action="store_true",
                   help="跑完或异常退出时尝试 drop 掉前缀匹配的 collection")
    args = p.parse_args()
    return cfg, args


def make_minimal_schema(client, dim: int):
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("v", DataType.FLOAT_VECTOR, dim=dim)
    return schema


def cleanup_prefix(client, prefix: str) -> int:
    try:
        names = client.list_collections()
    except Exception as e:
        log.error("list_collections failed during cleanup: %s", e)
        return 0
    cnt = 0
    for n in names:
        if n.startswith(prefix):
            try:
                client.drop_collection(n)
                cnt += 1
            except Exception as e:
                log.warning("drop %s failed: %s", n, e)
    log.info("Cleanup dropped %d collections with prefix '%s'", cnt, prefix)
    return cnt


def main() -> None:
    cfg, args = parse_args(CFG)
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    csv_path = Path(cfg.results_dir) / "test1_create_only.csv"
    log.info("Output CSV: %s", csv_path)

    client = get_client(cfg)
    try:
        existing = client.list_collections()
    except Exception as e:
        log.error("Initial list_collections failed: %s", e)
        existing = []
    log.info("Existing collections at start: %d", len(existing))

    created = 0
    last_err: str | None = None
    t_run_start = time.perf_counter()

    try:
        for i in range(1, args.target + 1):
            if _STOP:
                break
            name = f"{args.prefix}_{i:06d}"
            schema = make_minimal_schema(client, args.dim)
            try:
                with timer() as t:
                    client.create_collection(collection_name=name, schema=schema)
                created += 1
                create_ms = t.ms
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                log.error("create_collection failed at #%d (%s): %s", i, name, last_err)
                # 把失败点也作为一行写入
                append_csv(
                    csv_path,
                    {
                        "iteration": i,
                        "created_so_far": created,
                        "event": "create_failed",
                        "create_ms": -1,
                        "list_ms": -1,
                        "error": last_err,
                        "elapsed_s": round(time.perf_counter() - t_run_start, 2),
                    },
                )
                break

            if (i % args.sample_every == 0) or (i == args.target):
                # 采样：list_collections 延迟
                try:
                    with timer() as t_list:
                        names = client.list_collections()
                    list_ms = t_list.ms
                    list_n = len(names)
                except Exception as e:
                    list_ms = -1.0
                    list_n = -1
                    log.warning("list_collections failed at #%d: %s", i, e)

                append_csv(
                    csv_path,
                    {
                        "iteration": i,
                        "created_so_far": created,
                        "event": "sample",
                        "create_ms": round(create_ms, 2),
                        "list_ms": round(list_ms, 2),
                        "list_n": list_n,
                        "error": "",
                        "elapsed_s": round(time.perf_counter() - t_run_start, 2),
                    },
                )
                log.info(
                    "[%d] created=%d  create=%.1fms  list=%.1fms (n=%d)",
                    i, created, create_ms, list_ms, list_n,
                )

    except Exception:
        log.error("Unhandled exception:\n%s", traceback.format_exc())
    finally:
        log.info("Loop ended. created=%d, last_err=%s, wall=%.1fs",
                 created, last_err, time.perf_counter() - t_run_start)
        if args.cleanup:
            cleanup_prefix(client, args.prefix)


if __name__ == "__main__":
    main()
