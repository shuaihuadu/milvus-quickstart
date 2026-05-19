"""
Milvus 客户端封装：连接 / 健康检查 / collection 通用操作。
"""
from __future__ import annotations

from typing import Iterable

from pymilvus import MilvusClient

from .config import TestConfig
from .logging_utils import get_logger

log = get_logger(__name__)


def get_client(cfg: TestConfig) -> MilvusClient:
    kwargs: dict = {"uri": cfg.uri}
    if cfg.token:
        kwargs["token"] = cfg.token
    if cfg.db_name and cfg.db_name != "default":
        kwargs["db_name"] = cfg.db_name
    client = MilvusClient(**kwargs)
    return client


def drop_collection_if_exists(client: MilvusClient, name: str) -> bool:
    if client.has_collection(name):
        log.info("Dropping existing collection: %s", name)
        client.drop_collection(name)
        return True
    return False


def list_collections(client: MilvusClient) -> list[str]:
    return client.list_collections()


def get_load_state(client: MilvusClient, name: str) -> str:
    try:
        st = client.get_load_state(collection_name=name)
        return str(st)
    except Exception as e:
        return f"<err: {e}>"


def safe_release(client: MilvusClient, name: str) -> None:
    try:
        client.release_collection(collection_name=name)
    except Exception:
        pass
