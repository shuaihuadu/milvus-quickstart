"""
随机数据生成：embedding + 标量字段。

业务模型：Collection 存的是文档的 chunk。
  - 先决定文档边界：每文档 chunk 数 ∼ U(chunks_per_doc_min, chunks_per_doc_max)
  - 同一文档的所有 chunk 共享：documentId / kbId / study / site / patient / visit
  - 不同文档的文档级字段独立采样；embedding 仍按 chunk 独立采样
  - 总 chunk 达到 total_rows 后立即终止（最后一个文档可能被截断）

设计要点：
1. 不在内存里持有全量数据。按 batch 在生成器里出数据，配合流式 insert。
2. 标量字段值池在初始化时确定，整个跑测过程值集合稳定（便于复现查询）。
3. 低基数字段支持 Zipfian 偏斜 —— 通过给字典中每个值赋一个 1/rank^alpha 的概率。
4. embedding 用 float32 随机 + L2 归一化（IP 等价于 cosine）。
"""
from __future__ import annotations

import string
from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np


@dataclass
class ScalarPools:
    """每个字段的取值池（事先定好，便于查询时引用）。"""

    kb: list[str]
    study: list[str]
    site: list[str]
    patient: list[str]
    visit: list[str]

    def probs(self, key: str, alpha: float) -> np.ndarray:
        pool = getattr(self, key)
        ranks = np.arange(1, len(pool) + 1, dtype=np.float64)
        w = 1.0 / np.power(ranks, alpha)
        return w / w.sum()


def make_scalar_pools(
    kb_card: int,
    study_card: int,
    site_card: int,
    patient_card: int,
    visit_card: int,
) -> ScalarPools:
    return ScalarPools(
        # kbId 形如 kb_00..kb_09（默认 10 个）。业务侧如为 UUID 仅改这里。
        kb=[f"kb_{i:02d}" for i in range(kb_card)],
        study=[f"S{i:03d}" for i in range(study_card)],
        site=[f"SITE_{i:03d}" for i in range(site_card)],
        patient=[f"P{i:05d}" for i in range(patient_card)],
        visit=[f"V{i:02d}" for i in range(visit_card)],
    )


def _make_content_template(length: int) -> str:
    """固定模板，避免每行都做字符串构造。"""
    alphabet = string.ascii_letters + string.digits + " "
    rng = np.random.default_rng(0)
    chars = rng.choice(list(alphabet), size=length)
    return "".join(chars.tolist())


class BatchDataGenerator:
    """按批次产出 (rows: list[dict])。每行字段与 collection schema 对齐。

    chunk 模型：内部跟踪当前文档（kb/documentId/study/site/patient/visit）与
    剩余 chunk 数。当当前文档 chunks 耗尽时才开启新文档（重新采样文档级字段）。
    """

    def __init__(
        self,
        total_rows: int,
        batch_size: int,
        dim: int,
        pools: ScalarPools,
        zipf_alpha: float = 1.0,
        content_length: int = 256,
        seed: int = 42,
        start_id: int = 0,
        # 文档切片模型
        chunks_per_doc_min: int = 20,
        chunks_per_doc_max: int = 500,
        # 稀疏向量配置（多路召回）
        enable_sparse: bool = False,
        sparse_vocab_size: int = 30000,
        sparse_nnz_per_row: int = 64,
    ) -> None:
        self.total_rows = int(total_rows)
        self.batch_size = int(batch_size)
        self.dim = int(dim)
        self.pools = pools
        self.alpha = float(zipf_alpha)
        self.start_id = int(start_id)
        self.rng = np.random.default_rng(seed)
        self.content_template = _make_content_template(content_length)
        self.chunks_per_doc_min = int(chunks_per_doc_min)
        self.chunks_per_doc_max = int(chunks_per_doc_max)
        if self.chunks_per_doc_min < 1 or self.chunks_per_doc_max < self.chunks_per_doc_min:
            raise ValueError(
                f"invalid chunks_per_doc range: [{self.chunks_per_doc_min}, {self.chunks_per_doc_max}]"
            )
        self.enable_sparse = bool(enable_sparse)
        self.sparse_vocab_size = int(sparse_vocab_size)
        self.sparse_nnz_per_row = int(sparse_nnz_per_row)

        # 预计算分布概率（kbId 也带同一 zipfian【0=均匀】）
        self._probs = {
            "kb": pools.probs("kb", self.alpha),
            "study": pools.probs("study", self.alpha),
            "site": pools.probs("site", self.alpha),
            "patient": pools.probs("patient", self.alpha),
            "visit": pools.probs("visit", self.alpha),
        }

        # 文档级状态（生成过程中维护）
        self._doc_counter: int = 0
        self._remaining_in_doc: int = 0
        self._cur_doc_attrs: dict = {}

    @property
    def num_batches(self) -> int:
        return (self.total_rows + self.batch_size - 1) // self.batch_size

    # ---------- 采样辅助 ----------

    def _sample_one(self, key: str) -> str:
        pool = getattr(self.pools, key)
        idx = int(self.rng.choice(len(pool), p=self._probs[key]))
        return pool[idx]

    def _advance_to_next_doc(self) -> None:
        """当前文档 chunk 耗尽，采样下一份文档的所有共享属性。"""
        self._doc_counter += 1
        # 文档边界：U(min, max)，inclusive
        self._remaining_in_doc = int(
            self.rng.integers(self.chunks_per_doc_min, self.chunks_per_doc_max + 1)
        )
        self._cur_doc_attrs = {
            "documentId": f"doc_{self._doc_counter:08d}",
            "kbId": self._sample_one("kb"),
            "study": self._sample_one("study"),
            "site": self._sample_one("site"),
            "patient": self._sample_one("patient"),
            "visit": self._sample_one("visit"),
        }

    def _sample_vectors(self, n: int) -> np.ndarray:
        v = self.rng.standard_normal(size=(n, self.dim)).astype(np.float32)
        # L2 归一化，使 IP 等价 cosine
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        v /= norms
        return v

    def _sample_sparse(self, n: int, nnz: int | None = None) -> list[dict]:
        """生成 n 个稀疏向量，每个为 {token_id: weight} dict。

        - 位置：从 vocab 里无放回采 nnz 个
        - 权重：U(0,1)（模拟 BM25/SPLADE 归一化后的分布）
        pymilvus 2.5.10 原生接受这种 dict 形式作为 SPARSE_FLOAT_VECTOR 行。
        """
        k = int(nnz if nnz is not None else self.sparse_nnz_per_row)
        k = min(k, self.sparse_vocab_size)
        out: list[dict] = []
        for _ in range(n):
            idx = self.rng.choice(self.sparse_vocab_size, size=k, replace=False)
            vals = self.rng.random(k).astype(np.float32)
            out.append({int(i): float(v) for i, v in zip(idx, vals)})
        return out

    # ---------- 批次迭代 ----------

    def batches(self) -> Iterator[list[dict]]:
        produced = 0
        next_id = self.start_id
        while produced < self.total_rows:
            n = min(self.batch_size, self.total_rows - produced)
            ids = np.arange(next_id, next_id + n, dtype=np.int64)
            embeddings = self._sample_vectors(n)
            sparse = self._sample_sparse(n) if self.enable_sparse else None

            rows = []
            for i in range(n):
                # 需要时切换到下一份文档
                if self._remaining_in_doc <= 0:
                    self._advance_to_next_doc()
                self._remaining_in_doc -= 1

                row = {
                    "id": int(ids[i]),
                    "kbId": self._cur_doc_attrs["kbId"],
                    "documentId": self._cur_doc_attrs["documentId"],
                    "embedding": embeddings[i].tolist(),
                    "content": self.content_template,
                    "study": self._cur_doc_attrs["study"],
                    "site": self._cur_doc_attrs["site"],
                    "patient": self._cur_doc_attrs["patient"],
                    "visit": self._cur_doc_attrs["visit"],
                }
                if sparse is not None:
                    row["sparse_embedding"] = sparse[i]
                rows.append(row)
            yield rows
            produced += n
            next_id += n

    @property
    def total_docs_generated(self) -> int:
        """迭代完后的文档总数（只在 batches() 走完后有义）。"""
        return self._doc_counter

    # 仅用于查询向量的快速生成
    def sample_query_vectors(self, n: int) -> list[list[float]]:
        return self._sample_vectors(n).tolist()

    def sample_query_sparse(self, n: int, nnz: int | None = None) -> list[dict]:
        """生成 n 个查询用稀疏向量。默认用查询侧的 nnz（一般比插入侧小）。"""
        return self._sample_sparse(n, nnz=nnz)

