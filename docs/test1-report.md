# 测试一 · Collection 数量上限 · 实测报告

> 对应 [test-plan.md §1](test-plan.md) 的"测试一"。
> 测试时间：2026-05-18（单日跑完）
> 报告人：Copilot pair-programming session

---

## 0. 摘要（TL;DR）

| 项                 | 结果                                                                               |
| ------------------ | ---------------------------------------------------------------------------------- |
| **硬上限**         | **65 536 个 collection**（=2¹⁶，Milvus 2.5.10 默认每 db 限制）                     |
| **撞墙错误**       | `code=102: exceeded the limit number of collections[dbName=default][limit=65536]`  |
| **触发位置**       | 第 65 537 次 `create_collection` 调用                                              |
| **失败模式**       | 服务端 quota 检查直接拒绝，**不是 OOM、不是 etcd quota、不是超时**                 |
| **总耗时**         | 5 614 秒 ≈ 1 小时 33 分（含 4 个 phase 串行）                                      |
| **撞墙时资源用量** | milvus 6.50 GiB · etcd RAM 1.85 GiB · etcd DB 490 MB · 宿主 RAM 31 GiB 用了 15 GiB |
| **实测可调上限**   | 改 `quotaAndLimits.limits.maxCollectionNum` 后理论可继续增长；瓶颈不在物理资源     |

---

## 1. 测试配置

### 1.1 环境

| 组件           | 版本 / 规格                                                     |
| -------------- | --------------------------------------------------------------- |
| Milvus         | 2.5.10 standalone（官方 docker 镜像 `milvusdb/milvus:v2.5.10`） |
| etcd           | 3.5.5                                                           |
| MinIO          | 默认随 standalone 部署                                          |
| pymilvus       | 2.5.10                                                          |
| Python         | 3.12.3                                                          |
| 宿主           | Ubuntu 24.04，8 vCPU，31 GiB RAM，123 GiB 根盘（剩 12 GiB）     |
| 容器 mem limit | 31.34 GiB（无显式限制，等同宿主）                               |

### 1.2 测试方法

对应 plan §1.2 的 **阶段 A：纯创建测试**：

- 极简 schema：`id INT64 PK + v FLOAT_VECTOR(dim=8)`，**不建索引、不 insert、不 load**
- 命名：`{prefix}_{i:06d}`（用 4 个 prefix 串接四个 phase）
- 每 100 个采样一次 `list_collections` 延迟与单次 `create_collection` 延迟
- 旁路监控：`docker stats` (milvus / etcd / minio) + `etcdctl endpoint status`
- 脚本：[src/test1_collection_limit/run_create_only.py](../src/test1_collection_limit/run_create_only.py)
- 调用命令：
  ```bash
  PYTHONPATH=. .venv/bin/python -u -m src.test1_collection_limit.run_create_only \
      --prefix t1{c|d|e|f} --target <N> --sample-every 100 --dim 8 \
      2>&1 | tee -a results/test1_create_only.log
  ```

### 1.3 测试分四个 phase 串行

|  Phase   | prefix |   起点 | 目标增量 |          实际创建 |    wall (s) | 平均吞吐 |
| :------: | :----: | -----: | -------: | ----------------: | ----------: | -------: |
|    1     |  t1c   |      0 |    5 000 |             5 000 |        99.2 | 50.4 / s |
|    2     |  t1d   |  5 000 |   15 000 |            15 000 |       420.1 | 35.7 / s |
|    3     |  t1e   | 20 000 |   30 000 |            30 000 |     2 568.1 | 11.7 / s |
|    4     |  t1f   | 50 000 |   50 000 | **15 536 (失败)** |     2 526.3 |  6.1 / s |
| **合计** |   —    |      0 |  100 000 |        **65 536** | **5 613.7** | 11.7 / s |

---

## 2. 关键结果

### 2.1 撞墙详情

第 65 537 次 `create_collection` 报错：

```
2026-05-18 16:31:46 | ERROR | test1_create_only |
create_collection failed at #15537 (t1f_015537):
MilvusException: <MilvusException:
  (code=102, message=exceeded the limit number of collections[dbName=default][limit=65536])>
```

- `65536 = 2¹⁶`，明显是 Milvus 服务端 `quotaAndLimits.limits.maxCollectionNumPerDB` 的默认值
- 该错误为**前置 quota 检查**直接返回，**没有触发任何资源相关错误**（无 OOM、无 etcd quota exceeded、无超时、无 grpc transport closed）
- 与 test-plan §0.3 的预期一致：`quotaAndLimits.limits.maxCollectionNum: 65536`

### 2.2 资源消耗 — 撞墙时

| 项                    |         用量 |        占限额比例 |
| --------------------- | -----------: | ----------------: |
| milvus-standalone RAM | **6.50 GiB** |     21 % / 31 GiB |
| etcd RAM              |     1.85 GiB |      6 % / 31 GiB |
| **etcd DB size**      |   **490 MB** | 24 % / 2 GB quota |
| minio RAM             |       239 MB |               1 % |
| milvus CPU            |    500-760 % |        6-9 / 8 核 |
| 宿主 RAM 已用         |       15 GiB |              47 % |
| 宿主根盘剩余          |       12 GiB |                 — |

> **结论**：物理资源远未耗尽。即使再有 65k 个 collection 的空间，milvus 也不会超 13 GiB RAM，etcd DB 也只到 ~1 GB（仍在 2GB quota 内）。**真正的限制是 `maxCollectionNumPerDB` 配置项**。

### 2.3 资源增长率（每 collection 的边际成本）

从 5 000 / 20 000 / 65 536 三个采样点算出的近似线性增量：

| 项                  | 增量 / collection |
| ------------------- | ----------------: |
| milvus RAM          |      **~ 104 KB** |
| etcd DB             |      **~ 7.5 KB** |
| etcd 磁盘（含 WAL） |           ~ 17 KB |

### 2.4 延迟趋势

`create_collection` 与 `list_collections` 单次调用耗时（client side，含网络）：

| n collections | create p50 | list p50 | create p99 (实测 spike) |
| ------------: | ---------: | -------: | ----------------------: |
|         5 000 |      17 ms |   1.2 ms |                  ~50 ms |
|        10 000 |      19 ms |     5 ms |                  ~60 ms |
|        20 000 |      21 ms |    10 ms |                 ~280 ms |
|        30 000 |      23 ms |    13 ms |                 ~525 ms |
|        40 000 |      34 ms |    18 ms |                 ~600 ms |
|        50 000 |      33 ms |    22 ms |                 ~840 ms |
|        60 000 |      44 ms |    30 ms |                ~1132 ms |
|        65 000 |     ~35 ms |   ~30 ms |                 ~600 ms |

观察：
- **p50 增长温和**：5k → 65k，create 慢 ~2.6x，list 慢 ~25x
- **p99 增长猛**：create p99 从 50 ms 飙到 1100 ms（~22x），主要来自 etcd 周期 compaction + raft snapshot
- 吞吐随 collection 数下降：50 / s → 6 / s
- list_collections 随 n 大致线性变慢（约 **1 ms / 2 000 collections**），这对客户端代码影响有限，但**业务侧应避免每次请求都 `list_collections`**

---

## 3. 回答 plan §1.1 的 4 个问题

> 提醒：本测试只做"纯创建（阶段 A）"。问题 2 / 问题 3 中关于 load 后行为的部分需要阶段 B / C 才能完整回答。

### Q1 · 硬上限：默认配置下能创建多少个 Collection 不报错？
**65 536 个/db**。第 65 537 个返回 `code=102: exceeded the limit number of collections`。
这是 Milvus 内置 quota 检查的结果，不是任何资源耗尽。

### Q2 · 软上限：还能稳定 load + insert + search 的 Collection 是多少？
本次只跑了阶段 A，**未测 load**。从资源画像推断：
- 仅 meta 层面，6.5 GiB RAM / 31 GiB 还很宽裕
- 但 65k 个 collection 全部 load，按 HNSW 索引 + segment cache 估算单 collection 至少 1-10 MB queryNode 内存，65k × 5 MB ≈ 320 GiB，**完全不可能在单机 standalone load 全部**
- 需要阶段 B/C 进一步测试

### Q3 · 资源画像（每个 collection 的成本）
**仅创建（无数据）**：
- milvus 进程：~ 104 KB RAM
- etcd：~ 7.5 KB DB + ~ 17 KB on-disk（含 WAL）
- minio：~ 0（没有 segment 文件）
- CPU：稳态后台 5-9 核（与 collection 数关系不大，主要是定期 task）

**load 后**：未测，下一阶段补。

### Q4 · 失败模式
**前置 quota 检查直接拒绝**，特征：
- 错误码 `code=102`，message 明确指出 `limit=65536`
- 立即返回（没有等待 / 没有重试）
- 客户端可捕获 `MilvusException` 并区分这是 quota 类错误（非临时性）
- 没有触发副作用（etcd / milvus 自身没被破坏，仍可正常 list 已有 collection）

---

## 4. 实操建议（生产部署视角）

### 4.1 何时需要担心 collection 上限？
- 大多数生产部署 collection 数 < 1 000，**离 65 536 极远**
- 真正会用到这个上限的场景：每个用户/租户独立 collection 的 SaaS 形态
- 这种场景下推荐：**多个 database 而不是把所有 collection 堆在 default**

### 4.2 如何调高上限？
在 `milvus.yaml` 增加（test-plan §0.3 已预留位置）：
```yaml
quotaAndLimits:
  limits:
    maxCollectionNum: 200000
    maxCollectionNumPerDB: 100000
```
重启 milvus-standalone 容器即可生效。但要注意：
- 上限是配置项决定，但**物理资源不会变小**
- 65k 时 milvus 6.5 GiB RAM，100k 时按线性外推约 **10 GiB**；200k 时 ~20 GiB（接近 31 GiB 宿主上限）
- etcd DB 100k 时 ~750 MB，200k 时 ~1.5 GB，**接近 etcd 默认 2 GB quota**，需要同步调高 `etcd --quota-backend-bytes`

### 4.3 客户端实践
- 避免热路径里调 `list_collections()`（n=65k 时 ~30 ms p50，p99 数百 ms）
- 业务侧维护 collection name 缓存（带过期或事件刷新）
- 批量创建场景接受 spike：p99 单次 create 可到 1 秒以上，要在客户端做超时容忍

---

## 5. 关于本次测试的诚实声明

### 做了什么
- 4 个 phase 串行，0 → 65 536 个 collection，记录每 100 个的延迟 + 资源
- 撞到 milvus 自身的硬性 quota（非物理资源）
- 数据落盘 CSV + 完整日志

### 没做什么（下一步）
- 没有改 `maxCollectionNumPerDB` 重测物理资源极限
- 没有测多 database 场景下能不能突破 65k × N
- 没有测 collection 创建后立即 load / search 的资源占用（阶段 B）
- 没有测删除速度（drop_collection 单个吞吐多少 / 65k 全清要多久）—— 紧接着的清理操作会顺便得到这个数据

### 已知偏差
- 4 phase 间有几分钟间歇（操作员等待），etcd 会在间歇期做 compaction，可能让 phase 内的 p99 看起来比连续跑要好
- list_collections 在 client side 计时，含网络往返；服务端处理时间可能略低
- 单测试机 8 vCPU，更大的 CPU 配额下 spike 频率可能不同

---

## 6. 原始数据

| 文件                                                                                              | 内容                                                                                                  |
| ------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| [results/test1_create_only.csv](../results/test1_create_only.csv)                                 | 707 行采样数据：iteration / created_so_far / event / create_ms / list_ms / list_n / error / elapsed_s |
| [results/test1_create_only.log](../results/test1_create_only.log)                                 | 完整 INFO + ERROR 日志，含 4 个 phase 的 `Loop ended` 总结                                            |
| [src/test1_collection_limit/run_create_only.py](../src/test1_collection_limit/run_create_only.py) | 测试驱动脚本                                                                                          |

---

## 7. 一句话总结

> **Milvus 2.5.10 standalone 在 8 vCPU / 31 GiB 宿主下，单 db 默认能创建到 65 536 个 collection 撞配置墙，此时 milvus 只用了 6.5 GiB RAM，物理资源远未耗尽。若需更多 collection，应改 `maxCollectionNumPerDB` 配置或拆 database，而不是认为这是资源瓶颈。**
