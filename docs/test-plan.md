# Milvus 2.5.10 测试规划

> 目标：在 Milvus 2.5.10 Standalone 上完成两类测试
> 1. **Collection 数量上限**：能稳定承载多少个 Collection
> 2. **单 Collection 标量筛选性能**：在 10M 行规模下，按 `documentId/study/site/patient/visit` 过滤后做向量召回的性能特征

---

## 0. 测试前置与环境

### 0.1 软件版本

| 组件     | 版本   | 备注                                                   |
| -------- | ------ | ------------------------------------------------------ |
| Milvus   | 2.5.10 | Standalone，官方 Docker 镜像 `milvusdb/milvus:v2.5.10` |
| etcd     | 3.5.x  | 随 standalone compose 自带                             |
| MinIO    | latest | 随 standalone compose 自带                             |
| pymilvus | 2.5.10 | 与 server 版本对齐，避免协议差异                       |
| Python   | 3.10+  |                                                        |

### 0.2 部署形态

- 单机 Docker Compose（官方 `milvus-standalone-docker-compose.yml`）
- 数据盘建议 SSD，因为 10M × 3072 dim 的 binlog/index 落盘量较大
- 单机模式下 rootCoord/dataCoord/queryCoord/indexCoord 都在一个进程里，配置文件只有一份 `milvus.yaml`

### 0.3 关键配置项预留位（先跑默认，遇瓶颈再调）

```yaml
# milvus.yaml 里可能需要调整的项
rootCoord:
  maxDatabaseNum: 64               # 默认 64
  maxPartitionNum: 1024            # 单 collection 最大 partition 数（默认 1024）
proxy:
  maxFieldNum: 64                  # 单 collection 字段数上限
  maxShardNum: 16                  # 单 collection shard 数上限
quotaAndLimits:
  limits:
    maxCollectionNum: 65536        # 集群 Collection 上限（**测试 1 的核心**）
    maxCollectionNumPerDB: 65536
queryNode:
  mmap:
    mmapEnabled: false             # 3072 维 10M 行场景下可能需要打开
    vectorField: false
    vectorIndex: false
common:
  retentionDuration: 86400
```

> ⚠️ Collection 数量上限不只看 `maxCollectionNum`。真正限制因素是：
> - rootCoord 内存（每个 collection 的 meta cache）
> - etcd 中 meta key 数量与 watch 性能
> - 加载到 queryNode 的 segment / channel 数
> - proxy 端 schema cache

---

## 1. 测试一：Collection 数量上限

### 1.1 测试目标

回答 4 个问题：

1. **硬上限**：默认配置下，能创建多少个 Collection 不报错？
2. **软上限**：创建后还能稳定 load + insert + search 的 Collection 是多少？
3. **资源画像**：每个空 Collection、每个 load 后的 Collection，分别消耗多少内存 / etcd key？
4. **失败模式**：达到上限时，先报什么错？是 create 失败、load 失败，还是 search 超时？

### 1.2 测试方法

#### 阶段 A：纯创建测试（不 insert、不 load）

- 循环创建 collection：`coll_0001`, `coll_0002`, ...
- 每个 collection 用最小 schema（id + 一个 8 维 float vector，不建索引）
- 每 100 个采样一次：
  - `utility.list_collections()` 耗时
  - 进程 RSS 内存（`docker stats`）
  - etcd db size（`etcdctl endpoint status --write-out=table`）
  - 单次 create 调用耗时
- 终止条件：create 报错 或 内存 > 80% 或 单次 create > 5s

#### 阶段 B：创建 + insert 少量数据 + load

- 同样递增创建，但每个 collection 插入 1000 行 + load
- 关注 queryNode 内存增长
- 终止条件：load 失败 / queryNode OOM / search 延迟 > 1s

#### 阶段 C：贴近真实场景

- 每个 collection 插入 1 万行 + 建 HNSW + load
- 用更小一批（比如 100~500 个）观察增长曲线
- 目的：算出"每个 collection 平均占用多少内存"，从而预估单机能撑住多少业务 collection

### 1.3 待采集指标

| 指标                   | 来源           | 采集频率   |
| ---------------------- | -------------- | ---------- |
| collection 数          | client 计数    | 每次创建后 |
| create_collection 延迟 | client 计时    | 每次       |
| list_collections 延迟  | client 计时    | 每 100 个  |
| Milvus 容器 RSS        | `docker stats` | 每 100 个  |
| etcd db size           | etcdctl        | 每 100 个  |
| 错误类型与时间点       | 异常 stack     | 触发时     |

### 1.4 预期 & 风险

- 官方默认 `maxCollectionNum=65536`，但 standalone 实际跑不到这个数
- 经验上 standalone 在几千 collection 量级就会因为 rootCoord meta + queryNode load 而出现问题
- Milvus 2.5.x 引入了 collection-level 的 lazy load，对上限有正向影响
- 风险：测试机内存若小（如 16 GB），阶段 B/C 可能在几百个 collection 就 OOM

### 1.5 输出

- `results/collection_limit.csv`：每个采样点一行
- `results/collection_limit.png`：内存 vs collection 数 曲线
- 文字结论：「在 X GB 内存 standalone 下，建议单实例 collection 数不超过 N」

---

## 2. 测试二：单 Collection 标量筛选性能

### 2.1 Schema 设计

| 字段               | 类型                     | 索引                       | 说明                                                                              |
| ------------------ | ------------------------ | -------------------------- | --------------------------------------------------------------------------------- |
| `id`               | INT64, PK, auto_id=False | —                          | 主键                                                                              |
| `kbId`             | VARCHAR(16)              | BITMAP **+ partition_key** | **知识库 ID**，基数 ~10（业务确认），不可 null；详见 §2.3 S5 与 §2.7              |
| `documentId`       | VARCHAR(64)              | INVERTED                   | 文档 ID，**高基数**（≈ 38,500 个文档），同一 doc 的多个 chunk 共享同一 documentId |
| `embedding`        | FLOAT_VECTOR(3072)       | HNSW or IVF_FLAT           | dense 向量字段                                                                    |
| `sparse_embedding` | SPARSE_FLOAT_VECTOR      | SPARSE_INVERTED_INDEX      | **多路召回**用的稀疏向量（BM25/SPLADE 风格）                                      |
| `content`          | VARCHAR(2048)            | 无                         | 文本内容，仅返回不过滤                                                            |
| `study`            | VARCHAR(32)              | BITMAP                     | **低基数 ~几十**                                                                  |
| `site`             | VARCHAR(32)              | BITMAP                     | **低基数 ~几十**                                                                  |
| `patient`          | VARCHAR(64)              | BITMAP                     | **低基数 ~几十**（按用户描述）                                                    |
| `visit`            | VARCHAR(32)              | BITMAP                     | **低基数 ~几十**                                                                  |

> 说明：
> - **kbId**：业务场景下 10 个知识库，几乎每个 query 都带 `kbId == X` 过滤。**确定作为 `partition_key`**（§2.3 S5），隔离 + 加速一起做掉。同时保留 BITMAP 索引（低基数 + 等值过滤最佳组合）。作为 partition_key 后不能再手动创建 partition。
> - **文档切片模型**：一个文档随机产生 20–500 个 chunk（均值 ≈ 260）。总量 10M chunk ⇒ ≈ 38,500 个文档，均剀1 KB ≈ 3,850 个文档 / 1M 个 chunk。同一文档的所有 chunk 共享 `documentId` 与 `kbId`。
> - 用户场景下 patient 居然只有几十，意味着这是"病人级别聚合后"的逻辑分组而不是真实病人 ID。如果实际是按病人粒度，basis 会到万级，要换 INVERTED。**写代码时把基数做成可配置参数**。
> - documentId 走 INVERTED：高基数等值匹配场景，BITMAP 会爆炸
> - 几十基数的字段走 BITMAP：Milvus 2.5 BITMAP 在低基数 IN/EQ 上是最快的
> - **稀疏向量**：仅支持 metric=IP；用 dict `{token_id: weight}` 注入；用 `SPARSE_INVERTED_INDEX`。生成策略：固定词表（默认 30000）+ 每行 nnz=64 个非零项（U(0,1) 权重）。这一组合接近 BM25 / SPLADE 蒸馏后的稀疏向量量级
> - 也要对照测一组 **不建标量索引** 的 baseline，证明索引带来的提升
> - 多路召回入口：`MilvusClient.hybrid_search(reqs=[dense_req, sparse_req], ranker=...)`，默认 RRFRanker(k=60)，可换 WeightedRanker

### 2.2 数据生成策略

- 总量：10,000,000 行（chunk 粒度）
- **文档 → chunk 展开**：先采样出 N 个文档（`documentId`），每个文档随机 `chunks_per_doc ∼ Uniform(20, 500)`，使总 chunk 数 ≈ 10M（超过后截断最后一个文档）。预期文档总数 ≈ 38,500。
- embedding：随机 float32 向量，归一化（cosine 等价于内积）。**维度可配**（代码默认 3072，可调为 768/1024 适配不同机器）
- documentId：`doc_{i:08d}`，同一文档的所有 chunk 共享
- **kbId**：基数 K = 10（默认，可配），形如 `kb_00`..`kb_09`。默认**温和 Zipfian 偏斜**（s=1.0 附近，最大 KB 占 ~25% 总数据，最小 KB 占 ~2%）；也可切换到均匀分布。文档属于哪个 KB 在生成阶段决定，**同一文档的所有 chunk 一定属于同一 KB**
- study/site/patient/visit：从基数 N（默认 30，可配置）的预生成池里**带分布偏斜**采样
  - 用 Zipfian 分布而不是均匀分布，更接近真实数据
  - 这样能测出"热点值过滤"（命中 30% 数据）和"冷门值过滤"（命中 0.1% 数据）的性能差异
  - **文档级还是 chunk 级字段**：默认按文档级采样（同文档的 chunk 共享 study/site/patient/visit），更贴近现实
- content：固定长度占位字符串（避免生成开销）
- 分批插入：每批 5 万 ~ 10 万行，flush 一次后再下一批

### 2.3 索引策略矩阵

向量索引固定一种先跑出基线，再扩展：

| 向量索引 | metric | 参数                     | 用途                              |
| -------- | ------ | ------------------------ | --------------------------------- |
| HNSW     | IP     | M=16, efConstruction=200 | 主测对象                          |
| IVF_FLAT | IP     | nlist=4096               | 对照（filter 选择性高时表现更好） |

标量索引矩阵（每个 case 单独 build + load + 测）：

| Case | kbId   | documentId | study/site/patient/visit | partition_key | 备注                                                                  |
| ---- | ------ | ---------- | ------------------------ | ------------- | --------------------------------------------------------------------- |
| S0   | 无     | 无         | 无                       | —             | Baseline，brute force filter（不用 partition_key）                    |
| S1   | BITMAP | INVERTED   | 无                       | —             | 只索引高基数字段 + kbId                                               |
| S2   | BITMAP | 无         | BITMAP                   | —             | 只索引低基数字段                                                      |
| S3   | BITMAP | INVERTED   | BITMAP                   | —             | **推荐生产配置（不用 partition_key）**                                |
| S4   | BITMAP | INVERTED   | INVERTED                 | —             | 对比 BITMAP vs INVERTED 在低基数上的差异                              |
| S5   | BITMAP | INVERTED   | BITMAP                   | **kbId**      | **生产首选**：S3 + kbId 作 partition_key，`num_partitions=16`（默认） |

> **关于 S5 的 `num_partitions`**：10 个 kbId，hash 到 16 个分区上，期望每个分区 0–2 个 kbId（不会空太多，也不会振荡）。如需更严格隔离，可调到 32。**partition_key 使用后不能手动创建 partition，也不能在 query 中用 `partition_names` 参数**，所有分区路由靠引擎自动完成。

### 2.4 查询场景

每个查询带 `top_k = 10`，向量为随机查询向量（事先准备 1000 条 query set，重复使用做暖机）。

> **基线约定**：测试二的所有 query 默认前置 `kbId == "kb_xxx"`（贴近真实业务）。Q1/Q2/Q3 表达式里都包含 kbId。Q7 专门做"有/无 kbId 过滤"对照。

**Q1 — 单字段等值过滤 + 向量召回**
```python
expr = 'kbId == "kb_03" && study == "S05"'
```

**Q2 — 多字段 AND 过滤 + 向量召回**（核心场景）
```python
expr = 'kbId == "kb_03" && study == "S05" && site == "SITE_03" && patient == "P12" && visit == "V2"'
```

**Q3 — 高基数等值（documentId IN list）**
```python
expr = 'kbId == "kb_03" && documentId in ["doc_00000123", "doc_00005678", ... 100 items ...]'
```

**Q4 — 多字段过滤命中率扫描**
不同 expr 故意控制命中 hit_rate ∈ {0.01%, 0.1%, 1%, 10%, 30%}，看 expr 选择性对延迟的影响曲线。固定带 kbId 过滤，再叠加 study/site/patient/visit 调节命中率。

**Q5 — 纯标量 query（不带向量）**
```python
collection.query(expr=Q2_expr, limit=100, output_fields=["id","kbId","documentId"])
```
对照向量 search 的延迟，看出"标量过滤本身"的开销。

**Q6 — 多路召回 hybrid_search（dense + sparse）+ Q2 表达式**
```python
req_dense = AnnSearchRequest(
    data=[q_dense], anns_field="embedding",
    param={"metric_type":"IP","params":{"ef":64}}, limit=10, expr=Q2_expr,
)
req_sparse = AnnSearchRequest(
    data=[q_sparse], anns_field="sparse_embedding",
    param={"metric_type":"IP","params":{}}, limit=10, expr=Q2_expr,
)
client.hybrid_search(
    collection_name="...",
    reqs=[req_dense, req_sparse],
    ranker=RRFRanker(60),     # 或 WeightedRanker(1.0, 1.0)
    limit=10,
    output_fields=["kbId","documentId","study","site","patient","visit"],
)
```
关注：dense-only vs hybrid 的延迟差（多了一路稀疏 ANN + fusion），以及不同 ranker 对延迟/召回的影响。

**Q7 — kbId 过滤效果与 partition_key 加速对照**
- **Q7a**：`kbId == "kb_03"`（单 KB，覆盖业务 95% 场景）
- **Q7b**：`kbId in ["kb_03","kb_07","kb_09"]`（多 KB 联合，少数跨库场景）
- **Q7c**：**不带** kbId 过滤（全库 ANN，作为"无 KB 隔离"对照）
- 在 S3（kbId BITMAP，普通索引） 和 S5（kbId 作为 partition_key）下各测一遍，**直接量化分区裁剪带来的延迟降幅**
- 同时按 kbId 大小分桶测一次：热门大 KB（占 ~25% 数据，默认 Zipfian 下最大那个） / 中等 KB（~10%）/ 冷门小 KB（~2%），看分布偏斜下的延迟差异

### 2.5 测试流程

```
对每个 (索引方案 S0..S4) × (查询 Q1..Q5):
  1. drop_index → build_index → load
  2. warmup：100 次同样的查询（不计时）
  3. 串行测试：1000 次查询，记 p50 / p95 / p99
  4. 并发测试：concurrency ∈ {1, 4, 16, 32}，每档跑 60s，记 QPS 与 p99
  5. 释放：release()
```

### 2.6 待采集指标

| 指标                             | 单位   | 说明                                        |
| -------------------------------- | ------ | ------------------------------------------- |
| build_index 耗时                 | s      | 每种索引建一次                              |
| load 耗时                        | s      |                                             |
| query/search latency p50/p95/p99 | ms     | 串行 1000 次                                |
| QPS                              | req/s  | 并发 60s 平均                               |
| 召回率 recall@10                 | —      | 与 brute force 对比（S0 作为 ground truth） |
| 命中行数 nq_hit                  | 行     | 验证 filter 选择性符合预期                  |
| Milvus 容器 CPU/Mem              | % / MB | search 期间峰值                             |

### 2.7 预期 & 关注点

- **BITMAP 应该明显优于 INVERTED**（在几十基数下）— 这是要验证的核心假设之一
- 多字段 AND 时，Milvus 2.5 用 expr 优化器，BITMAP 可以并行 AND，性能不会随字段数线性下降
- 命中率 1%~10% 区间是 HNSW + filter 的"难点区"（既不能纯走 ANN，也不能纯走 filter）
- documentId IN 列表过大（>1000）可能触发慢路径，要注意
- 3072 维 + 10M 行，HNSW 索引会非常大（粗估 ~150 GB）：机器内存 < 200 GB 时 mmap 几乎一定要开（`queryNode.mmap.vectorIndex=true`）
- **kbId as partition_key（S5）应显著优于普通 BITMAP（S3）** ——单 KB 查询时 hash 路由只扫 1–2 个分区，避开整库 ANN；预期 S5 延迟为 S3 的 1/8 到 1/16。热分区风险在 10 个 kbId / 16 分区下很低（即使默认 Zipfian 偏斜，最大分区也不超过 ≈30% 数据，未超出没有 partition_key 时的全量）。
- **同一文档多 chunk 的影响**：Q3 里 `documentId in [...]` 的查询，单个 documentId 会命中 20–500 行，需要注意 filter 选择性不是 1/N 而是 ~260/N

---

## 3. 仓库结构规划

```
milvus-quickstart/
├── docs/
│   ├── test-plan.md             # 本文档
│   ├── results-collection-limit.md   # 测试一结论
│   └── results-scalar-filter.md      # 测试二结论
├── src/
│   ├── common/
│   │   ├── milvus_client.py     # 封装连接、collection 操作
│   │   ├── data_gen.py          # 随机数据生成（向量 + 标量字段，可配基数与偏斜）
│   │   └── metrics.py           # 延迟/QPS 统计、CSV 输出
│   ├── test1_collection_limit/
│   │   ├── run_create_only.py   # 阶段 A
│   │   ├── run_create_load.py   # 阶段 B / C
│   │   └── monitor.py           # docker stats / etcd 采样
│   └── test2_scalar_filter/
│       ├── prepare_dataset.py   # 生成并插入 10M 行
│       ├── build_indexes.py     # S0..S4 索引方案切换
│       ├── run_queries.py       # Q1..Q5 串行 + 并发
│       └── recall_check.py      # 召回率验证
├── docker/
│   └── milvus-standalone-compose.yml
├── results/                     # CSV / 图表输出
└── requirements.txt
```

---

## 4. 执行顺序建议

1. **环境**：拉起 standalone，pymilvus 装好，跑通 hello_world（一个 collection、插入、search）
2. **测试一阶段 A**：纯创建，半小时内能跑到瓶颈，先建立直觉
3. **测试二准备**：生成 10M 数据 + 建 S0 索引 + load（这一步最耗时，可能要数小时，先启动起来）
4. **测试二执行**：S0..S4 跑完 Q1..Q5
5. **测试一阶段 B/C**：放在最后跑，因为会反复占满内存
6. **报告**：把 CSV / 图表整理成 `results-*.md`

---

## 5. 待用户确认的开放问题

1. **跑数据机器配置**：用户表示会换高配置机器跑。本仓库在开发机上最多跑到 1M 行 smoke；正式 10M × 3072 维预计需 ≈ 200 GB 磁盘 + ≈ 32 GB 内存 × mmap 或 ≈ 64+ GB 全 load。
2. **跨平台：代码必须同时支持 Linux 和 Windows**：
   - 路径全用 `pathlib.Path`，不硬编码分隔符
   - 不在入口脚本依赖 `tee` / `&&` 等 shell 构造，**Python 内部自带 file logger**
   - 不走 `subprocess` 去调 `docker stats` / `etcdctl`：容器指标走 **Milvus `:9091/metrics` Prometheus 端点** 或 **`docker-py` Python 客户端**
   - 日志 / 输出文件名不包含 `:` `*` `?` `<` `>` `|` `"` 等 Windows 非法字符，时间戳用 `%Y%m%d_%H%M%S`
   - 多进程只用 `multiprocessing` 默认 API（Windows 上 spawn），不依赖 fork
3. **patient 字段实际基数**：用户答"几十"，但生产场景下如果到几万级，索引选型要换 INVERTED — 代码做成可配置
4. **kbId 与 partition_key（已确认）**：
   - kbId 基数 = **10**，作为 `partition_key`，`num_partitions=16`（默认）
   - 代码默认带 BITMAP 索引（低基数最佳选）
   - kbId ID 形如 `kb_00`..`kb_09`。如需换成 UUID / 业务 ID 只需改 `data_gen` 配置
5. **召回率验证基线**：S0 的"无标量索引 + 向量 brute force"作为 ground truth 是否够？还是要再加一个 FLAT 索引作为 100% 精确召回基线

---
*起草版本 v0.1。等用户确认方向后再开始写 `src/` 下的代码。*
