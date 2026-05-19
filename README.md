# milvus-quickstart

Milvus 2.5.10 测试代码。详细规划见 [docs/test-plan.md](docs/test-plan.md)。

包含两个独立测试：

| 测试 | 目标 | 入口 |
|---|---|---|
| **测试一** | Standalone 实例上能创建多少个 collection | [src/test1_collection_limit/run_create_only.py](src/test1_collection_limit/run_create_only.py) |
| **测试二** | 10M 行 × 3072 dim 标量筛选 + 多路召回性能 | [src/test2_scalar_filter/](src/test2_scalar_filter/) |

## 目录结构

```
docker/         docker-compose.yml（pin v2.5.10）+ volumes/
docs/           测试规划与结果
src/
  common/       配置 / 数据生成 / schema / 指标
  test1_collection_limit/   测试一：Collection 数量上限
  test2_scalar_filter/      测试二：标量筛选性能（prepare_dataset + run_queries）
results/        CSV / manifest（自动生成，已 gitignore）
.env            本地配置（已 gitignore）
.env.example    配置模板（154 行注释，保持与 .env 同步）
```

---

## 0. 一次性环境准备

### 0.1 Python venv 与依赖

```bash
cd /home/shuaihua/milvus-quickstart
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 0.2 配置 `.env`

```bash
cp .env.example .env       # 第一次
$EDITOR .env               # 改 Milvus URI / NUM_ROWS / SCALAR_PLAN / Azure 凭据 等
```

`.env` 会在脚本启动时被 `python-dotenv` 自动加载（详见 [src/common/config.py](src/common/config.py)），所以**不需要 `source .env`**。优先级：

> **CLI 参数 > 环境变量 (.env) > 代码默认值**

完整可配项见 [.env.example](.env.example)。

### 0.3 启动 Milvus 2.5.10 Standalone

```bash
cd docker && docker compose up -d && cd ..

# 等待 ~15s，然后验证
curl -fsS http://localhost:9091/healthz && echo " milvus ok"
```

3 个容器会起来：`milvus-standalone` `milvus-etcd` `milvus-minio`，数据卷落到 `docker/volumes/` 下（已 gitignore）。

### 0.4 端口冲突排查（机器上已有别的 Milvus / MinIO）

本仓库的 compose 监听 `19530 / 9091 / 9000 / 9001`。如果这些端口已经被占用，需要先 stop 占用者。**只 stop 不 rm，数据卷保留**：

```bash
# 1) 查端口被哪个进程 / 容器占用
sudo ss -lntp | grep -E ':(19530|9091|9000|9001)\s'

# 2) 如果是 docker 容器占用，按名字 stop 掉（替换成你实际看到的容器名）
#    可以先 docker ps 看一眼
docker ps --format 'table {{.Names}}\t{{.Ports}}' | grep -E '19530|9091|9000|9001'
docker stop <container-name-1> <container-name-2> ...

# 3) 再启本仓库的 milvus
cd docker && docker compose up -d && cd ..
```

> 把你 stop 掉的容器名记下来，测试结束后照原样 `docker start ...` 即可恢复。具体见 [§4.3](#43-恢复之前-stop-掉的容器可选)。

---

## 1. 测试二：标量筛选性能（推荐先跑通这条）

### 1.1 Smoke test（5 万行，几分钟跑完，先验证链路）

`.env` 默认就是 smoke 规模（`NUM_ROWS=50000`、`DIM=3072`、`SCALAR_PLAN=S3`），所以不需要改环境：

```bash
source .venv/bin/activate

# Phase A: 建 collection + 灌数据 + 建索引 + load
PYTHONPATH=. python -m src.test2_scalar_filter.prepare_dataset \
    --collection scalar_filter_smoke \
    --num-rows 50000 \
    --scalar-plan S3 \
    --drop-existing

# Phase B: 跑查询
#   Q1=单字段 eq  Q2=多字段 AND  Q3=documentId IN  Q5=纯标量 query  Q6=hybrid 多路召回
PYTHONPATH=. python -m src.test2_scalar_filter.run_queries \
    --collection scalar_filter_smoke \
    --queries Q1 Q2 Q3 Q5 Q6 \
    --warmup 50 --serial 200 \
    --concurrency 1 4 16
```

产物：

- `results/scalar_filter_smoke.manifest.json` — collection 元数据（行数、维度、scalar plan、稀疏配置）
- `results/test2_serial_latency.csv` — 单线程 p50/p95/p99
- `results/test2_concurrent_qps.csv` — 并发 QPS（如未加 `--no-concurrent`）

> 默认开启稀疏向量（`ENABLE_SPARSE=true`），所以 collection 多了一个 `sparse_embedding` 字段，Q6 走 `client.hybrid_search`（dense + sparse），`RRFRanker(k=60)` 融合。不需要稀疏时加 `--no-sparse` 关闭。

### 1.2 生产规模（10M × 3072 dim）

> ⚠️ **资源估算**：`3072 dim × 10M × 4 bytes ≈ 122 GB` 仅 dense 向量原始数据。常驻内存即使开 mmap 也是几十 GB 级。请确认机器内存与磁盘空间。

修改 `.env`：

```bash
NUM_ROWS=10000000
INSERT_BATCH=50000      # 大批次提速
FLUSH_EVERY=10
MMAP_ENABLED=true       # 备忘：实际生效要改 milvus.yaml
```

然后跑：

```bash
PYTHONPATH=. python -m src.test2_scalar_filter.prepare_dataset \
    --collection scalar_filter_10m \
    --scalar-plan S3 \
    --drop-existing

PYTHONPATH=. python -m src.test2_scalar_filter.run_queries \
    --collection scalar_filter_10m \
    --queries Q1 Q2 Q3 Q5 Q6 \
    --warmup 100 --serial 1000 \
    --ranker RRF --rrf-k 60 \
    --concurrency 1 4 16 32 --conc-dur 60
```

### 1.3 切换标量索引方案做对照（S0 ↔ S3 ↔ S4）

`SCALAR_PLAN` 可选 `S0/S1/S2/S3/S4`，详见 [docs/test-plan.md §2.3](docs/test-plan.md)：

| 方案 | 含义 |
|---|---|
| `S0` | 都不建（baseline） |
| `S1` | 只 `documentId(INVERTED)` |
| `S2` | 只低基数字段 `BITMAP` |
| `S3` | `documentId(INVERTED)` + 低基数字段 `BITMAP`（推荐生产） |
| `S4` | 全 `INVERTED`（对比 BITMAP vs INVERTED） |

每个方案用**不同的 collection 名**，重新 `prepare_dataset --drop-existing`，再 `run_queries`，所有结果会**追加到同一个 CSV**，便于对比：

```bash
for plan in S0 S2 S3 S4; do
  PYTHONPATH=. python -m src.test2_scalar_filter.prepare_dataset \
      --collection scalar_filter_smoke_${plan} \
      --num-rows 50000 \
      --scalar-plan ${plan} \
      --drop-existing
  PYTHONPATH=. python -m src.test2_scalar_filter.run_queries \
      --collection scalar_filter_smoke_${plan} \
      --queries Q1 Q2 Q3 Q5 Q6 \
      --warmup 50 --serial 200 --no-concurrent
done
```

---

## 2. 测试一：Collection 数量上限

阶段 A（纯创建，监控 etcd / milvus 元数据增长直到失败）：

```bash
source .venv/bin/activate

PYTHONPATH=. python -m src.test1_collection_limit.run_create_only \
    --target 5000 \
    --sample-every 50 \
    --dim 8
```

- 每 50 个 collection 采样一次（写 `results/test1_collection_limit.csv`）。
- 中途 Ctrl+C 会**安全退出**并保留已写入的 CSV。
- 加 `--cleanup` 在结束时自动 drop 所有创建出来的 collection。

> ⚠️ 这个测试会向 etcd 写入大量元数据，跑完后建议**清理掉测试 collection** 再去跑测试二，避免互相干扰：
> ```bash
> # 看名字符合测试一前缀的
> PYTHONPATH=. python -c "from pymilvus import MilvusClient; c=MilvusClient(uri='http://localhost:19530'); [c.drop_collection(x) for x in c.list_collections() if x.startswith('coll_limit_')]"
> ```

---

## 3. 结果文件位置

| 文件 | 来源 |
|---|---|
| `results/{collection}.manifest.json` | `prepare_dataset` 写入，记录 num_rows / dim / scalar_plan / 稀疏配置 |
| `results/test2_serial_latency.csv` | `run_queries` 单线程 p50/p95/p99 / mean |
| `results/test2_concurrent_qps.csv` | `run_queries` 并发吞吐 |
| `results/test1_collection_limit.csv` | `run_create_only` 每个采样点的耗时与累计数量 |

`results/` 目录已 gitignore，仅保留 `.gitkeep`。

---

## 4. 测试结束的清理与恢复

### 4.1 停掉本仓库的 Milvus（保留数据）

```bash
cd docker && docker compose stop && cd ..
```

### 4.2 彻底清理（包括数据卷）

```bash
cd docker && docker compose down -v && rm -rf volumes && cd ..
```

### 4.3 恢复之前 stop 掉的容器（可选）

如果在 [§0.4](#04-端口冲突排查机器上已有别的-milvus--minio) 为了腾端口而 stop 了别的容器，先把本仓库的 Milvus 关掉，再把它们启回来：

```bash
cd docker && docker compose down && cd ..

# 用 §0.4 记下来的容器名（注意：milvus 这种有依赖关系的栈，要按 minio → etcd → milvus 顺序启）
docker start <minio-container>
docker start <etcd-container>
docker start <milvus-container>
```

---

## 5. 配置覆盖方式速查

所有参数集中在 [src/common/config.py](src/common/config.py) 的 `TestConfig` dataclass：

- `.env` 文件 — 推荐，模板见 [.env.example](.env.example)，启动时自动加载
- 环境变量内联 — 如 `NUM_ROWS=1000000 DIM=768 python -m ...`
- 各脚本的 CLI 参数 — 见各脚本 `--help`

**CLI 参数 > 环境变量 > 默认值。**

```bash
# 查看每个脚本支持的所有参数
PYTHONPATH=. python -m src.test2_scalar_filter.prepare_dataset --help
PYTHONPATH=. python -m src.test2_scalar_filter.run_queries     --help
PYTHONPATH=. python -m src.test1_collection_limit.run_create_only --help
```
