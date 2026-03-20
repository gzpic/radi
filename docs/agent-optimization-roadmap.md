# 端侧 Agent 推理框架优化方案总表

> 基于 llama.cpp + radix tree prefix cache 改造为面向端侧 Agent 的推理框架
> 目标设备: Jetson Orin Nano (8GB) / 手机 SoC
> 日期: 2026-03-20

---

## 优化方案总表

| 编号 | 优化方向 | 优先级 | 改动量 | 难度 | 预期收益 | 依赖 | 说明 |
|------|---------|--------|--------|------|---------|------|------|
| **A1** | **真正跳过 prefill** | 🔴 P0 | ~200行 | 中高 | Agent 多轮 TTFT 降低 60-80% | 当前 prefix cache | 最关键的一步。当前 reclaim 恢复了 cell 元数据，但推理仍全量 prefill。需改 prepare() 只为未命中 token 创建 ubatch，跳过已命中前缀。没有这个，prefix cache 只是"能查到但省不了计算"的索引。 |
| **A2** | **Checkpoint 回退** | 🔴 P0 | ~150行 | 低 | 回退零延迟，避免重算 | seq_rm | Agent 走错路/tool 报错时回到历史状态。只需记录 turn 边界的 pos，rollback 时 seq_rm 删后续 cell。不需要拷贝 KV data，O(删除cell数)。prefix cache 树条目靠 generation 自动失效。 |
| **A3** | **多轮追加模式** | 🔴 P0 | ~100行 | 低 | 多轮不清空，增量 append | A1 | 当前每轮 decode 前要 seq_rm 清空再重建。改为保留历史 KV，只 append 新 turn 的 token。是 A1 的前提——不清空才有东西可复用。 |
| **A4** | **Selective Trim 裁剪** | 🟡 P1 | ~300行 | 中 | 上下文不超限，保留关键部分 | seq_rm + seq_add | 上下文接近上限时，删除中间旧 turn，保留 System+Tools+最近K轮。需要 pos 重编号(seq_add偏移) + re-promote 到树。Agent 长对话必须有。 |
| **A5** | **多级 Prefix Cache** | 🟡 P1 | ~200行 | 中 | 分层淘汰，System 永不丢 | LRU 实现 | 用 extra_key 区分层级: L0 System(永驻)、L1 Tools(长期)、L2 History(会话级)、L3 Current(不缓存)。OOM 时按层级优先淘汰。 |
| **A6** | **LRU 淘汰** | 🟡 P1 | ~200行 | 中 | 缓存满时自动释放 | 树节点已有字段 | 节点已有 ref_count/hit_count/last_accessed 字段但淘汰逻辑未写。缓存满时根据 A5 的层级 + 访问时间决定淘汰谁。 |
| **A7** | **KV Cache INT8 量化** | 🟡 P1 | ~400行 | 中高 | KV 内存减半 | 无 | apply_ubatch 写入时量化，attention 读取时反量化。Agent 对结构化输出精度敏感，建议关键层保持 FP16。与 prefix cache 兼容(cell_generations 不受影响)。 |
| **A8** | **Structured Output 加速** | 🟢 P2 | ~500行 | 中高 | decode JSON 提速 2-3x | 无 | Agent 输出高度结构化(JSON tool call)。Grammar-guided sampling 约束合法 token；确定性 token 跳过 forward pass；tool call 模板 prefill。 |
| **A9** | **KV Swap-out/in** | 🟢 P2 | ~400行 | 高 | 并发 Agent 数 3-4x | A5, A6 | Agent 等 tool 执行时 LLM 空闲。KV dump 到磁盘释放内存，下次推理时 swap-in。树条目标记 "paged" 不删除。配合 A5 层级决定 swap 优先级。 |
| **A10** | **Speculative Decoding** | 🟢 P2 | ~800行 | 高 | decode 速度 2-4x | 无 | 小 draft model 猜测 + 大 target model 验证。Agent 的 JSON 输出结构化程度高，draft 接受率可达 70%+。draft 和 target 可共享 prefix cache。 |
| **A11** | **跨 seq 共享 KV (CoW)** | 🟢 P2 | ~500行 | 高 | 多 Agent 内存节省 30-50% | cell refcount | 多个 Agent 共享 System+Tools 的 KV cells，Copy-on-Write 修改时才拷贝。需要 cell 级 refcount 大改。 |
| **A12** | **KV 持久化 (冷启动加速)** | 🔵 P3 | ~200行 | 低 | 冷启动 0ms prefill | A1 | System+Tools 的 KV data dump 到磁盘，启动时 mmap 加载 → promote 到树 → 首次请求直接命中。格式: tokens + kv_data + sha256。 |
| **A13** | **多 Agent 交错调度** | 🔵 P3 | ~600行 | 高 | GPU 利用率 30%→80% | A9 | continuous batching 感知 Agent 状态(THINKING/WAITING_TOOL/IDLE)。多 Agent 推理交错填充 GPU 空闲期。 |
| **A14** | **TensorRT 算子融合** | 🔵 P3 | ~1000行 | 高 | prefill 提速 30-50% | 无 | RMSNorm+Linear+SiLU+Mul 融合减少 global memory 读写。需要针对 Orin SM_87 架构调优 tile size。 |
| **A15** | **DLA Offload** | 🔵 P3 | ~800行 | 高 | 有效吞吐翻倍 | A14 | Orin 专有。GPU 跑 Attention(动态shape)，DLA 跑 FFN(固定shape, INT8)，流水线交错。 |
| **A16** | **手机 NPU 后端** | 🔵 P3 | ~1500行 | 很高 | 手机端可部署 | 无 | 高通 QNN / Apple CoreML / Android NNAPI 后端。W4A8 适配。需要 layout 转换(channel-last for ANE)。 |
| **A17** | **Tool 执行预测+预取** | 🔵 P3 | ~200行 | 中 | 端到端延迟减 200-500ms | A8 | LLM 生成 tool call 时解析 partial JSON，预测将调用哪个 tool，提前预热 tool 环境。 |

---

## 按优先级分组

### 🔴 P0 — 必须做（没有这些 prefix cache 等于没用）

| 编号 | 方向 | 一句话 | 改动量 |
|------|------|--------|--------|
| A1 | 真正跳过 prefill | reclaim 后只 prefill 增量，不重算前缀 | ~200行 |
| A2 | Checkpoint 回退 | 记录 turn 边界，rollback 用 seq_rm，零重算 | ~150行 |
| A3 | 多轮追加模式 | 不清空 KV，只 append 新 token | ~100行 |

> 这三个合计 ~450 行，是从"demo"到"可用"的最小改动集。

### 🟡 P1 — 应该做（长对话和内存管理）

| 编号 | 方向 | 一句话 | 改动量 |
|------|------|--------|--------|
| A4 | Selective Trim | 删旧 turn 保留 System+最近几轮，pos 重编号 | ~300行 |
| A5 | 多级 Prefix Cache | extra_key 分层，System 永不淘汰 | ~200行 |
| A6 | LRU 淘汰 | 缓存满时按层级+时间淘汰 | ~200行 |
| A7 | KV INT8 量化 | KV 内存减半，跑更大模型 | ~400行 |

### 🟢 P2 — 值得做（性能翻倍）

| 编号 | 方向 | 一句话 | 改动量 |
|------|------|--------|--------|
| A8 | Structured Output | JSON grammar sampling + 确定性 token 跳过 | ~500行 |
| A9 | KV Swap-out/in | 等 tool 时 KV 换出磁盘，释放给其他 Agent | ~400行 |
| A10 | Speculative Decoding | 小模型猜 + 大模型验，JSON 场景高接受率 | ~800行 |
| A11 | 跨 seq CoW 共享 | 多 Agent 共享 System KV，写时拷贝 | ~500行 |

### 🔵 P3 — 锦上添花（硬件专项 / 大工程）

| 编号 | 方向 | 一句话 | 改动量 |
|------|------|--------|--------|
| A12 | KV 持久化 | 固定前缀 KV dump 磁盘，冷启动 0ms | ~200行 |
| A13 | 多 Agent 调度 | 感知 Agent 状态，交错填充 GPU 空闲 | ~600行 |
| A14 | TensorRT 融合 | 算子融合减少 memory 读写 | ~1000行 |
| A15 | DLA Offload | GPU+DLA 流水线交错 (Orin 专有) | ~800行 |
| A16 | 手机 NPU 后端 | QNN/CoreML/NNAPI 适配 | ~1500行 |
| A17 | Tool 预测预取 | partial JSON 解析预热 tool | ~200行 |

---

## 收益预估

```
                        TTFT (首 token 延迟) 变化

基线 (当前):             Agent 10 轮, 累计 prefill ~15000 tok
                        每轮 TTFT ~300ms, 总延迟 ~3s

+P0 (A1+A2+A3):         累计 prefill ~2000 tok
                        每轮 TTFT ~60ms, 总延迟 ~0.6s    ← 5x 提速

+P1 (A4-A7):            上下文不超限 + 内存减半
                        可跑 7B 模型 (原来只能 3B)

+P2 (A8-A11):           decode 速度 2-4x
                        JSON 生成 ~50ms/call → ~15ms/call
                        并发 Agent 4路

+P3 (A12-A17):          冷启动 0ms + 硬件极限性能
```

---

## 与当前项目的关系

```
已完成 (当前 prefix cache)          最近需要补的
──────────────────────            ──────────────
✅ radix tree 数据结构      →→→    A1: 真正跳过 prefill
✅ promote / find           →→→    A3: 多轮追加模式
✅ generation 验证          →→→    A2: checkpoint 回退
✅ reclaim 恢复元数据       →→→    A4: selective trim
✅ cell_to_nodes 反向索引   →→→    A6: LRU 淘汰
✅ extra_key 隔离           →→→    A5: 多级缓存
✅ 57/57 测试通过           →→→    端到端验证增量 prefill
```

---

*此文档用于讨论端侧 Agent 推理框架的优化路线。*
