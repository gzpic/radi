# Radix Tree Prefix Cache 开发日志

> 项目目标：将 mllm 的 RadixTree 移植到 llama.cpp，为端侧 Agent 推理框架提供 KV cache 可回退/可裁剪/可复用/可分叉能力。

---

## 一、整体架构重点

### 1. 为什么需要 Radix Tree

原版 llama.cpp 已具备的能力：
- 同会话增量 decode（KV cache 天然保留，不需要每轮 prefill）
- `seq_rm` / `seq_cp` 做基本回退和分支
- 固定 system prompt 可通过提前配置缓存

**Radix Tree 真正多出的三个能力：**

| 能力 | 原版 | Radix Tree |
|------|------|------------|
| 动态前缀匹配 | 不支持（需提前知道前缀） | Agent 场景 token 序列不可预知，树自动索引 |
| 多前缀共存 | 只有一条活跃序列 | 任意多历史路径，自动共享公共子序列 |
| 清空后快速恢复 | 必须全量重算 | O(L) 查找 + reclaim，零重算 |

### 2. 六大优化项及依赖关系

```
A3(追加) ──→ A1(跳过prefill) ──→ A18(fork/merge)
                  │                    │
                  └──→ A2(回退) ──────┘
                        │
                        └──→ A4(裁剪) ──→ A6(LRU淘汰)
```

实施顺序：A3 → A1 → A2 → A18 → A4 → A6（跳过了 A5 多级缓存）

---

## 二、各模块重点与难点

### A3 多轮追加模式

**重点：** `clear(data=false)` 保留 prefix tree 和 generation counters，KV buffer 中的数据虽然 cell 元数据被重置，但实际数据仍在，通过 find+reclaim 可复用。

**难点：** generation counter 的一致性 —— clear 时不能重置 generation，否则 tree 中记录的旧 generation 全部失效。

**处理：** clear() 增加 `data` 参数，`data=false` 时只重置 cell 的占用状态和 head 指针，保留 `cell_generations[]` 和 `prefix_tree`。

### A1 跳过 Prefill（核心，难度最高）

**重点：** 这是整个项目的核心价值。没有 A1，prefix cache 只是"能查到"但"省不了计算"的索引。

**难点 1 — ubatch 裁剪：** `llama_ubatch` 是纯指针结构（`token*`, `pos*`, `seq_id**`, `output*`），不拥有数据。裁剪不能 memcpy 新数组，只能做指针偏移：
```cpp
ub.token   += skip;
ub.pos     += skip * ub.n_pos;
ub.seq_id  += skip;
ub.output  += skip;
ub.n_tokens -= skip;
```

**难点 2 — sinfo 同步：** `slot_info` 中的 `idxs[]` 索引也要同步裁剪前 skip 个元素，否则 `apply_ubatch` 写入的 cell 位置错乱。

**难点 3 — attention mask 正确性：** 裁剪后的 ubatch 只包含新 token，但 attention 必须覆盖 reclaimed 的前缀 cell。关键发现：`set_input_kq_mask()` 遍历 `0..n_kv` 的所有 cell 检查 position，reclaimed cell 的 position 已恢复，自然被包含在 mask 中。**不需要额外处理。**

**难点 4 — 数据流传递：** `prepare()` 检测前缀命中并填充 `llama_prefix_match_info`，但裁剪和 reclaim 发生在 `init_batch()` 和 `apply()` 中。需要通过 `pending_prefix_matches` 成员变量在三个阶段间传递状态。

**处理方案：**
- `prepare()`: 检测前缀 → 填充 `pending_prefix_matches[]`
- `init_batch()`: 读取 match info → 裁剪 ubatch/sinfo → 传入 `llama_kv_cache_context`
- `apply()`: 先 reclaim 前缀 cell → 然后只对裁剪后的 ubatch 执行 forward

### A2 Checkpoint 回退

**重点：** O(1) save（只记录 pos 边界），rollback 通过 `seq_rm` 删除边界之后的 cell。

**难点：** 多个 checkpoint 的级联删除 —— rollback 到 cp1 时，cp1 之后创建的 cp2、cp3 都要一起删除。

**处理：** `checkpoints` 用 vector 按创建顺序存储，rollback 时 `erase(it, end())`。

### A18 Fork/Merge

**重点：** `fork()` 基于 `seq_cp` 零拷贝分裂，`merge()` 保留 winner 的 cell 并 `seq_rm` 释放 losers。

**难点 1 — seq_id 上限：** `n_seq_max` 限制了最大序列数，fork 超限时必须优雅失败（返回 -1）。

**难点 2 — merge 的 seq_id 归一化：** winner 不一定是 seq 0，merge 后需要 `seq_cp(winner→0)` + `seq_rm(winner)` 把 winner 的数据转移到主序列。

### A4 Selective Trim

**重点：** 删除中间旧 turn 的 KV cell，然后 `seq_add` 偏移后续 position 填补空隙。

**难点：** position 连续性 —— `seq_rm(0, p0, p1)` 后 position 出现空洞，必须 `seq_add(0, p1, -1, -n_remove)` 把 `[p1, ∞)` 的 position 左移 `n_remove`，否则后续 decode 的 position 不连续会导致 RoPE 计算错误。

### A6 LRU 淘汰

**重点：** 基于 `last_accessed` 时间戳淘汰最久未使用的叶节点，控制 radix tree 内存增长。

**难点 1 — 时间戳更新时机：** 必须在 `search()` 命中时更新 `last_accessed` 和 `hit_count`，而非只在 `insert()` 时。否则频繁查询但未重新插入的热门前缀会被误淘汰。

**难点 2 — 淘汰后树结构维护：** 删除叶节点后，若其父节点只剩一个子节点且不是 root，理论上应合并（路径压缩）。当前选择"正确性优先"，暂不合并。

**处理：** `evict_lru()` 每轮重新 `collect_leaves()` → 找 `min(last_accessed)` → `remove_leaf()`。虽非最优时间复杂度，但逻辑清晰、正确性有保障。

---

## 三、碰到的问题及处理

### 问题 1：Benchmark 设计根本性缺陷

**现象：** 原 benchmark 每轮调用 `llama_memory_clear()` 全清 KV 再全量 decode，A 组（原版）和 C 组（prefix cache）结果几乎相同。

**根因：** 原版 llama.cpp 同会话内天然增量 decode，从不需要每轮 prefill。这个 benchmark 测的不是真实使用方式。

**处理：** 废弃旧 benchmark，重新设计 3 个针对 radix tree 真正优势的场景：
1. **上下文溢出重建** — n_ctx 设小 → 溢出 → clear → 重建（radix tree 可 reclaim 跳过前缀）
2. **Agent 回退重试** — 工具调用失败 → rollback → 重试（测正确性为主）
3. **共享 System Prompt 多请求** — 多请求共享 system prompt → 每次只 decode user 部分

### 问题 2：MSYS2 + MinGW 编译环境冲突

**现象：** Git Bash 中运行 `g++` 编译，退出码 1，无任何错误输出。

**根因：** 路径翻译冲突。Git Bash 的 `/tmp` 映射到 `C:/Users/.../AppData/Local/Temp/`，但 MinGW gcc 期望的临时目录不同。文件写到 Git Bash 的 `/tmp` 后，MinGW gcc 找不到。进一步，`cmake -G "Unix Makefiles"` 生成的 Makefile 中包含 POSIX 路径，MSYS make 传给 MinGW gcc 时无法解析。

**处理：**
1. 安装 `mingw-w64-ucrt-x86_64-cmake` + `mingw-w64-ucrt-x86_64-ninja`
2. 从 MSYS2 ucrt64 shell 运行 `cmake -G Ninja`
3. 所有编译/运行命令通过 `msys2_shell.cmd -ucrt64 -c "..."` 执行
4. 使用 Windows 原生路径（`E:/codex/...`）避免路径翻译

### 问题 3：`llama_prefix_match_info` 声明顺序

**现象：** 编译报 `llama_prefix_match_info` 未声明。

**根因：** 该结构体定义在 `llama-kv-cache.h` 中 `llama_kv_cache_context` 类之后，但被 `llama_kv_cache` 类（在它之前）引用。

**处理：** 将 `llama_prefix_match_info` 结构体定义移到 `llama_kv_cache` 类声明之前。

### 问题 4：编译失败是环境问题还是代码问题

**现象：** 改动后编译不通，怀疑是新代码引入的问题。

**处理：** `git stash` 暂存所有改动 → 编译原始代码 → 同样失败 → 确认是环境问题而非代码问题 → `git stash pop` 恢复 → 切换到正确的 MSYS2 ucrt64 环境解决。

---

## 四、测试覆盖

| 测试阶段 | 文件 | 数量 | 覆盖范围 |
|----------|------|------|----------|
| Phase 1-2 | test-radix-tree.cpp | 32 | VectorView + 树核心算法 |
| Phase 3 | test-radix-tree-integration.cpp | 16 | promote/find/reclaim 集成 |
| Phase 4 | test-radix-tree-features.cpp | 26 | A2/A18/A4/A6 + 组合场景 |
| **合计** | | **74** | **全部通过** |

Phase 4 测试通过 KVCacheHarness 模拟 `llama_kv_cache` 的 seq 操作语义，无需加载模型文件即可验证 checkpoint/fork/merge/trim/LRU 的逻辑正确性。

---

### 问题 5：`seq_add()/seq_div()` 后 prefix tree 不一致

**现象：** `selective_trim` 做 `seq_rm` + `seq_add` 后，tree 中旧的 token→cell 映射仍然存在。下次搜索 trim 后的新 token 序列时找不到匹配。

**根因：** `seq_rm` 已经对被释放的 cell 做了 `invalidate_cell`，但 `seq_add` 改变 position 时没有 invalidate 受影响的 cell。tree 中旧条目的 cell index 虽然没变，但 generation 也没变，导致 `find()` 仍然认为旧条目有效 —— 然而旧条目的 token 序列已经不反映 trim 后的实际状态。

**处理（两层）：**
1. **底层：** 在 `seq_add()` 和 `seq_div()` 中，对受影响的 cell 执行 `cell_generations[strm][i]++` + `prefix_tree->invalidate_cell(i)`，确保 position 变更后旧条目自动失效
2. **上层：** `selective_trim()` 增加可选参数 `remaining_tokens/remaining_cells/n_remaining`，trim 完成后自动调用 `prefix_cache_promote()` 将存活的 token→cell 映射重新插入 tree

**验证：** 5 个专项测试覆盖：
- `seq_add` 后旧条目被 invalidate
- trim 无 re-promote 时后续搜索失败
- trim 有 re-promote 时后续搜索命中
- re-promote 后前缀匹配仍可用
- 连续多次 trim + re-promote

---

## 五、当前状态

- 代码：A3/A1/A2/A18/A4/A6 全部实现 + tree 同步修复，libllama.a 编译通过
- 测试：79/79 通过（32 + 16 + 31）
- 未完成：
  - A5 多级 Prefix Cache（已跳过）
  - 端到端 Benchmark（需模型文件，3 个场景待实测）
  - ASan/UBSan/TSan 内存安全测试
