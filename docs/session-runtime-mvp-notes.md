# Session Runtime MVP 笔记（2026-03-29）

## 1. 目标与边界

目标是把已有 KV 能力（checkpoint / rollback / fork / merge / selective_trim / prefix evict）从“底层机制”提升为“上层可调度接口”。

本次只做 MVP：
- 打通 `llama_session_*` C API
- 新增 Session Runtime 调度层（turn 追踪 + 语义翻译）
- 不改底层数学语义，不引入新推理路径

不在本次范围：
- 复杂策略编排（留给 Agent Runtime）
- 多 stream 语义修复（BUG-2）
- 全量测试闭环（单独测试阶段做）

---

## 2. 本次改动清单

### 新增文件

- `src/llama-session.h`
- `src/llama-session.cpp`

### 修改文件

- `include/llama.h`
- `src/llama-context.cpp`
- `src/CMakeLists.txt`

### 已暴露的 Session C API（MVP）

- 生命周期：`llama_session_default_params` / `llama_session_init` / `llama_session_free`
- Turn 追踪：`llama_session_turn_begin` / `llama_session_turn_add_tokens` / `llama_session_turn_end`
- Agent 操作：`llama_session_checkpoint_save` / `llama_session_checkpoint_rollback` / `llama_session_fork` / `llama_session_merge` / `llama_session_trim_turn` / `llama_session_trim_turns`
- 自动维护：`llama_session_after_promote` / `llama_session_check_overflow`
- 查询：`llama_session_kv_usage` / `llama_session_n_turns` / `llama_session_current_pos` / `llama_session_get_turn`

---

## 3. 设计思路（实现原则）

- Session Runtime 只做“状态翻译”和“调度编排”，不替代 KV 内核。
- 所有底层动作走既有 `llama_memory_*` 接口，避免直接耦合私有细节。
- Turn 语义统一映射成 `turn_id -> [p0, p1)`，为 trim / rollback 提供稳定锚点。
- 合并（merge）后清空 turn 元数据，避免上层继续使用过期位置信息。
- 软溢出处理优先 `clear(data=false)`，保留 prefix 复用潜力。

---

## 4. 当前实现状态

已完成：
- Session Runtime MVP 代码接入并编译通过（`--target llama`）
- 远端分支已同步：`codex/prefix-prepare-sideeffect-fix`
- T1 已完成（新增 `tests/test-session.cpp`，模型实跑 4/4 通过）
- T2 已完成（L2 三件套回归通过：32/32、18/18、41/41）
- T3 已完成（L3 一致性 8/8 通过；C8 在 `n_seq_max=1` 下预期 skip）
- T4 已完成（10-run benchmark：Projected savings `69.46 ± 0.57%`）
- BUG-2 阶段 1 已完成：`prefix_cache_promote/find/reclaim` 绑定 `seq_id -> stream` 的 generation 视图

未完成（明确留到下一阶段）：
- T5 / BUG-2 阶段 2：stream-local `invalidate_cell` 语义与多 stream reclaim 的完整闭环
- C8 fork+merge 的多 seq 实跑（需 `n_seq_max > 1` 配置）

---

## 5. 主要风险与假设

- `trim_turn/trim_turns` 当前走最小路径（不在 Session 层重建 remaining token->cell 映射），依赖底层 selective_trim 行为。
- rollback 对 turn 元数据的裁剪基于 `seq_pos_max` 回推，假设 seq0 代表主会话状态。
- merge 后的 turn 元数据策略是“清空重建”，牺牲连续性换一致性。
- 目前没有并发会话下的线程安全验证。

---

## 6. 测试规划（执行版）

### T0. 编译门禁

- 命令：`cmake --build build --target llama -j4`
- 通过标准：`libllama.a` 成功链接

### T1. Session API 单测（新增）

- 新建：`tests/test-session.cpp`
- 覆盖：
  - `turn_begin/add/end` 的边界与顺序
  - `checkpoint_save/rollback` 后 turn 元数据裁剪
  - `trim_turn/trim_turns` 的位置平移
  - `fork/merge` 最小可用路径
  - `after_promote/check_overflow` 触发行为
- 通过标准：全部 PASS，无 crash

### T2. 现有 L2 回归

- 跑：
  - `tests/test-radix-tree.exe`
  - `tests/test-radix-tree-integration.exe`
  - `tests/test-radix-tree-features.exe`
- 通过标准：不新增失败

### T3. L3 推理一致性

- 跑：`tests/test-radix-tree-consistency.exe <model>`
- 重点：开启 `n_seq_max > 1` 后验证 C8 不再 SKIP
- 通过标准：C6/C7/C8 全通过（至少不回归）

### T4. L4 稳定性基准

- 跑：`tests/test-radix-tree-agent-bench.exe <model> --runs 10`
- 通过标准：
  - turn3-5 命中不掉 0
  - `Projected savings` 方差可控（建议 stddev < 2%）

### T5. BUG-2 专项

- 新增多 stream 语义测试（promote/find/reclaim）
- 通过标准：不同 stream 不串命中、不误复用

---

## 7. 推荐执行顺序

1. T0 编译门禁  
2. T1 Session 单测  
3. T2 L2 回归  
4. T3 L3 一致性  
5. T4 L4 稳定性  
6. T5 BUG-2 专项

先做前 4 步即可判断 Session Runtime MVP 是否可合入主干。

---

## 8. T1 实测记录（2026-03-29）

### 文件

- `tests/test-session.cpp`
- `tests/CMakeLists.txt`（新增 `test-session` 构建目标，手动传模型路径运行）

### 编译

```bash
cmake --build build --target test-session -j4
```

### 运行

```bash
build/bin/test-session.exe models/qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf
```

### 结果

- `turn + checkpoint + rollback`：PASS
- `trim_turn + trim_turns`：PASS
- `fork + merge (n_seq_max > 1)`：PASS
- `after_promote + overflow`：PASS
- 汇总：`pass=4 fail=0`

---

## 9. 测试方法与用例设计说明（补充）

### 9.1 测试方法（Method）

本次 Session Runtime 采用“分层 + 不变量”的测试方法：

1. 分层门禁  
先过编译门禁（T0），再测 Session API（T1），最后再做 L2/L3/L4 回归，避免在未可编译状态下投入高成本回归。

2. 不变量驱动  
每个 case 都围绕状态不变量断言，而不是只看“函数返回成功”：
- turn 数量变化是否符合预期
- `turn_id -> [p0, p1)` 是否连续且可解释
- rollback/trim/merge 后旧元数据是否被正确清理
- `seq_pos_max` 是否与当前状态一致

3. 真实模型最小集成  
T1 不是纯 mock，而是加载最小可用模型（Qwen3-0.6B-Q8_0）走真实 `decode` 路径，确保 Session API 与真实 KV 行为对齐。

4. 控制变量  
统一 CPU 跑法、固定 `n_ctx/n_batch`，并在 `fork+merge` 场景显式设置 `n_seq_max > 1`，确保测试目标可触发、结果可复现。

5. 先验证行为，再扩展性能  
T1 只验证语义正确性；性能与波动控制放在 T4（`--runs` 多次统计）执行。

### 9.2 用例设计（Case Design）

`tests/test-session.cpp` 当前 4 个用例分别覆盖不同风险面：

1. `turn + checkpoint + rollback`  
目的：验证最基础会话流。  
关键断言：
- rollback 前后 `n_turns` 从 2 回到 1
- 保留 turn 的 `turn_id` 正确
- `current_pos == p1 - 1`

2. `trim_turn + trim_turns`  
目的：验证按语义删 turn 后的位置平移与元数据裁剪。  
关键断言：
- `trim_turn` 后 turn 数量减少
- 存活 turn 的位置仍有序（前一段结束不晚于后一段开始）
- `trim_turns` 后可清空目标范围

3. `fork + merge (n_seq_max > 1)`  
目的：验证多序列路径可达，且 merge 后主序列状态可继续使用。  
关键断言：
- `fork` 返回有效 seq_id
- 分支 decode 成功
- `merge` 成功且主序列位置推进到预期范围
- 当前 MVP 语义下 turn 元数据在 merge 后清空

4. `after_promote + overflow`  
目的：验证自动维护钩子。  
关键断言：
- `after_promote` 触发后节点数不增加，且在阈值内
- `check_overflow` 触发 soft clear 后，turn 元数据清空，`current_pos == -1`

### 9.3 为什么这样设计

- 这 4 个 case 对应 Session Runtime 的 4 条关键能力链路：状态追踪、结构变更、分支语义、自动维护。  
- 每个 case 都覆盖“动作 + 状态后验”，能尽早发现“接口成功但状态错”的隐蔽问题。  
- 先做最小闭环，再向 L2/L3/L4 扩展，可在成本可控下快速建立可信基线。

---

## 10. T2/T3/T4 实测记录（2026-03-29）

### 10.1 运行环境约束

- Windows + MSYS2 UCRT 运行时；执行前需把 `D:\SoftwareFilePlace\MSYS2\ucrt64\bin` 加入 `PATH`。
- 模型：`models/qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf`

### 10.2 T2（L2 回归三件套）

执行：

```bash
tests/test-radix-tree.exe
tests/test-radix-tree-integration.exe
tests/test-radix-tree-features.exe
```

结果：

- `test-radix-tree.exe`：`32 passed, 0 failed`
- `test-radix-tree-integration.exe`：`18 passed, 0 failed`
- `test-radix-tree-features.exe`：`41 passed, 0 failed`

结论：L2 层无回归。

### 10.3 T3（L3 推理一致性）

执行：

```bash
tests/test-radix-tree-consistency.exe models/qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf
```

结果：

- 总体：`8 passed, 0 failed`
- C1~C6：通过（C1/C2/C3/C4/C6 为 bit-exact）
- C7：通过（`max_abs_diff=3.232903`，`cosine=0.9791025808`，argmax 一致）
- C8：`n_seq_max=1` 下 fork 不可用，按预期 skip 并计 PASS

结论：Session Runtime 接入未破坏现有一致性基线。

### 10.4 T4（L4 稳定性 benchmark）

执行：

```bash
tests/test-radix-tree-agent-bench.exe models/qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf --runs 10
```

关键输出（工具内置每个 run 前 `clear(true)`）：

- Baseline total TTFT：`5874.2 ± 94.7 ms`
- Projected total (A1)：`1793.7 ± 45.1 ms`
- Projected savings：`69.46 ± 0.57 %`
- Cache overhead：`0.070 ± 0.003 ms/turn`

结论：在 10-run 稳定性口径下，TTFT 理论节省约 69.5%，波动可控（stddev < 2%）。

---

## 11. BUG-2 阶段 1（stream 绑定）记录（2026-03-29）

### 11.1 代码改动

- `src/llama-kv-cache.h`
  - `prefix_cache_promote/find` 新增 `seq_id` 参数（默认 0），保持旧调用兼容。
- `src/llama-kv-cache.cpp`
  - `promote/find` generation 访问从固定 `cell_generations[0]` 改为 `cell_generations[seq_to_stream[seq_id]]`。
  - `prefix_cache_reclaim` 内部 `find` 改为传入同一 `seq_id`。
  - `prepare()/apply()/trim re-promote` 调用点补齐 `seq_id` 透传。

### 11.2 回归测试

- 文件：`tests/test-radix-tree-integration.cpp`
- 新增用例：
  - `harness_multistream_find_uses_seq_stream_generation`
  - `harness_multistream_seq_to_stream_mapping_applies`
- 当前汇总：`18 passed, 0 failed`

### 11.3 边界说明

- 本次只修复 generation 绑定错误（BUG-2 阶段 1）。
- `invalidate_cell(idx)` 仍是“仅按 cell idx”失效，不区分 stream；该问题留在 BUG-2 阶段 2 处理。

---

## 12. 遗留事项暂存（下轮优先级清单）

> 状态说明：以下事项已确认“暂缓当前执行”，后续恢复时按优先级顺序推进。

### P0（必须先做）

1. BUG-2 阶段 2：实现 stream-local `invalidate` 语义  
目标：避免不同 stream 在同 cell_idx 上互相误失效。

2. BUG-2 阶段 2 回归测试补齐（L2）  
范围：`promote/find/reclaim/overwrite/seq_rm` 的跨 stream 交叉场景。  
通过标准：不串命中、不误复用、不误失效。

### P1（完成 P0 后立即做）

3. 一轮完整门禁回归  
范围：`test-session` + `test-radix-tree` + `test-radix-tree-integration` + `test-radix-tree-features` + `test-radix-tree-consistency` + `test-radix-tree-agent-bench --runs 10`。  
产物：固定格式测试记录（命令、通过率、关键指标、与上一版差异）。

4. C8 fork+merge 一致性实跑（取消 skip）  
前置：`n_seq_max > 1` 的配置与用例路径打通。  
通过标准：C8 不再 SKIP，且不回归 C1~C7。

### P2（收尾质量项）

5. 并发与长稳健性补测  
范围：并发会话、多次 clear/reclaim 循环、长时间运行下的一致性与稳定性漂移。

6. 文档收口  
将 BUG-2 阶段 2 的设计、实现细节、测试结果回填到本文件与 `docs/dev-log.md`，形成可复现闭环记录。

---

## 13. Agent Runtime 场景对比测试（2026-03-29）

### 13.1 新增测试与脚本

- 测试：`tests/test-agent-runtime-planner-bench.cpp`
  - 场景：多路径路线规划（路径尝试、工具失败、回退、分支探索、删除无效路径、合并最优路径、trim 临时信息）。
  - 对比口径：
    - Baseline：每次尝试 `clear(true)` 后全量累计 prompt prefill
    - Enhanced：`checkpoint/rollback + fork/merge + seq_rm(prune) + trim`
  - 观测产物：
    - `run-XX-events.jsonl`（逐步状态回放）
    - `planner-comparison-summary.json`（结构化指标）
    - `planner-comparison-summary.md`（可读汇总）

- 运行脚本：`scripts/run-agent-runtime-comparison.ps1`
  - 一键构建 `test-agent-runtime-planner-bench`
  - 支持参数：`-Model`、`-Runs`、`-OutDir`

### 13.2 中间状态断言（Enhanced）

- base turn 后 `n_turns == 1`，`current_pos == seq0_pos_max`
- path-A 失败后 rollback 回到 root checkpoint（位置和 turn 元数据恢复）
- fork 后分支起点位置等于 root
- prune 后无效分支 `seq_pos_max == -1`
- merge 后 winner 状态进入 seq0，分支 id 被归一
- trim 后临时 turn 被删除，位置不漂移

### 13.3 Smoke 实测（Qwen3-0.6B-Q8_0，runs=1）

- Baseline：`4468.5 ms`, `1216 tokens`
- Enhanced：`1819.5 ms`, `409 tokens`
- Token reduction：`66.37%`
- Time reduction：`59.28%`
- 状态断言：`20 pass / 0 fail`
