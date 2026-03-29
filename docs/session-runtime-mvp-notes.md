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

未完成（明确留到下一阶段）：
- `test-session.cpp` 单元测试
- L2/L3/L4 全链路回归
- BUG-2（multi-stream generation 绑定）

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
