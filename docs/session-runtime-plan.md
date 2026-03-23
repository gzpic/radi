# Session Runtime 实施策略

> 目标：在 llama.cpp KV cache 层之上构建 Session Runtime 调度层，让 Agent Runtime 能通过语义化接口（turn ID、checkpoint ID）操控 KV 状态，而不需要了解 position/cell/generation 等底层细节。

---

## 一、背景：为什么需要 Session Runtime

KV cache 层已实现 6 大机制，但它们是"刀片"——没有人在正确时机拔出来用。

| 机制 | 状态 | 问题 |
|------|------|------|
| checkpoint_save / rollback | 已实现，C API 已暴露 | 无人调用 |
| fork / merge | 已实现，C API 已暴露 | 无人调用 |
| selective_trim | 已实现，C API 已暴露 | 无人调用，且需要 raw position 参数 |
| evict_lru | 已实现，**未暴露 C API** | 无人调用（BUG-5） |
| prefix cache (find/promote/reclaim) | 已实现，自动生效 | 正常工作 |
| clear(data=false) | 已实现 | 无人主动使用 soft clear |

---

## 二、三层架构定位

```
Agent Runtime（语义层）
│  "工具调用失败了，回退"
│  "第 2 轮对话不重要，删掉"
│  "并行跑两个方案"
│
├── 调用 ↓
│
Session Runtime（状态层）  ← 本次构建
│  Turn 追踪：turn_id → [p0, p1) position 范围
│  操作翻译：trim_turn(2) → selective_trim(120, 200, ...)
│  自动行为：LRU 淘汰、溢出 soft clear
│  状态查询：KV 使用率、turn 列表
│
├── 调用 ↓
│
KV Cache（机制层）
   checkpoint / rollback / fork / merge / trim / evict / prefix cache
```

**关键约束：**
- Session Runtime 不继承 `llama_memory_i`，它是调度层不是存储层
- 所有 KV 操作通过 `llama_memory_*` C API，不直接访问 `llama_kv_cache` 内部
- Session Runtime 是可选的，不使用它时现有代码路径完全不受影响
- Turn 追踪是纯内存元数据，KV cache 不知道 turn 概念

---

## 三、前置 Bug 修复

在构建 Session Runtime 之前，先修复 4 个已知 bug，确保底层机制可靠。

| Bug | 风险 | 文件 | 修复内容 | 状态 |
|-----|------|------|----------|------|
| BUG-1 | 高 | `src/llama-kv-cache.cpp` prepare() | prefix cache 路径加 `ubatch.token != nullptr` 守卫，防止 embedding 输入空指针 | ✅ 已修 |
| BUG-3 | 中高 | `src/llama-kv-cache.cpp` selective_trim() + merge() | trim 后删除 `pos_end >= p0` 的 checkpoint；merge 后清空所有 checkpoint | ✅ 已修 |
| BUG-4 | 中 | `src/llama-kv-cache.cpp` merge() | merge 完成后 `next_fork_seq_id = 1`，回收 seq_id 池 | ✅ 已修 |
| BUG-5 | 残余 | 多文件（见下表） | 暴露 `evict_lru` 和 `node_count` 的 C API | ✅ 已修 |

### BUG-5 改动文件明细

| 文件 | 改动 |
|------|------|
| `src/llama-kv-cache.h` | 新增 `prefix_evict_lru(max_nodes)` 和 `prefix_node_count()` 方法声明 |
| `src/llama-kv-cache.cpp` | 实现：委托到 `prefix_tree->evict_lru()` 和 `prefix_tree->node_count()` |
| `include/llama.h` | 新增 `llama_memory_prefix_evict_lru()` 和 `llama_memory_prefix_node_count()` C API |
| `src/llama-context.cpp` | C wrapper：dynamic_cast + 转发 |

---

## 四、Session Runtime 核心设计

### 4.1 数据结构

```cpp
struct llama_session_turn {
    int32_t     turn_id;        // 单调递增 ID
    llama_pos   p0;             // 起始 position（含）
    llama_pos   p1;             // 结束 position（不含）
    std::vector<llama_token> tokens;  // 该 turn 的 token 序列
};

struct llama_session_params {
    float    evict_threshold;   // tree 节点数超过 n_ctx * ratio 时触发淘汰（默认 0.8）
    float    pressure_warn;     // KV 使用率告警阈值（默认 0.9）
    uint32_t n_ctx;             // context 大小
};

class llama_session {
    llama_memory_t mem_;                        // 非拥有指针
    llama_session_params params_;
    std::vector<llama_session_turn> turns_;     // turn 元数据
    int32_t next_turn_id_ = 0;
};
```

### 4.2 能力矩阵

| 类别 | 方法 | 输入 | 输出 | 底层调用 |
|------|------|------|------|----------|
| **Turn 追踪** | `turn_begin()` | — | turn_id | 记录 `p0 = seq_pos_max(0) + 1` |
| | `turn_add_tokens(id, tokens, n)` | turn_id + tokens | — | 扩展 turn 的 token/p1 |
| | `turn_end(id)` | turn_id | — | 固化边界 |
| **Agent 触发** | `checkpoint_save()` | — | cp_id | `llama_memory_checkpoint_save` |
| | `checkpoint_rollback(cp_id)` | cp_id | bool | `llama_memory_checkpoint_rollback` + 删除回退区间 turn |
| | `fork(parent)` | parent_seq | new_seq | `llama_memory_fork` |
| | `merge(winner)` | winner_seq | bool | `llama_memory_merge`（底层已含 BUG-4 修复） |
| | `trim_turn(turn_id)` | turn_id | n_removed | 计算 p0/p1 → `llama_memory_selective_trim` |
| | `trim_turns(first, last)` | turn_id 范围 | n_removed | 合并范围 → 单次 selective_trim |
| **自动行为** | `after_promote()` | — | — | 检查 node_count > 阈值 → `evict_lru` |
| | `check_overflow()` | — | — | pos_max >= n_ctx → `clear(data=false)` |
| **查询** | `kv_usage_ratio()` | — | float | `seq_pos_max(0) / n_ctx` |
| | `get_turns()` | — | turn 列表 | 返回内部 vector |
| | `current_pos()` | — | llama_pos | 最后 turn 的 p1 |

### 4.3 trim_turn 执行流程

```
输入: trim_turn(turn_id=2)

1. 查找: turns_[idx] → {turn_id=2, p0=120, p1=200, tokens=[...]}
2. 收集存活 turn 的 tokens + cell indices → remaining 数组
3. 调用: llama_memory_selective_trim(mem_, 120, 200, remaining_tokens, remaining_cells, n_remaining)
4. 平移: 后续 turn 的 p0/p1 全部减去 (200-120)=80
5. 删除: 从 turns_ 中移除 idx
6. 安全网: 作废 pos_end >= 120 的 checkpoint（BUG-3 底层已处理，此处双保险）

结果:
  turn 0: [0, 30)   system     → 不变
  turn 1: [30, 120)  turn 1    → 不变
  turn 2: [120, 200) turn 2    → 已删除
  turn 3: [200, 280) turn 3    → 平移为 [120, 200)
  turn 4: [280, 360) turn 4    → 平移为 [200, 280)
```

### 4.4 C API 设计

```c
// --- 生命周期 ---
LLAMA_API llama_session_t llama_session_init(struct llama_context * ctx);
LLAMA_API void            llama_session_free(llama_session_t session);

// --- Turn 追踪 ---
LLAMA_API int32_t llama_session_turn_begin    (llama_session_t s);
LLAMA_API void    llama_session_turn_add_tokens(llama_session_t s, int32_t turn_id,
                      const llama_token * tokens, int32_t n_tokens);
LLAMA_API void    llama_session_turn_end      (llama_session_t s, int32_t turn_id);

// --- Agent 操作 ---
LLAMA_API int32_t      llama_session_checkpoint_save    (llama_session_t s);
LLAMA_API bool         llama_session_checkpoint_rollback(llama_session_t s, int32_t cp_id);
LLAMA_API llama_seq_id llama_session_fork               (llama_session_t s, llama_seq_id parent);
LLAMA_API bool         llama_session_merge              (llama_session_t s, llama_seq_id winner);
LLAMA_API int32_t      llama_session_trim_turn          (llama_session_t s, int32_t turn_id);
LLAMA_API int32_t      llama_session_trim_turns         (llama_session_t s, int32_t first, int32_t last);

// --- 自动行为 ---
LLAMA_API void  llama_session_after_promote  (llama_session_t s);
LLAMA_API void  llama_session_check_overflow (llama_session_t s);

// --- 查询 ---
LLAMA_API float llama_session_kv_usage(llama_session_t s);
LLAMA_API int32_t llama_session_n_turns(llama_session_t s);
```

---

## 五、文件变更清单

### Phase 1: Bug 修复

| 文件 | 操作 | 改动 |
|------|------|------|
| `src/llama-kv-cache.cpp` | 修改 | BUG-1 null 守卫 + BUG-3 checkpoint 作废 + BUG-4 seq_id 回收 + BUG-5 evict/count 实现 |
| `src/llama-kv-cache.h` | 修改 | BUG-5 新增 prefix_evict_lru / prefix_node_count 声明 |
| `include/llama.h` | 修改 | BUG-5 新增 2 个 C API |
| `src/llama-context.cpp` | 修改 | BUG-5 新增 2 个 C wrapper |

### Phase 2: Session Runtime

| 文件 | 操作 | 内容 |
|------|------|------|
| `src/llama-session.h` | **新建** | llama_session 类 + turn/params 结构体 |
| `src/llama-session.cpp` | **新建** | 全部方法实现 |
| `include/llama.h` | 修改 | 新增 `llama_session_*` C API 区块 |
| `src/llama-context.cpp` | 修改 | 新增 C wrapper 实现 |
| `src/CMakeLists.txt` | 修改 | 加入 llama-session.cpp / .h |

### Phase 3: 测试

| 文件 | 操作 | 内容 |
|------|------|------|
| `tests/test-session.cpp` | **新建** | Session Runtime 单元测试 |
| `tests/test-radix-tree-features.cpp` | 修改 | BUG-3/4 回归测试 |

---

## 六、实施顺序与依赖

```
Phase 1: Bug 修复（前置条件）
  ├── 1.1 BUG-1 null 守卫           [独立]     ✅
  ├── 1.2 BUG-3 checkpoint 作废     [独立]     ✅
  ├── 1.3 BUG-4 seq_id 回收         [独立]     ✅
  ├── 1.4 BUG-5 暴露 evict_lru      [Phase 2 依赖] ✅
  └── 编译 + 跑 89 测试确认无回归    [阻塞 commit]
      └── commit: "Fix BUG-1/3/4/5: null guard, checkpoint invalidation, seq_id recycle, evict_lru API"

Phase 2: Session Runtime 核心
  ├── 2.1 llama-session.h            [头文件先行]
  ├── 2.2 llama-session.cpp          [依赖 2.1]
  ├── 2.3 llama.h C API              [依赖 2.1]
  ├── 2.4 llama-context.cpp wrappers [依赖 2.2 + 2.3]
  ├── 2.5 CMakeLists.txt             [依赖 2.1 + 2.2]
  └── cmake --build 编译通过         [阻塞 commit]
      └── commit: "Add Session Runtime layer with turn tracking, trim-by-turn, auto-evict"

Phase 3: 测试验证
  ├── 3.1 test-session.cpp           [依赖 Phase 2]
  ├── 3.2 bug 回归测试更新           [依赖 Phase 1]
  └── 全部通过                       [阻塞 commit + push]
      └── commit + push: "Add Session Runtime tests + bug fix regression tests"
```

---

## 七、验证方案

| 阶段 | 验证方式 | 通过标准 |
|------|----------|----------|
| Phase 1 后 | 跑 test-radix-tree / integration / features | 89/89 通过 |
| Phase 2 后 | cmake --build build | 编译 0 error |
| Phase 3 后 | 跑 test-session + 全部已有测试 | 全部通过 |
| 可选 | 跑 test-radix-tree-consistency.exe (L3) | 推理一致性不受影响 |

---

## 八、Agent 使用示例（最终效果）

```cpp
// 初始化
llama_session_t session = llama_session_init(ctx);

// Turn 1: System prompt
int32_t t0 = llama_session_turn_begin(session);
llama_session_turn_add_tokens(session, t0, sys_tokens, n_sys);
llama_decode(ctx, batch_sys);
llama_session_turn_end(session, t0);

// Turn 2: User query
int32_t t1 = llama_session_turn_begin(session);
llama_session_turn_add_tokens(session, t1, user_tokens, n_user);
llama_decode(ctx, batch_user);
llama_session_turn_end(session, t1);

// Checkpoint before tool call
int32_t cp = llama_session_checkpoint_save(session);

// Turn 3: Tool call + result
int32_t t2 = llama_session_turn_begin(session);
// ... decode tool response ...
llama_session_turn_end(session, t2);

// Tool failed? Rollback
llama_session_checkpoint_rollback(session, cp);
// turns after checkpoint are gone, retry with different tool

// Context pressure? Trim old turn
llama_session_trim_turn(session, t1);  // "删第 2 轮" — 不需要知道 position

// Cleanup
llama_session_free(session);
```
