# Radix Tree Prefix Cache — 测试报告

> 日期：2026-03-19
> 环境：Windows 11, MSYS2 ucrt64, GCC 15.2.0, C++17
> 模型：Qwen3 0.6B Instruct (Q8_0, 604MB)
> 测试文件：`tests/test-radix-tree.cpp`, `tests/test-radix-tree-integration.cpp`, `tests/test-radix-tree-e2e.cpp`

---

## 测试架构

```
阶段 1: VectorView 单元测试       9 个用例   纯数据结构        ✅ 全部通过
阶段 2: Radix Tree 核心算法测试   23 个用例   纯算法            ✅ 全部通过
阶段 3: Prefix Cache 集成测试     16 个用例   Harness 模拟      ✅ 全部通过
阶段 4: 端到端流水线测试           4 个步骤   真实模型推理      ✅ 全部通过
                                  ─────────
                                  共 52 个测试点通过
```

测试框架：自定义 assert 宏 + 静态注册，与 llama.cpp 现有测试风格一致。

编译命令：
```bash
# Phase 1+2: 纯数据结构 + 算法
g++ -std=c++17 -I./include -I./ggml/include -I./src \
    -o tests/test-radix-tree.exe \
    tests/test-radix-tree.cpp src/llama-radix-tree.cpp

# Phase 3: 集成测试 (Harness 模拟 KV cache)
g++ -std=c++17 -I./include -I./ggml/include -I./src \
    -o tests/test-radix-tree-integration.exe \
    tests/test-radix-tree-integration.cpp src/llama-radix-tree.cpp

# Phase 4: 端到端 (链接完整 llama 库 + 真实模型)
g++ -std=c++17 -O2 -I./include -I./ggml/include -I./src \
    -o tests/test-radix-tree-e2e.exe tests/test-radix-tree-e2e.cpp \
    -L./build/src -lllama \
    -L./build/ggml/src -Wl,--whole-archive ./build/ggml/src/ggml.a \
    -Wl,--no-whole-archive ./build/ggml/src/ggml-base.a \
    ./build/ggml/src/ggml-cpu.a -lgomp -lpthread -lm
# 运行: ./tests/test-radix-tree-e2e.exe ./models/qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf
```

---

## 阶段 1: VectorView 单元测试 (9/9 PASS)

| # | 用例名 | 验证点 | 结果 |
|---|--------|--------|------|
| 1 | vectorview_construct_from_vector | 从 vector 构造，size/data/operator[] 正确 | PASS |
| 2 | vectorview_subview | subview 后 offset/length 正确 | PASS |
| 3 | vectorview_subview_shares_storage | subview 与原 view 共享底层存储，修改可见 | PASS |
| 4 | vectorview_empty_default | 默认构造 data()=nullptr, size()=0, 不崩溃 | PASS |
| 5 | vectorview_subview_out_of_range | 越界 subview 抛出 out_of_range | PASS |
| 6 | vectorview_equality | operator== 值语义比较，不同存储相同内容返回 true | PASS |
| 7 | vectorview_iterator | begin/end 遍历和 range-for 结果正确 | PASS |
| 8 | vectorview_empty_iterator | 空 view 迭代 0 次，不崩溃 | PASS |
| 9 | vectorview_nested_subview | 嵌套 subview(subview 的 subview) 偏移计算正确 | PASS |

---

## 阶段 2: Radix Tree 核心算法测试 (23/23 PASS)

### 插入 (insert)

| # | 用例名 | 验证点 | 结果 |
|---|--------|--------|------|
| 10 | tree_insert_single | 插入单条序列后 search 找到，cell_indices 正确 | PASS |
| 11 | tree_insert_common_prefix | 插入 [1,2,3] 和 [1,2,4]，自动 split，两条都能找到 | PASS |
| 12 | tree_insert_duplicate | 重复插入同一序列，value 被更新（re-promote 场景） | PASS |
| 13 | tree_insert_one_is_prefix_of_other | [1,2] 和 [1,2,3,4]，短的是长的前缀 | PASS |
| 14 | tree_insert_reverse_order | 先插长序列再插短前缀，split 正确 | PASS |
| 15 | tree_insert_empty_key | 空 key 插入不崩溃 | PASS |
| 16 | tree_single_token_sequences | 单 token 序列插入和查找 | PASS |

### 搜索 (search)

| # | 用例名 | 验证点 | 结果 |
|---|--------|--------|------|
| 17 | tree_search_prefix_match | 查找比树中更长的序列，返回前缀匹配长度 | PASS |
| 18 | tree_search_no_match | 完全不匹配的序列返回 success=false | PASS |
| 19 | tree_search_partial_edge_match | 边内部分匹配，matched_length 只算匹配部分 | PASS |
| 20 | tree_search_empty_key | 空 key 搜索不崩溃 | PASS |
| 21 | tree_multiple_branches | 5 条分支全部可查找，前缀匹配正确 | PASS |

### 失效 (invalidate)

| # | 用例名 | 验证点 | 结果 |
|---|--------|--------|------|
| 22 | tree_invalidate_cell | invalidate 后对应 generation 变为 0 | PASS |
| 23 | tree_invalidate_nonexistent_cell | invalidate 不存在的 cell 不崩溃 | PASS |
| 24 | tree_invalidate_after_split | split 后 invalidate 共享前缀的 cell，两条路径都受影响 | PASS |
| 25 | tree_invalidate_all_cells_in_node | 全部 cell 失效后树结构仍完整 | PASS |
| 26 | tree_cell_reuse_different_nodes | 不同 key 引用同一 cell，invalidate 同时影响两者 | PASS |

### 其他

| # | 用例名 | 验证点 | 结果 |
|---|--------|--------|------|
| 27 | tree_clear | clear 后所有 search 返回无匹配 | PASS |
| 28 | tree_extra_key_isolation | 同 token 不同 extra_key 互不干扰 | PASS |
| 29 | tree_dot_visualization | dot() 输出合法 GraphViz 格式 | PASS |
| 30 | tree_hash_consistency | hash(k1)==hash(k2) when k1==k2 | PASS |
| 31 | tree_deep_chain | 50 层深链，search 匹配全部 50 token | PASS |
| 32 | tree_large_scale_insert | 500 条随机序列(长度5-20, vocab=100)全部插入和查找 | PASS |

---

## 阶段 3: Prefix Cache 集成测试 (16/16 PASS)

使用 `PrefixCacheHarness` 模拟 llama_kv_cache 的 prefix cache 逻辑：
- 复刻 promote / find / reclaim 的完整逻辑
- 模拟 cell_generations[] 和 CellState（占用/空闲/seq_id/pos）
- 模拟 overwrite_cell（apply_ubatch 覆写）和 free_cell（seq_rm 释放）

| # | 用例名 | 验证点 | 结果 |
|---|--------|--------|------|
| 33 | harness_promote_then_find | promote 后 find 返回完整匹配 | PASS |
| 34 | harness_find_prefix_of_longer_query | promote [1,2,3], find [1,2,3,4,5] 返回 3 | PASS |
| 35 | harness_generation_invalidation | 覆写中间 cell 后 find 在失效处截断 | PASS |
| 36 | harness_generation_invalidation_first_cell | 覆写第一个 cell 后 find 返回 0 | PASS |
| 37 | harness_reclaim_empty_cells | 空 cell 被 reclaim 恢复 pos/seq_id | PASS |
| 38 | harness_reclaim_occupied_by_other | 被其他 seq 占用的 cell 阻止 reclaim | PASS |
| 39 | harness_reclaim_already_correct | 已有正确 seq/pos 的 cell 计为 reclaimed | PASS |
| 40 | harness_seq_rm_invalidates_tree | free_cell 后树中对应条目失效 | PASS |
| 41 | harness_overwrite_then_repromote | 覆写后 re-promote 新 cells，find 返回新值 | PASS |
| 42 | harness_promote_find_with_extra_key | 不同 extra_key 隔离，互不影响 | PASS |
| 43 | harness_reclaim_after_generation_invalidation | 部分 cell 失效时 reclaim 在失效处停止 | PASS |
| 44 | harness_full_cycle_promote_free_repromote | 完整生命周期：promote→free→re-promote→find | PASS |
| 45 | harness_multiple_sequences_sharing_prefix | 两个请求共享 system prompt 前缀 | PASS |
| 46 | harness_reclaim_then_free_rollback | 模拟 prepare() dry-run 的 reclaim→rollback | PASS |
| 47 | harness_interleaved_promote_invalidate | 交错 promote/invalidate 后数据一致 | PASS |
| 48 | harness_clear_tree_preserves_generations | clear 树不影响 generation 计数器 | PASS |

---

## 阶段 4: 端到端流水线测试 (4/4 PASS)

使用真实模型 **Qwen3 0.6B Instruct (Q8_0)** 加载推理，通过 `dynamic_cast<llama_kv_cache*>`
直接访问 KV cache 的 prefix cache API。

### 测试场景

两个查询共享同一 system prompt 前缀：
```
共享前缀 (17 tokens): "You are a helpful assistant. Please answer the following
                        question carefully and concisely."
查询 A (24 tokens):    前缀 + " What is the capital of France?"
查询 B (27 tokens):    前缀 + " What is the largest planet in the solar system?"
实际共享 token 数:      20 tokens (tokenizer 将部分边界 token 合并)
```

### 测试步骤与结果

| 步骤 | 操作 | 预期 | 实际 | 结果 |
|------|------|------|------|------|
| [5] | decode query A (首次推理) | 成功，KV data 写入 cells | llama_decode 返回 0 | PASS |
| [6] | auto-promote 验证 | apply() 后树自动注册 | find(A)=24, find(B)=20 | PASS |
| [8] | 前缀匹配 query B | 共享前缀 20 tokens 被识别 | matched=20, cells 正确 | PASS |
| [9] | clear 后失效检测 | 清空 KV cache 后树条目全部失效 | matched=0 | PASS |

### 关键验证

```
[6] Checking prefix cache after decode A...
    prefix_cache_find(query_A): matched=24        ← auto-promote 生效
    prefix_cache_find(query_B): matched=20        ← 前缀共享识别正确
    SUCCESS: Prefix cache found 20 shared tokens for query B

[9] Testing generation invalidation...
    KV cache cleared.
    prefix_cache_find after clear: matched=0       ← generation 失效机制正确
    PASS: Cache correctly invalidated after clear
```

### 编译与运行说明

**注意**: llama-cli 存在上游 winsock 链接问题（`-lws2_32` 缺失，与 prefix cache 无关），
因此端到端测试绕过 `common` 库，直接链接 `libllama.a` + `ggml*.a`。

---

## 测试过程中发现并修复的 Bug

### Bug #5: insert full match 不更新 value (Phase 3 发现)

**问题**: 当对同一 key 重复 insert（re-promote 场景），insert 只做 `ref_count++`，
不更新 cell_indices 和 cell_generations。导致覆写后 re-promote 无效。

**测试暴露**: `harness_overwrite_then_repromote` 和 `harness_full_cycle_promote_free_repromote`

**修复**: full match 时沿路径重新 walk，对每个节点执行 unregister→update subview→register。

**修复位置**: `llama-radix-tree.cpp` insert() full match 分支

### Bug #6: key 是已有 edge 前缀时未 split (Phase 3 发现)

**问题**: 当 `matched == key.size()` 但 `edge_matched < deepest->key.size()` 时
（key 是某条边的前缀），直接进入 full match 更新路径，用短 value 替换长节点的 value，
导致 `vector::operator[]` 越界断言失败。

**测试暴露**: `tree_insert_reverse_order` (Phase 2 回归测试)

**修复**: 在 full match 判断前增加条件 split：如果 key 消耗完毕但边未完全匹配，
先 split 出上半部分再进入更新路径。确保节点的 key 长度始终与 value 长度一致。

**修复位置**: `llama-radix-tree.cpp` insert() 在 full match 检查之前

---

## 执行结果汇总

```
Phase 1 (VectorView):             9 /  9  PASS
Phase 2 (Radix Tree Core):       23 / 23  PASS
Phase 3 (Prefix Cache API):      16 / 16  PASS
Phase 4 (End-to-End Pipeline):    4 /  4  PASS
──────────────────────────────────────────────
Total:                            52 / 52  PASS

llama library build:              ✅ 编译通过 (无回归)
Bug fixes during testing:         2 个 (Bug #5, #6)
Model used for e2e:               Qwen3 0.6B Instruct Q8_0
```

---

## 测试覆盖分析

### 已覆盖的核心路径

- **VectorView**: 构造、切分、共享存储、空安全、越界、迭代、嵌套 subview
- **insert**: 首次插入、公共前缀 split、重复插入(value 更新)、前缀/后缀关系、空 key、key 是 edge 前缀时的 split
- **search**: 精确匹配、前缀匹配、无匹配、部分边匹配、空 key
- **split**: 通过 insert 间接测试，包括 split 后 cell 追踪和 re-promote 后的一致性
- **invalidate**: 单 cell、全 cell、不存在 cell、split 后 cell、多节点共享 cell
- **promote → find 往返**: generation 快照 + 验证
- **generation 失效**: 覆写(apply_ubatch)、释放(seq_rm)、清空(clear) 三种触发路径
- **reclaim**: 空 cell 恢复、已占用 cell 阻塞、已正确 cell 跳过
- **re-promote**: 旧条目失效后 re-promote 新 cells
- **dry-run rollback**: reclaim 后 rollback 恢复原状
- **多请求前缀共享**: 两个请求复用同一 system prompt（真实模型验证）
- **extra_key 隔离**: 不同上下文互不干扰
- **auto-promote**: apply() 后自动注册到树（端到端验证）
- **真实推理**: llama_decode 与 prefix cache 协同工作

### 未覆盖

- prepare() 完整 reclaim→find_slot→apply→restore 循环（需要两次 decode 的场景）
- 多 stream / 多 seq_id 场景
- LRU 淘汰策略（功能未实现）
- seq_add / seq_div 后树一致性（功能未实现）

---

*测试完成于 2026-03-19。所有 4 个阶段 52 个测试点全部通过。*
