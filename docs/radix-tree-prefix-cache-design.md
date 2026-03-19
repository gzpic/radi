# Radix Tree Prefix Cache — 设计与实现文档

> 将 mllm 的 RadixTree KV Cache 前缀匹配系统移植到 llama.cpp
> 作者：Claude Opus 4.6 + 陈俊
> 日期：2026-03-19

---

## 目录

1. [整体目标](#1-整体目标)
2. [架构概览（字符图）](#2-架构概览)
3. [mllm vs llama.cpp KV Cache 对比](#3-mllm-vs-llamacpp-kv-cache-对比)
4. [数据结构设计](#4-数据结构设计)
5. [核心算法](#5-核心算法)
6. [集成方案](#6-集成方案)
7. [重难点与解决方案](#7-重难点与解决方案)
8. [Bug 修复记录](#8-bug-修复记录)
9. [文件清单](#9-文件清单)
10. [待完成项](#10-待完成项)

---

## 1. 整体目标

多轮对话和批量推理中，不同请求经常共享相同的 system prompt 或历史上下文前缀。如果
每次都重新计算这些前缀的 KV 数据，会浪费大量的 GPU 计算。

**Radix Tree Prefix Cache** 的目标：用一棵基数树（压缩前缀树）索引已经计算过的 token
序列及其对应的 KV cache 位置，当新请求到来时，查找最长公共前缀，直接复用已有的 KV 数据，
只计算剩余的新 token。

```
请求 A: [SYS][SYS][SYS][用户1][用户1]      → 全量计算 5 token
请求 B: [SYS][SYS][SYS][用户2][用户2][用户2] → 命中 3 token 前缀，只算 3 个新 token
                                                 节省 60% 计算量
```

---

## 2. 架构概览

### 2.1 系统总体结构

```
┌─────────────────────────────────────────────────────────────┐
│                    llama_context                            │
│                                                             │
│  ┌──────────────┐    ┌───────────────────────────────────┐  │
│  │  llama_batch  │───▶│     llama_kv_cache_context        │  │
│  │  (用户输入)   │    │                                   │  │
│  └──────────────┘    │  prepare()  ──┐                    │  │
│                      │  apply()   ──┤ 推理流水线          │  │
│                      │  next()    ──┘                    │  │
│                      └───────────────┬───────────────────┘  │
│                                      │                      │
│  ┌───────────────────────────────────▼───────────────────┐  │
│  │               llama_kv_cache                          │  │
│  │                                                       │  │
│  │  ┌─────────────┐  ┌──────────────────────────────┐    │  │
│  │  │ v_cells[]   │  │   Prefix Cache (新增)        │    │  │
│  │  │ v_heads[]   │  │                              │    │  │
│  │  │ KV tensors  │  │  ┌────────────────────────┐  │    │  │
│  │  │ (GPU memory)│  │  │  llama_radix_tree      │  │    │  │
│  │  │             │  │  │  (token→cell 索引)     │  │    │  │
│  │  └─────────────┘  │  └────────────────────────┘  │    │  │
│  │                    │  ┌────────────────────────┐  │    │  │
│  │                    │  │  cell_generations[][]  │  │    │  │
│  │                    │  │  (失效检测计数器)      │  │    │  │
│  │                    │  └────────────────────────┘  │    │  │
│  │                    └──────────────────────────────┘    │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 推理流水线中的前缀缓存工作流

```
  prepare()                         apply()
  ─────────                         ───────
  ┌──────────────────┐
  │ 1. 搜索前缀树    │   tokens: [A B C D E F]
  │    prefix_find() │   树中有: [A B C] → cells [5,8,2]
  │    命中 3 token  │
  └────────┬─────────┘
           ▼
  ┌──────────────────┐
  │ 2. 回收缓存cell  │   cell[5].pos=0, seq=1  (恢复元数据)
  │    reclaim()     │   cell[8].pos=1, seq=1
  │    标记为已占用   │   cell[2].pos=2, seq=1
  └────────┬─────────┘
           ▼
  ┌──────────────────┐
  │ 3. find_slot()   │   为剩余 token [D E F] 分配 cells
  │    分配新 slot   │   → cells [10,11,12]
  └────────┬─────────┘
           ▼
  ┌──────────────────┐
  │ 4. 替换cell索引  │   最终 slot: [5,8,2, 10,11,12]
  │    前缀用缓存cell│               ▲缓存▲  ▲新分配▲
  └────────┬─────────┘
           ▼                        ┌──────────────────┐
  ┌──────────────────┐              │ 5. auto-promote  │
  │  apply_ubatch()  │─────────────▶│    将完整序列     │
  │  写入KV数据      │              │    注册到前缀树   │
  └──────────────────┘              └──────────────────┘
```

---

## 3. mllm vs llama.cpp KV Cache 对比

这是移植的核心难点。两个系统的 KV cache 地址模型完全不同：

```
                    mllm                          llama.cpp
            ┌─────────────────┐            ┌─────────────────────┐
 地址模型   │  vp_addr_t       │            │  cell index (uint32) │
            │  虚拟地址，稳定   │            │  环形缓冲区下标       │
            │  分配后不变       │            │  随时可被覆写         │
            └─────────────────┘            └─────────────────────┘
            ┌─────────────────┐            ┌─────────────────────┐
 层独立性   │  每层独立地址     │            │  所有层共享同一 cell  │
            │  layer_addrs[]   │            │  cell_idx 跨所有层   │
            └─────────────────┘            └─────────────────────┘
            ┌─────────────────┐            ┌─────────────────────┐
 内存管理   │  专用 Allocator   │            │  ring buffer         │
            │  TLB 页表映射     │            │  find_slot 线性搜索   │
            └─────────────────┘            └─────────────────────┘
            ┌─────────────────┐            ┌─────────────────────┐
 失效机制   │  地址稳定，       │            │  cell 随时被新 token  │
            │  无需失效检测     │            │  覆写，必须检测失效   │
            └─────────────────┘            └─────────────────────┘
```

### 关键适配点

| 问题 | mllm 做法 | llama.cpp 适配方案 |
|------|-----------|-------------------|
| 地址稳定性 | vp_addr_t 分配后不变 | generation counter 检测 cell 是否被覆写 |
| 层映射 | 每层独立 vp_addr_t 数组 | 单个 cell_idx，所有层共享 |
| 失效检测 | 不需要 | cell_to_nodes_ 反向索引 + invalidate_cell() |
| 哈希函数 | XXHash (HaHaHash) | FNV-1a（无外部依赖） |
| 内存分配 | 自定义 Allocator | std::shared_ptr + VectorView |

---

## 4. 数据结构设计

### 4.1 Radix Tree 节点结构

```
llama_radix_tree
├── root_ (unique_ptr<llama_radix_node>)    哨兵根节点, key=[-1]
├── cell_to_nodes_                          反向索引: cell_idx → [(node,offset)]
├── node_count_
└── options_

llama_radix_node
├── key: llama_radix_node_key
│   ├── token_ids: VectorView<llama_token>  边上的 token 序列
│   └── extra_key: int64_t                  区分上下文(如 LoRA adapter)
├── value: llama_radix_node_value
│   ├── cell_indices: VectorView<uint32_t>  对应 KV cache cell 下标
│   └── cell_generations: VectorView<uint64_t>  插入时的 generation 快照
├── parent: llama_radix_node*
├── children: unordered_map<key, node*>     子节点映射
├── ref_count, hit_count, last_accessed     LRU 淘汰元数据
```

### 4.2 VectorView 零拷贝视图

```
                        shared_ptr<vector<T>>
                        ┌───┬───┬───┬───┬───┬───┬───┬───┐
   底层存储:            │ A │ B │ C │ D │ E │ F │ G │ H │
                        └───┴───┴───┴───┴───┴───┴───┴───┘
                          0   1   2   3   4   5   6   7

   split 之前:          ┌───────────────────────────────┐
   原始节点 view:       │  offset=0, length=8            │ → [A B C D E F G H]
                        └───────────────────────────────┘

   split(node, 3) 之后:
   上半部 (upper):      ┌───────────────┐
                        │ off=0, len=3  │ → [A B C]       ← 共享同一底层存储
                        └───────────────┘
   下半部 (lower):              ┌───────────────────────┐
                                │ off=3, len=5          │ → [D E F G H]
                                └───────────────────────┘

   内存开销: O(1) 而非 O(n) 拷贝
```

### 4.3 Radix Tree 示例

```
插入序列:
  [1,2,3,4,5]   → cells [10,11,12,13,14]
  [1,2,3,6,7,8] → cells [10,11,12,20,21,22]
  [1,2,9]       → cells [10,11,30]

树结构:

            ROOT [-1]
              │
         key=[1,2]
         cells=[10,11]
        ╱              ╲
  key=[3]              key=[9]
  cells=[12]           cells=[30]
  ╱        ╲
key=[4,5]  key=[6,7,8]
cells=     cells=
[13,14]    [20,21,22]

查找 [1,2,3,6,7,8]:
  ROOT → [1,2](匹配2) → [3](匹配1) → [6,7,8](匹配3)
  总匹配: 6 tokens
  返回 cells: [10,11,12,20,21,22]
```

### 4.4 Generation Counter 失效检测

```
时间线:

  T0: cell[5] 存储 token A 的 KV, generation = 1
      树中记录: token A → cell 5, gen=1  ✓

  T1: cell[5] 被新 token X 覆写
      cell_generations[0][5]++ → generation = 2
      invalidate_cell(5) → 树中 gen 标记为 0

  T2: 查询 token A → cell 5
      树中 gen=0, 当前 gen=2
      gen==0 → 已失效，不使用 ✗

  无需遍历树！覆写时 O(1) 通过 cell_to_nodes_ 反向索引定位
```

```
cell_to_nodes_ 反向索引:

  cell_idx=5 → [(node_A, offset=0), (node_B, offset=2)]
                       │                      │
                       ▼                      ▼
                  node_A.value.          node_B.value.
                  cell_gens[0]=0         cell_gens[2]=0
                  (被标记失效)            (被标记失效)
```

---

## 5. 核心算法

### 5.1 插入 (insert)

```
insert(key=[1,2,3,4], cells=[a,b,c,d]):

  1. 从 root 开始向下走, 逐边匹配 token 序列
  2. 匹配到 [1,2,3] 后, 剩余 [4] 无法继续

  情况 A: 完全匹配边        → 继续向下走
  情况 B: 部分匹配边 [1,2]  → split 该节点
  情况 C: 无匹配子节点       → 创建新叶子

  split 过程:
  ┌──────────┐          ┌──────────┐
  │ [1,2,3,4]│   ──▶    │ [1,2]    │ (upper, 截断)
  │ cells=   │          │ cells=   │
  │ [a,b,c,d]│          │ [a,b]    │
  └──────────┘          └────┬─────┘
                             │
                        ┌────▼─────┐
                        │ [3,4]    │ (lower, 新建)
                        │ cells=   │
                        │ [c,d]    │  ← VectorView subview, 零拷贝
                        └──────────┘
```

### 5.2 搜索 (search)

```
search(key=[1,2,3,6]):

  cur = root
  matched = 0
  path = [(root, 0)]

  while matched < len(key):
    for child in cur.children:
      ml = matched_length(key[matched:], child.key)
      if ml > 0:
        matched += ml
        path.append((child, ml))
        if ml < len(child.key):
          break  ← 部分匹配，不能继续向下！
        cur = child
        break

  result.matched_length = matched
  result.cell_indices = flatten(path 中每个 (node, len) 的前 len 个 cell)
```

### 5.3 失效 (invalidate_cell)

```
invalidate_cell(cell_idx=5):

  entries = cell_to_nodes_[5]
  // → [(node_X, 0), (node_Y, 3)]

  for (node, offset) in entries:
    node.value.cell_generations[offset] = 0  // 标记为无效

  erase cell_to_nodes_[5]

  时间复杂度: O(k), k = 引用该 cell 的节点数 (通常很小)
```

---

## 6. 集成方案

### 6.1 修改点一览

```
llama-kv-cache.h         新增 4 个 public API + 3 个 private 成员
llama-kv-cache.cpp       修改 6 个函数，新增 4 个函数
llama-radix-tree.h       新文件，~230 行
llama-radix-tree.cpp     新文件，~360 行
CMakeLists.txt           新增编译目标
```

### 6.2 API 设计

```cpp
// 启用前缀缓存
void prefix_cache_enable(const llama_radix_tree_options & options = {});

// 将已计算的 token 序列注册到树中
void prefix_cache_promote(
    const std::vector<llama_token> & token_ids,
    const std::vector<uint32_t>    & cell_indices,
    int64_t extra_key = 0);

// 查找最长有效前缀，返回匹配长度
int32_t prefix_cache_find(
    const std::vector<llama_token> & token_ids,
    std::vector<uint32_t>          & out_cell_indices,
    int64_t extra_key = 0);

// 回收匹配的 cell（恢复 pos + seq_id 元数据）
int32_t prefix_cache_reclaim(
    llama_seq_id seq_id,
    const llama_token * tokens,
    uint32_t n_tokens,
    const llama_pos * positions,
    int64_t extra_key = 0);
```

### 6.3 在推理流水线中的插入点

```
llama_context::encode/decode()
  └── llama_kv_cache_context::prepare()
        │
        │  ┌─── 新增: prefix_cache_find() ──────────┐
        │  │    搜索前缀树, 获取匹配的 cell indices   │
        │  └─── prefix_cache_reclaim() ──────────────┘
        │        恢复 cell 元数据 (pos, seq_id)
        │
        ├── find_slot()  // 缓存 cell 已标记占用, 不会被重新分配
        │
        └── 替换 slot_info 中前缀部分的 cell 索引
              sinfo.idxs[0][0..n_prefix] = cached_cells

  └── llama_kv_cache_context::apply()
        │
        ├── kv->apply_ubatch(sinfo, ubatch)
        │     │
        │     └── 每个被覆写的 cell:
        │           cell_generations[strm][idx]++
        │           prefix_tree->invalidate_cell(idx)
        │
        └── 新增: kv->prefix_cache_promote(tokens, cells)
              将刚完成的 ubatch 注册到树中
```

### 6.4 prepare() 中的 dry-run 模式处理

```
prepare() 是试运行(dry-run): 先分配 slot, 做完之后全部回滚.
这对前缀缓存有特殊影响:

  ┌────────────────────────────────────────────────────────┐
  │ 正向: reclaim cells → find_slot → apply_ubatch (模拟) │
  │                                                        │
  │ 回滚: 逆序遍历 states[]                               │
  │   - 恢复 v_cells 到修改前状态                          │
  │   - 恢复 v_heads 到修改前状态                          │
  │   - 对 reclaimed_cells 做 cells.rm() 撤销回收         │
  └────────────────────────────────────────────────────────┘

state_t 中新增:
  std::vector<uint32_t> reclaimed_cells;  // 本轮回收的 cell 列表
  uint32_t reclaimed_stream;              // 所属 stream
```

### 6.5 seq_rm / seq_keep 同步

```
seq_rm(seq_id, p0, p1):
  for each cell i in range [p0, p1]:
    if cell freed:
      cell_generations[strm][i]++     ← 递增 generation
      prefix_tree->invalidate_cell(i) ← 标记树中引用为无效

seq_keep(seq_id):
  for each cell i:
    if cell freed (didn't have seq_id):
      cell_generations[strm][i]++
      prefix_tree->invalidate_cell(i)
```

---

## 7. 重难点与解决方案

### 7.1 核心难点：cell 地址不稳定

**问题**: mllm 的 vp_addr_t 一旦分配就不变, 但 llama.cpp 的 cell index 随时可被覆写。
树中记录的 `token A → cell 5` 可能在下一次 apply_ubatch 后就失效了。

**解决方案**: Generation Counter + 反向索引

```
                    ┌───────────────────────┐
                    │  cell_generations[][] │
                    │  每次覆写 cell 时 +1   │
                    └───────────┬───────────┘
                                │
            ┌───────────────────┼───────────────────┐
            ▼                   ▼                   ▼
    insert 时快照 gen    find 时比对 gen      覆写时 invalidate
    存入树节点          不匹配则丢弃          O(1) 反向索引定位

    树中: gen=3          当前: gen=5          cell_to_nodes_[idx]
    → 失效!              → 不使用            → 所有引用节点 gen=0
```

### 7.2 难点：prepare() 的 dry-run 模式

**问题**: prepare() 需要"先模拟执行, 记录分配结果, 然后回滚"。但 reclaim 修改了
cells 的状态(恢复了 pos 和 seq_id), 回滚时必须撤销这些修改。

**解决方案**: 在 state_t 中记录 reclaimed_cells, 回滚时 cells.rm(idx) 恢复

```
prepare() 执行流:

  for each ubatch:
    reclaim prefix cells (修改 cells 状态)  ← 需要回滚
    find_slot (基于已修改的 cells)
    apply_ubatch (模拟)
    save state (含 reclaimed_cells)

  for each state (逆序):
    restore v_cells from saved copy
    restore v_heads
    for idx in reclaimed_cells:
      cells.rm(idx)                          ← 撤销 reclaim
```

### 7.3 难点：部分边匹配的搜索终止

**问题**: 在原始 mllm 代码中, 部分匹配一条边后(如边是 [A,B,C], 只匹配了 [A,B]),
搜索/插入错误地继续进入子节点。子节点的 token 假设从 C 之后开始, 导致结果错误。

**解决方案**: 匹配完一条边后检查是否完整匹配, 如果不是则 break

```cpp
// 修复前 (BUG):
for (auto & [child_key, child_ptr] : cur->children) {
    size_t ml = matched_length(rest_key, child_key);
    if (ml > 0) { deepest = child_ptr; cur = deepest; break; }
    //                                  ^^^ 即使部分匹配也继续下降
}

// 修复后:
for (...) {
    if (ml > 0) { deepest = child_ptr; edge_matched = ml; break; }
}
if (edge_matched < deepest->key.token_ids.size()) {
    break;  // 部分匹配, 不能继续下降到子节点
}
cur = deepest;
```

### 7.4 难点：operator== 与 hash 一致性

**问题**: 哈希函数包含了 extra_key, 但 operator== 只比较 token_ids。
unordered_map 要求: `a == b` ⟹ `hash(a) == hash(b)`。违反这个契约会导致
map 查找行为未定义。

**解决方案**: operator== 中加入 extra_key 比较

```cpp
bool operator==(const llama_radix_node_key & o) const noexcept {
    return extra_key == o.extra_key && token_ids == o.token_ids;
    //     ^^^^^^^^^^^^^^^^^^^^^^^^ 修复: 必须包含 extra_key
}
```

### 7.5 难点：VectorView 空指针防护

**问题**: 默认构造的 VectorView 其 data_ 为 nullptr, 调用 data() 或 end() 时
会解引用空指针崩溃。

**解决方案**: 空指针检查

```cpp
pointer data() { return data_ ? data_->data() + offset_ : nullptr; }
iterator end() { return data() ? data() + length_ : nullptr; }
```

---

## 8. Bug 修复记录

| # | Bug | 影响 | 修复 |
|---|-----|------|------|
| 1 | operator== 不含 extra_key | unordered_map 行为未定义 | 加入 extra_key 比较 |
| 2 | VectorView::data() 空指针 | 默认构造后崩溃 | 空指针检查 |
| 3 | 部分边匹配后继续下降 | 搜索/插入结果错误 | break 终止 |
| 4 | emplace 失败时内存泄漏 | new 的节点无人释放 | 检查 inserted, 失败则 delete |

---

## 9. 文件清单

```
llama.cpp/src/
├── llama-radix-tree.h        新文件  ~230 行  数据结构定义
├── llama-radix-tree.cpp      新文件  ~360 行  算法实现
├── llama-kv-cache.h          修改    +25 行   API 声明 + 私有成员
├── llama-kv-cache.cpp        修改    +180 行  集成逻辑
│   ├── 构造函数              初始化 cell_generations
│   ├── clear()               重置前缀树 + generation
│   ├── prefix_cache_enable() 创建树实例
│   ├── prefix_cache_promote()注册 token→cell 映射
│   ├── prefix_cache_find()   查找 + generation 验证
│   ├── prefix_cache_reclaim()恢复 cell 元数据
│   ├── seq_rm()              释放时 invalidate
│   ├── seq_keep()            释放时 invalidate
│   ├── apply_ubatch()        覆写时 generation++ & invalidate
│   └── (context) apply()     自动 promote
└── CMakeLists.txt            修改    +1 行    添加编译目标
```

---

## 10. 待完成项

- [ ] **seq_add / seq_div 同步**: position 变化时可能需要更新树（当前靠 generation 兜底）
- [ ] **LRU 淘汰机制**: 节点已有 ref_count/hit_count/last_accessed 字段, 需实现淘汰策略
- [ ] **启用入口**: 在 llama_context 初始化或 API 层暴露 prefix_cache_enable() 调用
- [ ] **单元测试**: 独立测试 radix tree 的 insert/search/split/invalidate
- [ ] **多 stream 支持**: 当前 promote/find 仅使用 stream 0, 需扩展到多 stream 场景

---

*本文档随代码演进持续更新。*
