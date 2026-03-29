# 编译与测试指南

---

## 一、环境要求

| 项目 | 要求 |
|------|------|
| 操作系统 | Windows 11（通过 MSYS2 ucrt64 shell 编译） |
| 编译器 | MinGW-w64 GCC（ucrt64 工具链） |
| 构建系统 | CMake + Ninja |
| MSYS2 包 | `mingw-w64-ucrt-x86_64-gcc` `mingw-w64-ucrt-x86_64-cmake` `mingw-w64-ucrt-x86_64-ninja` |
| 模型文件 | `models/qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf`（L3/L4 测试需要） |

**MSYS2 shell 启动命令：**

```bash
D:/SoftwareFilePlace/MSYS2/msys2_shell.cmd -defterm -here -no-start -ucrt64 -c "<命令>"
```

> 不要用 Git Bash 编译。Git Bash 的路径翻译会和 MinGW gcc 冲突，导致静默失败。

---

## 二、编译主项目（libllama.a）

```bash
# 在项目根目录下
cd e:/codex/llama.cpp

# 配置（只需一次）
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release

# 编译
cmake --build build -j$(nproc)
```

编译产物：

| 文件 | 路径 |
|------|------|
| libllama.a | `build/src/libllama.a` |
| ggml.a | `build/ggml/src/ggml.a` |
| ggml-base.a | `build/ggml/src/ggml-base.a` |
| ggml-cpu.a | `build/ggml/src/ggml-cpu.a` |

> 注意：ggml 库文件名没有 `lib` 前缀（是 `ggml.a` 不是 `libggml.a`），链接时必须用 `-l:ggml.a` 语法。

---

## 三、测试文件一览

### 测试分层

| 层级 | 文件 | 用例数 | 需要模型 | 说明 |
|------|------|--------|----------|------|
| L1 | `test-radix-tree.cpp` | 32 | 否 | Radix Tree 纯数据结构测试 |
| L2 | `test-radix-tree-integration.cpp` | 16 | 否 | KV 语义集成测试（promote/find/reclaim） |
| L2 | `test-radix-tree-features.cpp` | 41 | 否 | A2/A18/A4/A6 特性 + 10 个组合路径测试 |
| L3 | `test-radix-tree-consistency.cpp` | 8 | 是 | 推理一致性（logits bit-exact 验证） |
| L4 | `test-radix-tree-agent-bench.cpp` | — | 是 | 场景化 benchmark（5 轮 Agent 对话） |

### 其他测试文件

| 文件 | 说明 |
|------|------|
| `test-radix-tree-e2e.cpp` | 早期端到端测试 |
| `test-agent-bench-ab.cpp` | A/B 对比 benchmark |

---

## 四、编译测试

### L1 + L2 测试（不需要模型，纯逻辑验证）

```bash
# L1: Radix Tree 数据结构
g++ -std=c++17 -O2 \
    -I include -I src -I ggml/include \
    -o tests/test-radix-tree.exe \
    tests/test-radix-tree.cpp \
    -L build/src -L build/ggml/src \
    -lllama -l:ggml.a -l:ggml-base.a -l:ggml-cpu.a \
    -lpthread -lgomp

# L2: KV 语义集成
g++ -std=c++17 -O2 \
    -I include -I src -I ggml/include \
    -o tests/test-radix-tree-integration.exe \
    tests/test-radix-tree-integration.cpp \
    -L build/src -L build/ggml/src \
    -lllama -l:ggml.a -l:ggml-base.a -l:ggml-cpu.a \
    -lpthread -lgomp

# L2: 特性 + 组合路径
g++ -std=c++17 -O2 \
    -I include -I src -I ggml/include \
    -o tests/test-radix-tree-features.exe \
    tests/test-radix-tree-features.cpp \
    -L build/src -L build/ggml/src \
    -lllama -l:ggml.a -l:ggml-base.a -l:ggml-cpu.a \
    -lpthread -lgomp
```

### L3 推理一致性测试（需要模型）

```bash
g++ -std=c++17 -O2 \
    -I include -I src -I ggml/include \
    -o tests/test-radix-tree-consistency.exe \
    tests/test-radix-tree-consistency.cpp \
    -L build/src -L build/ggml/src \
    -lllama -l:ggml.a -l:ggml-base.a -l:ggml-cpu.a \
    -lpthread -lgomp
```

### L4 场景 Benchmark（需要模型）

```bash
g++ -std=c++17 -O2 \
    -I include -I src -I ggml/include \
    -o tests/test-radix-tree-agent-bench.exe \
    tests/test-radix-tree-agent-bench.cpp \
    -L build/src -L build/ggml/src \
    -lllama -l:ggml.a -l:ggml-base.a -l:ggml-cpu.a \
    -lpthread -lgomp
```

### 通用编译模板

所有测试的编译命令结构相同，只换源文件名和输出名：

```bash
g++ -std=c++17 -O2 \
    -I include -I src -I ggml/include \
    -o tests/<输出名>.exe \
    tests/<源文件>.cpp \
    -L build/src -L build/ggml/src \
    -lllama -l:ggml.a -l:ggml-base.a -l:ggml-cpu.a \
    -lpthread -lgomp
```

| 参数 | 说明 |
|------|------|
| `-std=c++17` | C++17 标准（llama.cpp 要求） |
| `-O2` | 优化级别 |
| `-I include -I src -I ggml/include` | 头文件搜索路径 |
| `-L build/src -L build/ggml/src` | 库文件搜索路径 |
| `-lllama` | 链接 libllama.a |
| `-l:ggml.a -l:ggml-base.a -l:ggml-cpu.a` | 链接 ggml 库（用 `-l:` 精确匹配文件名） |
| `-lpthread` | POSIX 线程库 |
| `-lgomp` | OpenMP（ggml-cpu.a 的并行依赖） |

---

## 五、运行测试

### L1 + L2（无参数，直接运行）

```bash
./tests/test-radix-tree.exe
./tests/test-radix-tree-integration.exe
./tests/test-radix-tree-features.exe
```

预期输出：每个 PASS/FAIL 行，末尾汇总 `X/X tests passed`。

### L3（需要模型路径参数）

```bash
./tests/test-radix-tree-consistency.exe models/qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf
```

预期输出：C1-C8 每个用例 PASS/FAIL + max_abs_diff / cosine_sim 指标。

### L4（需要模型路径参数）

```bash
./tests/test-radix-tree-agent-bench.exe models/qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf
```

预期输出：5 轮 Agent 对话的 TTFT / hit rate / projected savings 表格。

### 快速全量回归

```bash
./tests/test-radix-tree.exe && \
./tests/test-radix-tree-integration.exe && \
./tests/test-radix-tree-features.exe
```

三个命令链式执行，任一失败则中断。预期 89/89 通过。

加上 L3：

```bash
./tests/test-radix-tree.exe && \
./tests/test-radix-tree-integration.exe && \
./tests/test-radix-tree-features.exe && \
./tests/test-radix-tree-consistency.exe models/qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf
```

预期 97/97 通过。

---

## 六、常见编译问题

| 问题 | 现象 | 解决 |
|------|------|------|
| Git Bash 编译静默失败 | g++ 返回 1 无输出 | 改用 MSYS2 ucrt64 shell |
| `-lggml` 找不到库 | `cannot find -lggml` | 改用 `-l:ggml.a`（精确文件名匹配） |
| OpenMP 链接错误 | `undefined reference to GOMP_barrier` | 添加 `-lgomp` |
| `llama_prefix_match_info` 未声明 | 编译报错 | 确认 `src/llama-kv-cache.h` 已包含结构体定义 |
| 环境 vs 代码问题分不清 | 改代码后编不过 | `git stash` → 编译原始代码 → 同样失败则是环境问题 |
| L3 cosine_sim 偏低 | C6/C7 首次 FAIL | BPE 边界效应：分别 tokenize 后拼接 vector，不要 tokenize 拼接字符串 |
| fork 返回 -1 | C8 SKIP | 默认 `n_seq_max=1`，需 cmake 配置启用多序列 |
