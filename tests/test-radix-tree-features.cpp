// Phase 4: Tests for A2 (Checkpoint/Rollback), A18 (Fork/Merge),
//          A4 (Selective Trim), A6 (LRU Eviction)
//
// A6 tests operate directly on llama_radix_tree.
// A2/A18/A4 operate on llama_kv_cache which requires a model, so we test
// them through an extended PrefixCacheHarness that simulates the same
// seq_rm / seq_cp / seq_add semantics.

#include "llama-radix-tree.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

static int n_pass = 0;
static int n_fail = 0;

#define TEST(name) \
    static void test_##name(); \
    struct test_reg_##name { \
        test_reg_##name() { \
            printf("  TEST %-55s ", #name); \
            try { \
                test_##name(); \
                printf("[PASS]\n"); \
                n_pass++; \
            } catch (const std::exception & e) { \
                printf("[FAIL] %s\n", e.what()); \
                n_fail++; \
            } catch (...) { \
                printf("[FAIL] unknown exception\n"); \
                n_fail++; \
            } \
        } \
    } test_instance_##name; \
    static void test_##name()

#define ASSERT_TRUE(x)  do { if (!(x)) { throw std::runtime_error(std::string("ASSERT_TRUE failed: ") + #x + " at line " + std::to_string(__LINE__)); } } while(0)
#define ASSERT_FALSE(x) ASSERT_TRUE(!(x))
#define ASSERT_EQ(a, b) do { if ((a) != (b)) { throw std::runtime_error(std::string("ASSERT_EQ failed: ") + #a + "=" + std::to_string(a) + " != " + #b + "=" + std::to_string(b) + " at line " + std::to_string(__LINE__)); } } while(0)
#define ASSERT_GE(a, b) do { if ((a) < (b)) { throw std::runtime_error(std::string("ASSERT_GE failed: ") + #a + "=" + std::to_string(a) + " < " + #b + "=" + std::to_string(b) + " at line " + std::to_string(__LINE__)); } } while(0)
#define ASSERT_LE(a, b) do { if ((a) > (b)) { throw std::runtime_error(std::string("ASSERT_LE failed: ") + #a + "=" + std::to_string(a) + " > " + #b + "=" + std::to_string(b) + " at line " + std::to_string(__LINE__)); } } while(0)

// ============================================================================
// Helper: build key/value from token+cell vectors
// ============================================================================

static llama_radix_node_key make_key(const std::vector<llama_token> & tokens, int64_t extra = 0) {
    return llama_radix_node_key(llama_vector_view<llama_token>(tokens), extra);
}

static llama_radix_node_value make_value(const std::vector<uint32_t> & cells,
                                          const std::vector<uint64_t> & gens) {
    llama_radix_node_value v;
    v.cell_indices     = llama_vector_view<uint32_t>(cells);
    v.cell_generations = llama_vector_view<uint64_t>(gens);
    return v;
}

// ============================================================================
// A6: LRU Eviction Tests
// ============================================================================

TEST(lru_evict_basic) {
    // Insert several sequences, then evict down to a limit
    llama_radix_tree tree;

    // Insert 5 disjoint single-token sequences
    for (int i = 0; i < 5; i++) {
        std::vector<llama_token> tok = {(llama_token)(100 + i)};
        std::vector<uint32_t> cell = {(uint32_t)i};
        std::vector<uint64_t> gen = {1};
        tree.insert(make_key(tok), make_value(cell, gen));
    }

    // root + 5 leaves = 6 nodes
    ASSERT_EQ(tree.node_count(), 6);

    // Evict down to 3 nodes: should remove 3 leaves
    int32_t evicted = tree.evict_lru(3);
    ASSERT_EQ(evicted, 3);
    ASSERT_EQ(tree.node_count(), 3);
}

TEST(lru_evict_nothing_when_under_limit) {
    llama_radix_tree tree;
    std::vector<llama_token> tok = {1, 2, 3};
    std::vector<uint32_t> cell = {0, 1, 2};
    std::vector<uint64_t> gen = {1, 1, 1};
    tree.insert(make_key(tok), make_value(cell, gen));

    // root + 1 leaf = 2 nodes, limit is 10
    int32_t evicted = tree.evict_lru(10);
    ASSERT_EQ(evicted, 0);
    ASSERT_EQ(tree.node_count(), 2);
}

TEST(lru_evict_respects_access_order) {
    // Insert 3 sequences, access them in a specific order,
    // verify eviction removes the least-recently-accessed first
    llama_radix_tree tree;

    std::vector<llama_token> tok_a = {10};
    std::vector<llama_token> tok_b = {20};
    std::vector<llama_token> tok_c = {30};
    std::vector<uint32_t> cell_a = {0}, cell_b = {1}, cell_c = {2};
    std::vector<uint64_t> gen = {1};

    tree.insert(make_key(tok_a), make_value(cell_a, gen));
    // small delay so timestamps differ
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    tree.insert(make_key(tok_b), make_value(cell_b, gen));
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    tree.insert(make_key(tok_c), make_value(cell_c, gen));

    // Now search for tok_a to bump its last_accessed
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    auto r = tree.search(make_key(tok_a));
    ASSERT_TRUE(r.success);

    // root(1) + 3 leaves = 4
    ASSERT_EQ(tree.node_count(), 4);

    // Evict 1: should evict tok_b (oldest non-accessed), not tok_a (just searched)
    int32_t evicted = tree.evict_lru(3);
    ASSERT_EQ(evicted, 1);

    // tok_a should still be findable (it was accessed most recently)
    auto r2 = tree.search(make_key(tok_a));
    ASSERT_TRUE(r2.success);
    ASSERT_EQ(r2.matched_length, 1);
}

TEST(lru_evict_shared_prefix) {
    // Insert sequences sharing a prefix, verify eviction handles the tree structure
    llama_radix_tree tree;

    // [1, 2, 3] and [1, 2, 4] share prefix [1, 2]
    std::vector<llama_token> tok1 = {1, 2, 3};
    std::vector<llama_token> tok2 = {1, 2, 4};
    std::vector<uint32_t> cells1 = {0, 1, 2};
    std::vector<uint32_t> cells2 = {0, 1, 3};
    std::vector<uint64_t> gens = {1, 1, 1};

    tree.insert(make_key(tok1), make_value(cells1, gens));
    tree.insert(make_key(tok2), make_value(cells2, gens));

    // root -> [1,2] -> {[3], [4]} = 4 nodes
    ASSERT_EQ(tree.node_count(), 4);

    // Evict down to 3: should remove one leaf
    int32_t evicted = tree.evict_lru(3);
    ASSERT_EQ(evicted, 1);
    ASSERT_EQ(tree.node_count(), 3);

    // The shared prefix [1, 2] should still match
    std::vector<llama_token> prefix = {1, 2};
    auto r = tree.search(make_key(prefix));
    ASSERT_TRUE(r.success);
    ASSERT_GE(r.matched_length, 2);
}

TEST(lru_evict_all_leaves) {
    // Evict down to 1 (just root)
    llama_radix_tree tree;

    for (int i = 0; i < 3; i++) {
        std::vector<llama_token> tok = {(llama_token)(i + 1)};
        std::vector<uint32_t> cell = {(uint32_t)i};
        std::vector<uint64_t> gen = {1};
        tree.insert(make_key(tok), make_value(cell, gen));
    }

    ASSERT_EQ(tree.node_count(), 4); // root + 3

    int32_t evicted = tree.evict_lru(1);
    ASSERT_EQ(evicted, 3);
    ASSERT_EQ(tree.node_count(), 1); // just root
}

TEST(lru_hit_count_updates_on_search) {
    llama_radix_tree tree;

    std::vector<llama_token> tok = {1, 2, 3};
    std::vector<uint32_t> cells = {0, 1, 2};
    std::vector<uint64_t> gens = {1, 1, 1};
    tree.insert(make_key(tok), make_value(cells, gens));

    // Search 3 times
    for (int i = 0; i < 3; i++) {
        auto r = tree.search(make_key(tok));
        ASSERT_TRUE(r.success);
    }

    // The matched node should have hit_count >= 3
    auto r = tree.search(make_key(tok));
    ASSERT_TRUE(r.success);
    ASSERT_TRUE(!r.path.empty());
    // path[1] is the leaf node (path[0] is root with 0 matched tokens)
    if (r.path.size() > 1) {
        ASSERT_GE(r.path[1].first->hit_count, 3);
    }
}

// ============================================================================
// KVCacheHarness — simulates llama_kv_cache seq operations for A2/A18/A4
// ============================================================================

struct KVCell {
    bool     occupied = false;
    int32_t  pos      = -1;
    // seq_id bitmask (simplified: up to 8 sequences)
    uint8_t  seq_mask = 0;

    bool has_seq(int32_t seq) const { return seq >= 0 && seq < 8 && (seq_mask & (1 << seq)); }
    void add_seq(int32_t seq) { if (seq >= 0 && seq < 8) seq_mask |= (1 << seq); }
    void rm_seq(int32_t seq) { if (seq >= 0 && seq < 8) seq_mask &= ~(1 << seq); }
};

class KVCacheHarness {
public:
    uint32_t n_cells;
    std::vector<KVCell> cells;

    // checkpoint state
    struct Checkpoint {
        int32_t id;
        int32_t pos_end;   // max position at save time
    };
    std::vector<Checkpoint> checkpoints;
    int32_t next_checkpoint_id = 0;

    // fork state
    std::vector<int32_t> forked_branches;
    int32_t next_fork_seq_id = 1;

    explicit KVCacheHarness(uint32_t n = 64) : n_cells(n) {
        cells.resize(n);
    }

    // Simulate filling cells for a sequence
    void fill(int32_t seq_id, int32_t start_pos, int32_t count) {
        for (int32_t i = 0; i < count && (uint32_t)(start_pos + i) < n_cells; i++) {
            uint32_t idx = start_pos + i;
            cells[idx].occupied = true;
            cells[idx].pos = start_pos + i;
            cells[idx].add_seq(seq_id);
        }
    }

    int32_t seq_pos_max(int32_t seq_id) const {
        int32_t max_pos = -1;
        for (uint32_t i = 0; i < n_cells; i++) {
            if (cells[i].occupied && cells[i].has_seq(seq_id)) {
                max_pos = std::max(max_pos, cells[i].pos);
            }
        }
        return max_pos;
    }

    // --- A2: Checkpoint ---
    int32_t checkpoint_save() {
        Checkpoint cp;
        cp.id = next_checkpoint_id++;
        cp.pos_end = seq_pos_max(0);
        checkpoints.push_back(cp);
        return cp.id;
    }

    bool checkpoint_rollback(int32_t checkpoint_id) {
        // find the checkpoint
        auto it = std::find_if(checkpoints.begin(), checkpoints.end(),
            [checkpoint_id](const Checkpoint & c) { return c.id == checkpoint_id; });
        if (it == checkpoints.end()) return false;

        int32_t rollback_pos = it->pos_end;

        // remove cells with pos > rollback_pos for seq 0
        for (uint32_t i = 0; i < n_cells; i++) {
            if (cells[i].occupied && cells[i].has_seq(0) && cells[i].pos > rollback_pos) {
                cells[i].rm_seq(0);
                if (cells[i].seq_mask == 0) {
                    cells[i].occupied = false;
                    cells[i].pos = -1;
                }
            }
        }

        // erase this checkpoint and all later ones
        checkpoints.erase(it, checkpoints.end());
        return true;
    }

    // --- A18: Fork/Merge ---
    int32_t fork(int32_t parent_seq = 0) {
        int32_t new_seq = next_fork_seq_id++;
        if (new_seq >= 8) { next_fork_seq_id--; return -1; } // max 8 seqs

        // seq_cp: copy parent's cells to new_seq
        for (uint32_t i = 0; i < n_cells; i++) {
            if (cells[i].occupied && cells[i].has_seq(parent_seq)) {
                cells[i].add_seq(new_seq);
            }
        }
        forked_branches.push_back(new_seq);
        return new_seq;
    }

    bool merge(int32_t winner) {
        // remove all non-winner branches
        for (auto branch : forked_branches) {
            if (branch != winner) {
                for (uint32_t i = 0; i < n_cells; i++) {
                    if (cells[i].has_seq(branch)) {
                        cells[i].rm_seq(branch);
                        if (cells[i].seq_mask == 0) {
                            cells[i].occupied = false;
                            cells[i].pos = -1;
                        }
                    }
                }
            }
        }

        // if winner != 0, copy winner to seq 0 then remove winner
        if (winner != 0) {
            // remove seq 0
            for (uint32_t i = 0; i < n_cells; i++) {
                cells[i].rm_seq(0);
            }
            // copy winner to 0
            for (uint32_t i = 0; i < n_cells; i++) {
                if (cells[i].has_seq(winner)) {
                    cells[i].add_seq(0);
                    cells[i].rm_seq(winner);
                }
            }
        }

        forked_branches.clear();
        next_fork_seq_id = 1;
        return true;
    }

    std::vector<int32_t> get_fork_branches() const {
        return forked_branches;
    }

    // --- A4: Selective Trim ---
    int32_t selective_trim(int32_t p0, int32_t p1) {
        if (p0 < 0 || p1 <= p0) return -1;

        int32_t n_remove = p1 - p0;

        // Remove cells in [p0, p1) for seq 0
        for (uint32_t i = 0; i < n_cells; i++) {
            if (cells[i].occupied && cells[i].has_seq(0) &&
                cells[i].pos >= p0 && cells[i].pos < p1) {
                cells[i].rm_seq(0);
                if (cells[i].seq_mask == 0) {
                    cells[i].occupied = false;
                    cells[i].pos = -1;
                }
            }
        }

        // Shift positions >= p1 down by n_remove (seq_add)
        for (uint32_t i = 0; i < n_cells; i++) {
            if (cells[i].occupied && cells[i].has_seq(0) && cells[i].pos >= p1) {
                cells[i].pos -= n_remove;
            }
        }

        return n_remove;
    }

    // count occupied cells for a sequence
    int32_t count_cells(int32_t seq_id) const {
        int32_t count = 0;
        for (uint32_t i = 0; i < n_cells; i++) {
            if (cells[i].occupied && cells[i].has_seq(seq_id)) count++;
        }
        return count;
    }
};

// ============================================================================
// A2: Checkpoint / Rollback Tests
// ============================================================================

TEST(checkpoint_save_returns_incrementing_ids) {
    KVCacheHarness kv;
    kv.fill(0, 0, 10);

    int32_t id1 = kv.checkpoint_save();
    int32_t id2 = kv.checkpoint_save();
    int32_t id3 = kv.checkpoint_save();

    ASSERT_EQ(id1, 0);
    ASSERT_EQ(id2, 1);
    ASSERT_EQ(id3, 2);
}

TEST(checkpoint_rollback_removes_tokens_after_checkpoint) {
    KVCacheHarness kv;
    // Fill 10 tokens at positions 0..9
    kv.fill(0, 0, 10);
    ASSERT_EQ(kv.seq_pos_max(0), 9);

    // Save checkpoint at pos 9
    int32_t cp = kv.checkpoint_save();

    // Add 5 more tokens
    kv.fill(0, 10, 5);
    ASSERT_EQ(kv.seq_pos_max(0), 14);
    ASSERT_EQ(kv.count_cells(0), 15);

    // Rollback — should remove tokens at pos 10..14
    bool ok = kv.checkpoint_rollback(cp);
    ASSERT_TRUE(ok);
    ASSERT_EQ(kv.seq_pos_max(0), 9);
    ASSERT_EQ(kv.count_cells(0), 10);
}

TEST(checkpoint_rollback_invalid_id) {
    KVCacheHarness kv;
    kv.fill(0, 0, 5);

    bool ok = kv.checkpoint_rollback(999);
    ASSERT_FALSE(ok);
}

TEST(checkpoint_rollback_erases_later_checkpoints) {
    KVCacheHarness kv;
    kv.fill(0, 0, 5);
    int32_t cp1 = kv.checkpoint_save();

    kv.fill(0, 5, 5);
    int32_t cp2 = kv.checkpoint_save();
    (void)cp2;

    kv.fill(0, 10, 5);

    // Rollback to cp1 — cp2 should also be erased
    bool ok = kv.checkpoint_rollback(cp1);
    ASSERT_TRUE(ok);
    ASSERT_EQ(kv.seq_pos_max(0), 4);

    // cp2 should no longer be valid
    bool ok2 = kv.checkpoint_rollback(cp2);
    ASSERT_FALSE(ok2);
}

TEST(checkpoint_multiple_save_rollback_cycles) {
    KVCacheHarness kv;

    // Round 1: fill, save, extend, rollback
    kv.fill(0, 0, 5);
    int32_t cp1 = kv.checkpoint_save();
    kv.fill(0, 5, 10);
    ASSERT_EQ(kv.count_cells(0), 15);
    kv.checkpoint_rollback(cp1);
    ASSERT_EQ(kv.count_cells(0), 5);

    // Round 2: extend again, save, extend, rollback
    kv.fill(0, 5, 3);
    int32_t cp2 = kv.checkpoint_save();
    kv.fill(0, 8, 7);
    ASSERT_EQ(kv.count_cells(0), 15);
    kv.checkpoint_rollback(cp2);
    ASSERT_EQ(kv.count_cells(0), 8);
    ASSERT_EQ(kv.seq_pos_max(0), 7);
}

// ============================================================================
// A18: Fork / Merge Tests
// ============================================================================

TEST(fork_creates_new_sequence) {
    KVCacheHarness kv;
    kv.fill(0, 0, 10);

    int32_t branch = kv.fork(0);
    ASSERT_TRUE(branch > 0);

    // New branch should have same cells as parent
    ASSERT_EQ(kv.count_cells(branch), 10);
    ASSERT_EQ(kv.seq_pos_max(branch), 9);
}

TEST(fork_independent_extension) {
    KVCacheHarness kv;
    kv.fill(0, 0, 5); // shared prefix: pos 0..4

    int32_t b1 = kv.fork(0);
    int32_t b2 = kv.fork(0);

    // Extend branches independently (using different cells)
    kv.fill(b1, 5, 3); // b1 gets pos 5,6,7
    kv.fill(b2, 5, 2); // b2 gets pos 5,6

    ASSERT_EQ(kv.count_cells(0), 5);  // parent unchanged
    // branches share parent cells + have their own
    ASSERT_EQ(kv.seq_pos_max(b1), 7);
    ASSERT_EQ(kv.seq_pos_max(b2), 6);
}

TEST(fork_merge_keeps_winner) {
    KVCacheHarness kv;
    kv.fill(0, 0, 5);

    int32_t b1 = kv.fork(0);
    int32_t b2 = kv.fork(0);

    // Extend b1
    kv.fill(b1, 5, 3);

    // Merge: b1 wins
    bool ok = kv.merge(b1);
    ASSERT_TRUE(ok);

    // After merge, seq 0 should have b1's content
    ASSERT_EQ(kv.seq_pos_max(0), 7);

    // b2 should be cleaned up
    ASSERT_EQ(kv.count_cells(b2), 0);
}

TEST(fork_merge_winner_is_zero) {
    KVCacheHarness kv;
    kv.fill(0, 0, 5);

    int32_t b1 = kv.fork(0);
    kv.fill(b1, 5, 3);

    // Merge with winner = 0 (keep original)
    bool ok = kv.merge(0);
    ASSERT_TRUE(ok);

    // seq 0 should still have original content
    ASSERT_EQ(kv.seq_pos_max(0), 4);
    ASSERT_EQ(kv.count_cells(0), 5);
}

TEST(fork_max_sequences) {
    KVCacheHarness kv;
    kv.fill(0, 0, 5);

    // Fork up to the limit (8 sequences total, seq 0 is used)
    std::vector<int32_t> branches;
    for (int i = 0; i < 10; i++) {
        int32_t b = kv.fork(0);
        if (b >= 0) branches.push_back(b);
    }

    // Should have at most 7 branches (seqs 1..7)
    ASSERT_LE((int32_t)branches.size(), 7);
}

TEST(fork_get_branches) {
    KVCacheHarness kv;
    kv.fill(0, 0, 5);

    kv.fork(0);
    kv.fork(0);
    kv.fork(0);

    auto branches = kv.get_fork_branches();
    ASSERT_EQ((int32_t)branches.size(), 3);
}

// ============================================================================
// A4: Selective Trim Tests
// ============================================================================

TEST(selective_trim_basic) {
    KVCacheHarness kv;
    // Fill 20 tokens at pos 0..19
    kv.fill(0, 0, 20);
    ASSERT_EQ(kv.count_cells(0), 20);

    // Trim pos [5, 10): removes 5 tokens, shifts 10..19 -> 5..14
    int32_t removed = kv.selective_trim(5, 10);
    ASSERT_EQ(removed, 5);
    ASSERT_EQ(kv.count_cells(0), 15);
    ASSERT_EQ(kv.seq_pos_max(0), 14);
}

TEST(selective_trim_invalid_range) {
    KVCacheHarness kv;
    kv.fill(0, 0, 10);

    // p0 >= p1 should fail
    ASSERT_EQ(kv.selective_trim(5, 5), -1);
    ASSERT_EQ(kv.selective_trim(5, 3), -1);
    ASSERT_EQ(kv.selective_trim(-1, 5), -1);
}

TEST(selective_trim_beginning) {
    KVCacheHarness kv;
    kv.fill(0, 0, 10);

    // Trim pos [0, 3): removes first 3, shifts rest down by 3
    int32_t removed = kv.selective_trim(0, 3);
    ASSERT_EQ(removed, 3);
    ASSERT_EQ(kv.count_cells(0), 7);
    ASSERT_EQ(kv.seq_pos_max(0), 6);
}

TEST(selective_trim_end) {
    KVCacheHarness kv;
    kv.fill(0, 0, 10);

    // Trim pos [7, 10): removes last 3
    int32_t removed = kv.selective_trim(7, 10);
    ASSERT_EQ(removed, 3);
    ASSERT_EQ(kv.count_cells(0), 7);
    ASSERT_EQ(kv.seq_pos_max(0), 6);
}

TEST(selective_trim_preserves_gap_closure) {
    KVCacheHarness kv;
    // Fill 10 tokens: positions 0..9
    kv.fill(0, 0, 10);

    // Trim middle [3, 6) — removes pos 3,4,5 and shifts 6..9 -> 3..6
    int32_t removed = kv.selective_trim(3, 6);
    ASSERT_EQ(removed, 3);

    // Check position continuity: max should be 6 (was 9, removed 3)
    ASSERT_EQ(kv.seq_pos_max(0), 6);

    // Verify there are no gaps: positions should be 0,1,2,3,4,5,6
    std::vector<int32_t> positions;
    for (uint32_t i = 0; i < kv.n_cells; i++) {
        if (kv.cells[i].occupied && kv.cells[i].has_seq(0)) {
            positions.push_back(kv.cells[i].pos);
        }
    }
    std::sort(positions.begin(), positions.end());
    ASSERT_EQ((int32_t)positions.size(), 7);
    for (int i = 0; i < 7; i++) {
        ASSERT_EQ(positions[i], i);
    }
}

// ============================================================================
// Combined scenario tests
// ============================================================================

TEST(checkpoint_then_fork_rollback) {
    // Simulate: agent generates, checkpoints, forks for retry, rolls back
    KVCacheHarness kv;

    // System prompt + user query
    kv.fill(0, 0, 20);

    // Checkpoint before agent response
    int32_t cp = kv.checkpoint_save();

    // Agent generates response (10 tokens)
    kv.fill(0, 20, 10);
    ASSERT_EQ(kv.count_cells(0), 30);

    // Response was bad — rollback
    kv.checkpoint_rollback(cp);
    ASSERT_EQ(kv.count_cells(0), 20);
    ASSERT_EQ(kv.seq_pos_max(0), 19);
}

TEST(fork_with_selective_trim) {
    // Simulate: fork for parallel tool calls, trim middle context from winner
    KVCacheHarness kv;
    kv.fill(0, 0, 20); // shared context

    int32_t b1 = kv.fork(0);
    kv.fill(b1, 20, 5); // branch 1 tool result

    // Merge b1 as winner
    kv.merge(b1);
    ASSERT_EQ(kv.seq_pos_max(0), 24);

    // Now trim some middle context (e.g., old tool output at pos [5, 10))
    int32_t removed = kv.selective_trim(5, 10);
    ASSERT_EQ(removed, 5);
    ASSERT_EQ(kv.count_cells(0), 20); // was 25, removed 5
}

TEST(lru_evict_after_multiple_inserts_and_searches) {
    // Simulate a realistic pattern: many prefixes inserted, some searched,
    // eviction should keep the searched ones
    llama_radix_tree tree;

    // Insert 10 different sequences
    for (int i = 0; i < 10; i++) {
        std::vector<llama_token> tok = {(llama_token)(i * 10 + 1), (llama_token)(i * 10 + 2)};
        std::vector<uint32_t> cells = {(uint32_t)(i * 2), (uint32_t)(i * 2 + 1)};
        std::vector<uint64_t> gens = {1, 1};
        tree.insert(make_key(tok), make_value(cells, gens));
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // root + 10 leaves = 11
    ASSERT_EQ(tree.node_count(), 11);

    // Search for sequences 7, 8, 9 to bump their access time
    for (int i = 7; i < 10; i++) {
        std::vector<llama_token> tok = {(llama_token)(i * 10 + 1), (llama_token)(i * 10 + 2)};
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        auto r = tree.search(make_key(tok));
        ASSERT_TRUE(r.success);
    }

    // Evict down to 4 (root + 3 recently searched)
    int32_t evicted = tree.evict_lru(4);
    ASSERT_EQ(evicted, 7);
    ASSERT_EQ(tree.node_count(), 4);

    // The 3 recently searched should survive
    for (int i = 7; i < 10; i++) {
        std::vector<llama_token> tok = {(llama_token)(i * 10 + 1), (llama_token)(i * 10 + 2)};
        auto r = tree.search(make_key(tok));
        ASSERT_TRUE(r.success);
        ASSERT_EQ(r.matched_length, 2);
    }
}

// ============================================================================
// A3: Multi-turn append (clear with data=false preserves tree)
// ============================================================================

TEST(tree_clear_and_reinsert) {
    // Simulates A3: clear tree state, then re-insert with same tokens
    llama_radix_tree tree;

    std::vector<llama_token> tok = {1, 2, 3, 4, 5};
    std::vector<uint32_t> cells = {0, 1, 2, 3, 4};
    std::vector<uint64_t> gens = {1, 1, 1, 1, 1};
    tree.insert(make_key(tok), make_value(cells, gens));

    auto r1 = tree.search(make_key(tok));
    ASSERT_TRUE(r1.success);
    ASSERT_EQ(r1.matched_length, 5);

    // Clear and re-insert (simulates clear(data=false) keeping tree,
    // but here we test the tree's ability to handle re-insertion)
    tree.clear();
    ASSERT_EQ(tree.node_count(), 1); // just root after clear

    // Re-insert with new generations
    std::vector<uint64_t> gens2 = {2, 2, 2, 2, 2};
    tree.insert(make_key(tok), make_value(cells, gens2));

    auto r2 = tree.search(make_key(tok));
    ASSERT_TRUE(r2.success);
    ASSERT_EQ(r2.matched_length, 5);
    // Generations should be the new ones
    for (int i = 0; i < 5; i++) {
        ASSERT_EQ(r2.cell_generations[i], (uint64_t)2);
    }
}

// ============================================================================
// Tree sync: seq_add/seq_div invalidation + re-promote after trim
// ============================================================================

// Extended harness with radix tree to test tree synchronization
class TreeSyncHarness {
public:
    uint32_t n_cells;
    std::vector<uint64_t> cell_generations;
    std::vector<KVCell>   cells;
    llama_radix_tree      tree;

    explicit TreeSyncHarness(uint32_t n = 64) : n_cells(n) {
        cell_generations.resize(n, 1);
        cells.resize(n);
    }

    void fill(int32_t seq_id, int32_t start_pos, int32_t count) {
        for (int32_t i = 0; i < count && (uint32_t)(start_pos + i) < n_cells; i++) {
            uint32_t idx = start_pos + i;
            cells[idx].occupied = true;
            cells[idx].pos = start_pos + i;
            cells[idx].add_seq(seq_id);
        }
    }

    // Promote a token sequence into the tree
    void promote(const std::vector<llama_token> & tokens,
                 const std::vector<uint32_t> & cell_idx) {
        std::vector<uint64_t> gens(cell_idx.size());
        for (size_t i = 0; i < cell_idx.size(); i++) {
            gens[i] = (cell_idx[i] < n_cells) ? cell_generations[cell_idx[i]] : 0;
        }
        tree.insert(make_key(tokens), make_value(cell_idx, gens));
    }

    // Find prefix match
    int32_t find(const std::vector<llama_token> & tokens, std::vector<uint32_t> & out_cells) {
        out_cells.clear();
        auto result = tree.search(make_key(tokens));
        if (!result.success) return 0;

        int32_t valid = 0;
        for (int32_t i = 0; i < result.matched_length; i++) {
            uint32_t idx = result.cell_indices[i];
            uint64_t gen = result.cell_generations[i];
            if (gen == 0) break;
            if (idx < n_cells && cell_generations[idx] == gen) {
                out_cells.push_back(idx);
                valid++;
            } else {
                break;
            }
        }
        return valid;
    }

    // Simulate seq_rm: free cells in [p0, p1), bump generation, invalidate tree
    void seq_rm(int32_t seq_id, int32_t p0, int32_t p1) {
        for (uint32_t i = 0; i < n_cells; i++) {
            if (!cells[i].occupied || !cells[i].has_seq(seq_id)) continue;
            if (cells[i].pos < p0 || cells[i].pos >= p1) continue;

            cells[i].rm_seq(seq_id);
            if (cells[i].seq_mask == 0) {
                cells[i].occupied = false;
                cells[i].pos = -1;
            }
            // bump gen + invalidate tree (same as real seq_rm)
            cell_generations[i]++;
            tree.invalidate_cell(i);
        }
    }

    // Simulate seq_add: shift positions, bump generation + invalidate tree
    void seq_add(int32_t seq_id, int32_t p0, int32_t shift) {
        for (uint32_t i = 0; i < n_cells; i++) {
            if (!cells[i].occupied || !cells[i].has_seq(seq_id)) continue;
            if (cells[i].pos < p0) continue;

            // invalidate before shifting (same as updated seq_add)
            cell_generations[i]++;
            tree.invalidate_cell(i);

            cells[i].pos += shift;
        }
    }

    // selective_trim with re-promote
    int32_t trim_and_repromote(int32_t p0, int32_t p1,
                               const std::vector<llama_token> & remaining_tokens,
                               const std::vector<uint32_t> & remaining_cells) {
        int32_t n_remove = p1 - p0;

        // step 1: seq_rm [p0, p1)
        seq_rm(0, p0, p1);

        // step 2: seq_add shift
        seq_add(0, p1, -n_remove);

        // step 3: re-promote
        if (!remaining_tokens.empty()) {
            promote(remaining_tokens, remaining_cells);
        }

        return n_remove;
    }
};

TEST(tree_sync_seq_add_invalidates_old_entry) {
    // After seq_add shifts positions, the old tree entry should be invalidated
    TreeSyncHarness h;

    // tokens [A, B, C, D, E] at cells [0, 1, 2, 3, 4]
    std::vector<llama_token> tokens = {10, 20, 30, 40, 50};
    std::vector<uint32_t>    cells  = {0, 1, 2, 3, 4};
    h.fill(0, 0, 5);
    h.promote(tokens, cells);

    // Verify initial find works
    std::vector<uint32_t> out;
    ASSERT_EQ(h.find(tokens, out), 5);

    // seq_add: shift positions [3, ∞) down by 2 (simulating gap closure)
    h.seq_add(0, 3, -2);

    // The old entry [A,B,C,D,E] should now fail for cells 3,4 (invalidated)
    int32_t matched = h.find(tokens, out);
    // cells 0,1,2 are untouched, cells 3,4 were invalidated
    ASSERT_LE(matched, 3);
}

TEST(tree_sync_trim_without_repromote_loses_suffix) {
    // Without re-promote, the post-trim sequence can't be found
    TreeSyncHarness h;

    std::vector<llama_token> tokens = {10, 20, 30, 40, 50};
    std::vector<uint32_t>    cells  = {0, 1, 2, 3, 4};
    h.fill(0, 0, 5);
    h.promote(tokens, cells);

    // Trim [1, 3) — removes tokens 20, 30
    h.seq_rm(0, 1, 3);
    h.seq_add(0, 3, -2);

    // Post-trim sequence is [10, 40, 50], but tree has no such entry
    std::vector<llama_token> post_trim = {10, 40, 50};
    std::vector<uint32_t> out;
    int32_t matched = h.find(post_trim, out);
    // Should find nothing (or at most partial) because we didn't re-promote
    ASSERT_LE(matched, 1); // only token 10 at cell 0 might match (cell 0 untouched)
}

TEST(tree_sync_trim_with_repromote_recovers) {
    // With re-promote after trim, the post-trim sequence IS findable
    TreeSyncHarness h;

    std::vector<llama_token> tokens = {10, 20, 30, 40, 50};
    std::vector<uint32_t>    cells  = {0, 1, 2, 3, 4};
    h.fill(0, 0, 5);
    h.promote(tokens, cells);

    // Trim [1, 3) and re-promote surviving sequence
    // Surviving: token 10 at cell 0, token 40 at cell 3 (now pos 1), token 50 at cell 4 (now pos 2)
    std::vector<llama_token> remaining_tokens = {10, 40, 50};
    std::vector<uint32_t>    remaining_cells  = {0, 3, 4};
    h.trim_and_repromote(1, 3, remaining_tokens, remaining_cells);

    // Now search for [10, 40, 50] — should find all 3
    std::vector<uint32_t> out;
    int32_t matched = h.find(remaining_tokens, out);
    ASSERT_EQ(matched, 3);
    ASSERT_EQ(out[0], (uint32_t)0);
    ASSERT_EQ(out[1], (uint32_t)3);
    ASSERT_EQ(out[2], (uint32_t)4);
}

TEST(tree_sync_trim_prefix_still_matches_after_repromote) {
    // After trim + re-promote, searching a prefix of the remaining tokens works
    TreeSyncHarness h;

    std::vector<llama_token> tokens = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<uint32_t>    cells  = {0, 1, 2, 3, 4, 5, 6, 7};
    h.fill(0, 0, 8);
    h.promote(tokens, cells);

    // Trim [2, 5) — remove tokens 3,4,5 → remaining [1,2,6,7,8]
    std::vector<llama_token> remaining = {1, 2, 6, 7, 8};
    std::vector<uint32_t>    rem_cells = {0, 1, 5, 6, 7};
    h.trim_and_repromote(2, 5, remaining, rem_cells);

    // Search prefix [1, 2, 6]
    std::vector<llama_token> prefix = {1, 2, 6};
    std::vector<uint32_t> out;
    int32_t matched = h.find(prefix, out);
    ASSERT_EQ(matched, 3);
}

TEST(tree_sync_multiple_trims) {
    // Multiple consecutive trims with re-promote
    TreeSyncHarness h;

    std::vector<llama_token> tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<uint32_t>    cells;
    for (int i = 0; i < 10; i++) cells.push_back(i);
    h.fill(0, 0, 10);
    h.promote(tokens, cells);

    // First trim: remove [2, 4) → [1,2,5,6,7,8,9,10]
    std::vector<llama_token> rem1 = {1, 2, 5, 6, 7, 8, 9, 10};
    std::vector<uint32_t>    cells1 = {0, 1, 4, 5, 6, 7, 8, 9};
    h.trim_and_repromote(2, 4, rem1, cells1);

    std::vector<uint32_t> out;
    ASSERT_EQ(h.find(rem1, out), 8);

    // Second trim: remove [3, 6) from the new sequence → [1,2,5,9,10]
    // New positions after first trim: 1@0, 2@1, 5@2, 6@3, 7@4, 8@5, 9@6, 10@7
    // Trim [3,6) removes pos 3,4,5 (tokens 6,7,8) → remaining [1,2,5,9,10]
    std::vector<llama_token> rem2 = {1, 2, 5, 9, 10};
    std::vector<uint32_t>    cells2 = {0, 1, 4, 8, 9};
    h.trim_and_repromote(3, 6, rem2, cells2);

    ASSERT_EQ(h.find(rem2, out), 5);
}

// ============================================================================
// main
// ============================================================================

int main() {
    printf("=== Phase 4: Feature Tests (A2/A18/A4/A6 + Tree Sync) ===\n");
    // Tests are auto-registered via static constructors
    printf("\nResults: %d passed, %d failed\n", n_pass, n_fail);
    return n_fail ? 1 : 0;
}
