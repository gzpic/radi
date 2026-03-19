// Phase 3: Integration tests for prefix cache logic
//
// llama_kv_cache requires a full model to construct, which is too heavy for
// unit tests. Instead we use a PrefixCacheHarness that replicates the exact
// same promote/find/invalidate logic from llama_kv_cache, operating on the
// same llama_radix_tree. This validates the integration patterns without
// needing GPU tensors or model files.

#include "llama-radix-tree.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
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

// ============================================================================
// PrefixCacheHarness
//
// Replicates the exact prefix cache logic from llama_kv_cache:
//   - cell_generations[] with bump on overwrite
//   - promote: snapshot generations at insert time
//   - find: validate generations, stop at first mismatch
//   - reclaim: restore cell metadata if cell is empty
//   - invalidate: bump generation + tree.invalidate_cell()
// ============================================================================

struct CellState {
    bool     occupied = false;
    int32_t  seq_id   = -1;
    int32_t  pos      = -1;
};

class PrefixCacheHarness {
public:
    uint32_t                        n_cells;
    std::vector<uint64_t>           cell_generations;
    std::vector<CellState>          cells;
    std::unique_ptr<llama_radix_tree> prefix_tree;

    explicit PrefixCacheHarness(uint32_t n = 128) : n_cells(n) {
        cell_generations.resize(n, 1); // start at 1, 0 = invalid
        cells.resize(n);
        prefix_tree = std::make_unique<llama_radix_tree>();
    }

    // --- promote: same logic as llama_kv_cache::prefix_cache_promote ---
    void promote(const std::vector<llama_token> & tokens,
                 const std::vector<uint32_t> & cell_indices,
                 int64_t extra_key = 0) {
        if (tokens.empty() || tokens.size() != cell_indices.size()) return;

        std::vector<uint64_t> gens(cell_indices.size());
        for (size_t i = 0; i < cell_indices.size(); i++) {
            uint32_t idx = cell_indices[i];
            gens[i] = (idx < n_cells) ? cell_generations[idx] : 0;
        }

        llama_radix_node_key key(llama_vector_view<llama_token>(tokens), extra_key);
        llama_radix_node_value value;
        value.cell_indices     = llama_vector_view<uint32_t>(cell_indices);
        value.cell_generations = llama_vector_view<uint64_t>(gens);
        prefix_tree->insert(key, value);
    }

    // --- find: same logic as llama_kv_cache::prefix_cache_find ---
    int32_t find(const std::vector<llama_token> & tokens,
                 std::vector<uint32_t> & out_cells,
                 int64_t extra_key = 0) {
        out_cells.clear();

        llama_radix_node_key key(llama_vector_view<llama_token>(tokens), extra_key);
        auto result = prefix_tree->search(key);

        if (!result.success || result.matched_length == 0) return 0;

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

    // --- reclaim: same logic as llama_kv_cache::prefix_cache_reclaim ---
    int32_t reclaim(int32_t seq_id,
                    const std::vector<llama_token> & tokens,
                    const std::vector<int32_t> & positions) {
        std::vector<uint32_t> cached_cells;
        int32_t matched = find(tokens, cached_cells);
        if (matched <= 0) return 0;

        int32_t reclaimed = 0;
        for (int32_t i = 0; i < matched; i++) {
            uint32_t idx = cached_cells[i];
            if (idx >= n_cells) break;

            if (cells[idx].occupied) {
                if (cells[idx].seq_id == seq_id && cells[idx].pos == positions[i]) {
                    reclaimed++;
                    continue;
                }
                break; // occupied by something else
            }

            // restore metadata
            cells[idx].occupied = true;
            cells[idx].seq_id   = seq_id;
            cells[idx].pos      = positions[i];
            reclaimed++;
        }
        return reclaimed;
    }

    // --- overwrite a cell: simulates apply_ubatch writing new KV data ---
    void overwrite_cell(uint32_t idx, int32_t seq_id, int32_t pos) {
        if (idx >= n_cells) return;

        // bump generation + invalidate tree entry (same as apply_ubatch)
        cell_generations[idx]++;
        prefix_tree->invalidate_cell(idx);

        cells[idx].occupied = true;
        cells[idx].seq_id   = seq_id;
        cells[idx].pos      = pos;
    }

    // --- free a cell: simulates seq_rm freeing a cell ---
    void free_cell(uint32_t idx) {
        if (idx >= n_cells) return;

        cell_generations[idx]++;
        prefix_tree->invalidate_cell(idx);

        cells[idx].occupied = false;
        cells[idx].seq_id   = -1;
        cells[idx].pos      = -1;
    }

    void clear_tree() {
        prefix_tree->clear();
    }
};

// ============================================================================
// Phase 3 Tests
// ============================================================================

TEST(harness_promote_then_find) {
    PrefixCacheHarness h(64);

    std::vector<llama_token> tokens = {100, 200, 300, 400, 500};
    std::vector<uint32_t>    cells  = {0, 1, 2, 3, 4};

    // Mark cells as occupied (simulating apply_ubatch wrote KV data)
    for (int i = 0; i < 5; i++) {
        h.cells[i].occupied = true;
        h.cells[i].seq_id = 0;
        h.cells[i].pos = i;
    }

    h.promote(tokens, cells);

    std::vector<uint32_t> out;
    int32_t matched = h.find(tokens, out);

    ASSERT_EQ(matched, 5);
    ASSERT_EQ((int)out.size(), 5);
    for (int i = 0; i < 5; i++) {
        ASSERT_EQ(out[i], (uint32_t)i);
    }
}

TEST(harness_find_prefix_of_longer_query) {
    PrefixCacheHarness h(64);

    std::vector<llama_token> tokens3 = {10, 20, 30};
    std::vector<uint32_t>    cells3  = {5, 6, 7};
    h.promote(tokens3, cells3);

    // Search for a longer sequence
    std::vector<llama_token> tokens5 = {10, 20, 30, 40, 50};
    std::vector<uint32_t> out;
    int32_t matched = h.find(tokens5, out);

    ASSERT_EQ(matched, 3);
    ASSERT_EQ(out[0], 5u);
    ASSERT_EQ(out[1], 6u);
    ASSERT_EQ(out[2], 7u);
}

TEST(harness_generation_invalidation) {
    PrefixCacheHarness h(64);

    std::vector<llama_token> tokens = {1, 2, 3};
    std::vector<uint32_t>    cells  = {10, 11, 12};
    h.promote(tokens, cells);

    // Verify it's findable
    std::vector<uint32_t> out;
    ASSERT_EQ(h.find(tokens, out), 3);

    // Overwrite cell 11 (simulates new data written to that cell)
    h.overwrite_cell(11, 99, 99);

    // Now find should stop at the first invalid cell
    out.clear();
    int32_t matched = h.find(tokens, out);
    // cell 10 is still valid (gen unchanged), cell 11 gen was bumped → stop
    ASSERT_EQ(matched, 1);
    ASSERT_EQ(out[0], 10u);
}

TEST(harness_generation_invalidation_first_cell) {
    PrefixCacheHarness h(64);

    std::vector<llama_token> tokens = {1, 2, 3};
    std::vector<uint32_t>    cells  = {10, 11, 12};
    h.promote(tokens, cells);

    // Overwrite the FIRST cell
    h.overwrite_cell(10, 99, 99);

    std::vector<uint32_t> out;
    int32_t matched = h.find(tokens, out);
    // First cell invalid → matched = 0
    ASSERT_EQ(matched, 0);
}

TEST(harness_reclaim_empty_cells) {
    PrefixCacheHarness h(64);

    std::vector<llama_token> tokens = {1, 2, 3};
    std::vector<uint32_t>    cells  = {10, 11, 12};
    h.promote(tokens, cells);

    // Cells are empty (not occupied) → reclaim should restore them
    std::vector<int32_t> positions = {0, 1, 2};
    int32_t reclaimed = h.reclaim(/*seq_id=*/0, tokens, positions);

    ASSERT_EQ(reclaimed, 3);
    ASSERT_TRUE(h.cells[10].occupied);
    ASSERT_EQ(h.cells[10].seq_id, 0);
    ASSERT_EQ(h.cells[10].pos, 0);
    ASSERT_TRUE(h.cells[11].occupied);
    ASSERT_EQ(h.cells[11].pos, 1);
    ASSERT_TRUE(h.cells[12].occupied);
    ASSERT_EQ(h.cells[12].pos, 2);
}

TEST(harness_reclaim_occupied_by_other) {
    PrefixCacheHarness h(64);

    std::vector<llama_token> tokens = {1, 2, 3};
    std::vector<uint32_t>    cells  = {10, 11, 12};
    h.promote(tokens, cells);

    // Cell 11 is occupied by another sequence
    h.cells[11].occupied = true;
    h.cells[11].seq_id = 99;
    h.cells[11].pos = 77;

    std::vector<int32_t> positions = {0, 1, 2};
    int32_t reclaimed = h.reclaim(/*seq_id=*/0, tokens, positions);

    // Cell 10 reclaimed, cell 11 occupied by other → stop
    ASSERT_EQ(reclaimed, 1);
    ASSERT_TRUE(h.cells[10].occupied);
    ASSERT_EQ(h.cells[10].seq_id, 0);
    // Cell 11 should be unchanged
    ASSERT_EQ(h.cells[11].seq_id, 99);
}

TEST(harness_reclaim_already_correct) {
    PrefixCacheHarness h(64);

    std::vector<llama_token> tokens = {1, 2, 3};
    std::vector<uint32_t>    cells  = {10, 11, 12};
    h.promote(tokens, cells);

    // Cell 10 already has the right seq_id and pos
    h.cells[10].occupied = true;
    h.cells[10].seq_id = 0;
    h.cells[10].pos = 0;

    std::vector<int32_t> positions = {0, 1, 2};
    int32_t reclaimed = h.reclaim(/*seq_id=*/0, tokens, positions);

    // Cell 10 already correct, cells 11-12 reclaimed → 3 total
    ASSERT_EQ(reclaimed, 3);
}

TEST(harness_seq_rm_invalidates_tree) {
    PrefixCacheHarness h(64);

    std::vector<llama_token> tokens = {1, 2, 3, 4, 5};
    std::vector<uint32_t>    cells  = {0, 1, 2, 3, 4};
    h.promote(tokens, cells);

    // Free cells 2 and 3 (simulating seq_rm removing positions 2-3)
    h.free_cell(2);
    h.free_cell(3);

    std::vector<uint32_t> out;
    int32_t matched = h.find(tokens, out);

    // Cells 0,1 still valid. Cell 2 was freed → gen bumped → stop at 2
    ASSERT_EQ(matched, 2);
    ASSERT_EQ(out[0], 0u);
    ASSERT_EQ(out[1], 1u);
}

TEST(harness_overwrite_then_repromote) {
    PrefixCacheHarness h(64);

    std::vector<llama_token> tokens = {1, 2, 3};
    std::vector<uint32_t>    cells  = {10, 11, 12};
    h.promote(tokens, cells);

    // Overwrite cell 11 (invalidates old entry)
    h.overwrite_cell(11, 5, 5);

    // Old entry is now partially invalid
    std::vector<uint32_t> out;
    ASSERT_EQ(h.find(tokens, out), 1); // only cell 10 valid

    // Re-promote with new cells (e.g. recomputed)
    std::vector<uint32_t> new_cells = {10, 20, 21};
    h.promote(tokens, new_cells);

    out.clear();
    int32_t matched = h.find(tokens, out);
    ASSERT_EQ(matched, 3);
    ASSERT_EQ(out[0], 10u);
    ASSERT_EQ(out[1], 20u);
    ASSERT_EQ(out[2], 21u);
}

TEST(harness_promote_find_with_extra_key) {
    PrefixCacheHarness h(64);

    std::vector<llama_token> tokens = {1, 2, 3};
    h.promote(tokens, {10, 11, 12}, /*extra_key=*/0);
    h.promote(tokens, {20, 21, 22}, /*extra_key=*/42);

    std::vector<uint32_t> out;
    ASSERT_EQ(h.find(tokens, out, 0), 3);
    ASSERT_EQ(out[0], 10u);

    out.clear();
    ASSERT_EQ(h.find(tokens, out, 42), 3);
    ASSERT_EQ(out[0], 20u);

    out.clear();
    ASSERT_EQ(h.find(tokens, out, 99), 0); // no match
}

TEST(harness_reclaim_after_generation_invalidation) {
    PrefixCacheHarness h(64);

    std::vector<llama_token> tokens = {1, 2, 3};
    std::vector<uint32_t>    cells  = {10, 11, 12};
    h.promote(tokens, cells);

    // Overwrite cell 12 → invalidates the 3rd position
    h.overwrite_cell(12, 77, 77);

    // Reclaim should only recover 2 cells (10, 11 valid; 12 gen mismatch)
    std::vector<int32_t> positions = {0, 1, 2};
    int32_t reclaimed = h.reclaim(0, tokens, positions);
    ASSERT_EQ(reclaimed, 2);
    ASSERT_TRUE(h.cells[10].occupied);
    ASSERT_TRUE(h.cells[11].occupied);
    // Cell 12 not reclaimed (it was overwritten)
}

TEST(harness_full_cycle_promote_free_repromote) {
    PrefixCacheHarness h(64);

    // 1. First computation: tokens [1,2,3,4,5] → cells [0..4]
    std::vector<llama_token> tokens = {1, 2, 3, 4, 5};
    std::vector<uint32_t>    cells  = {0, 1, 2, 3, 4};
    for (int i = 0; i < 5; i++) {
        h.cells[i].occupied = true;
        h.cells[i].seq_id = 0;
        h.cells[i].pos = i;
    }
    h.promote(tokens, cells);

    // 2. Free all cells (simulating context clear / seq_rm)
    for (int i = 0; i < 5; i++) {
        h.free_cell(i);
    }

    // 3. Tree entries are now all invalid
    std::vector<uint32_t> out;
    ASSERT_EQ(h.find(tokens, out), 0);

    // 4. Re-compute and re-promote
    std::vector<uint32_t> new_cells = {10, 11, 12, 13, 14};
    for (int i = 0; i < 5; i++) {
        h.cells[new_cells[i]].occupied = true;
        h.cells[new_cells[i]].seq_id = 0;
        h.cells[new_cells[i]].pos = i;
    }
    h.promote(tokens, new_cells);

    // 5. Should find the new cells
    out.clear();
    int32_t matched = h.find(tokens, out);
    ASSERT_EQ(matched, 5);
    ASSERT_EQ(out[0], 10u);
    ASSERT_EQ(out[4], 14u);
}

TEST(harness_multiple_sequences_sharing_prefix) {
    PrefixCacheHarness h(128);

    // Simulate: two requests share the same system prompt [1,2,3]
    // Request A: [1,2,3, 10,11]  → cells [0,1,2, 3,4]
    // Request B: [1,2,3, 20,21]  → cells [0,1,2, 5,6]  (reuses prefix cells)

    std::vector<llama_token> seq_a = {1, 2, 3, 10, 11};
    std::vector<uint32_t>  cells_a = {0, 1, 2, 3, 4};
    for (auto c : cells_a) {
        h.cells[c].occupied = true;
        h.cells[c].seq_id = 0;
    }
    h.promote(seq_a, cells_a);

    // Now request B comes, shares prefix [1,2,3]
    std::vector<llama_token> seq_b = {1, 2, 3, 20, 21};
    std::vector<uint32_t> out;
    int32_t matched = h.find(seq_b, out);

    // Should match 3 tokens of shared prefix
    ASSERT_EQ(matched, 3);
    ASSERT_EQ(out[0], 0u); // same cells as seq_a's prefix
    ASSERT_EQ(out[1], 1u);
    ASSERT_EQ(out[2], 2u);
}

TEST(harness_reclaim_then_free_rollback) {
    // Simulates the prepare() dry-run pattern:
    // reclaim → (do stuff) → rollback by freeing reclaimed cells
    PrefixCacheHarness h(64);

    std::vector<llama_token> tokens = {1, 2, 3};
    std::vector<uint32_t>    cells  = {10, 11, 12};
    h.promote(tokens, cells);

    // Reclaim
    std::vector<int32_t> positions = {0, 1, 2};
    int32_t reclaimed = h.reclaim(0, tokens, positions);
    ASSERT_EQ(reclaimed, 3);

    // Verify cells are occupied
    ASSERT_TRUE(h.cells[10].occupied);
    ASSERT_TRUE(h.cells[11].occupied);
    ASSERT_TRUE(h.cells[12].occupied);

    // Rollback: un-reclaim (simulate prepare's reverse restore)
    for (auto idx : cells) {
        h.cells[idx].occupied = false;
        h.cells[idx].seq_id = -1;
        h.cells[idx].pos = -1;
    }

    // Cells should be empty again
    ASSERT_FALSE(h.cells[10].occupied);
    ASSERT_FALSE(h.cells[11].occupied);
    ASSERT_FALSE(h.cells[12].occupied);

    // But tree entries should still be valid (we didn't bump generations)
    std::vector<uint32_t> out;
    ASSERT_EQ(h.find(tokens, out), 3);
}

TEST(harness_interleaved_promote_invalidate) {
    PrefixCacheHarness h(64);

    // Promote sequence A
    h.promote({1, 2, 3}, {0, 1, 2});

    // Overwrite cell 1 with new data
    h.overwrite_cell(1, 5, 5);

    // Promote sequence B using cells that include the overwritten one
    h.promote({4, 5, 6}, {1, 3, 4});

    // Sequence A: only cell 0 valid (cell 1 was overwritten before B's promote)
    std::vector<uint32_t> out;
    ASSERT_EQ(h.find({1, 2, 3}, out), 1);

    // Sequence B: cell 1 was overwritten THEN re-promoted with new gen → should be valid
    out.clear();
    int32_t matched = h.find({4, 5, 6}, out);
    ASSERT_EQ(matched, 3);
    ASSERT_EQ(out[0], 1u);
    ASSERT_EQ(out[1], 3u);
    ASSERT_EQ(out[2], 4u);
}

TEST(harness_clear_tree_preserves_generations) {
    PrefixCacheHarness h(64);

    h.promote({1, 2, 3}, {0, 1, 2});

    // Bump some generations
    h.overwrite_cell(0, 0, 0);
    h.overwrite_cell(1, 0, 1);

    uint64_t gen0_before = h.cell_generations[0];
    uint64_t gen1_before = h.cell_generations[1];

    // Clear the tree
    h.clear_tree();

    // Generations should be preserved (they track cell state, not tree state)
    ASSERT_EQ(h.cell_generations[0], gen0_before);
    ASSERT_EQ(h.cell_generations[1], gen1_before);

    // Find should return nothing
    std::vector<uint32_t> out;
    ASSERT_EQ(h.find({1, 2, 3}, out), 0);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    printf("=== Radix Tree Integration Test Suite (Phase 3) ===\n\n");

    // Tests auto-registered via static constructors

    printf("\n=== Results: %d passed, %d failed ===\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
