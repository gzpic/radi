// Unit tests for llama_radix_tree and llama_vector_view
// Phase 1: VectorView tests
// Phase 2: Radix Tree core algorithm tests

#include "llama-radix-tree.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <vector>

static int n_pass = 0;
static int n_fail = 0;

#define TEST(name) \
    static void test_##name(); \
    struct test_reg_##name { \
        test_reg_##name() { \
            printf("  TEST %-50s ", #name); \
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
// Phase 1: VectorView tests
// ============================================================================

TEST(vectorview_construct_from_vector) {
    std::vector<int> v = {10, 20, 30, 40, 50};
    llama_vector_view<int> view(v);

    ASSERT_EQ(view.size(), 5);
    ASSERT_EQ(view[0], 10);
    ASSERT_EQ(view[1], 20);
    ASSERT_EQ(view[4], 50);
    ASSERT_TRUE(view.data() != nullptr);
}

TEST(vectorview_subview) {
    std::vector<int> v = {1, 2, 3, 4, 5, 6, 7, 8};
    llama_vector_view<int> view(v);
    auto sub = view.subview(2, 4); // [3, 4, 5, 6]

    ASSERT_EQ(sub.size(), 4);
    ASSERT_EQ(sub[0], 3);
    ASSERT_EQ(sub[1], 4);
    ASSERT_EQ(sub[2], 5);
    ASSERT_EQ(sub[3], 6);
}

TEST(vectorview_subview_shares_storage) {
    auto ptr = std::make_shared<std::vector<int>>(std::vector<int>{1, 2, 3, 4, 5});
    llama_vector_view<int> view(ptr);
    auto sub = view.subview(1, 3); // [2, 3, 4]

    // Modify through sub, should be visible through original shared_ptr
    sub[0] = 99;
    ASSERT_EQ((*ptr)[1], 99);
    ASSERT_EQ(view[1], 99);
}

TEST(vectorview_empty_default) {
    llama_vector_view<int> view;

    ASSERT_EQ(view.size(), 0);
    ASSERT_TRUE(view.empty());
    ASSERT_TRUE(view.data() == nullptr);
    ASSERT_TRUE(view.begin() == nullptr);
    ASSERT_TRUE(view.end() == nullptr);
}

TEST(vectorview_subview_out_of_range) {
    std::vector<int> v = {1, 2, 3};
    llama_vector_view<int> view(v);

    bool threw = false;
    try {
        view.subview(2, 5); // offset=2, length=5 → 2+5=7 > 3
    } catch (const std::out_of_range &) {
        threw = true;
    }
    ASSERT_TRUE(threw);
}

TEST(vectorview_equality) {
    std::vector<int> a = {1, 2, 3};
    std::vector<int> b = {1, 2, 3};
    std::vector<int> c = {1, 2, 4};

    llama_vector_view<int> va(a);
    llama_vector_view<int> vb(b);
    llama_vector_view<int> vc(c);

    ASSERT_TRUE(va == vb);  // same content, different storage
    ASSERT_TRUE(va != vc);  // different content
}

TEST(vectorview_iterator) {
    std::vector<int> v = {10, 20, 30};
    llama_vector_view<int> view(v);

    int sum = 0;
    for (auto it = view.begin(); it != view.end(); ++it) {
        sum += *it;
    }
    ASSERT_EQ(sum, 60);

    // range-for
    int sum2 = 0;
    for (int x : view) {
        sum2 += x;
    }
    ASSERT_EQ(sum2, 60);
}

TEST(vectorview_empty_iterator) {
    llama_vector_view<int> view;
    int count = 0;
    for (auto it = view.begin(); it != view.end(); ++it) {
        count++;
    }
    ASSERT_EQ(count, 0);
}

TEST(vectorview_nested_subview) {
    std::vector<int> v = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    llama_vector_view<int> view(v);
    auto sub1 = view.subview(2, 6);  // [2,3,4,5,6,7]
    auto sub2 = sub1.subview(1, 3);  // [3,4,5]

    ASSERT_EQ(sub2.size(), 3);
    ASSERT_EQ(sub2[0], 3);
    ASSERT_EQ(sub2[1], 4);
    ASSERT_EQ(sub2[2], 5);
}

// ============================================================================
// Phase 2: Radix Tree core algorithm tests
// ============================================================================

// Helper: build a key from a vector of ints (treated as llama_token)
static llama_radix_node_key make_key(const std::vector<llama_token> & tokens, int64_t extra = 0) {
    return llama_radix_node_key(llama_vector_view<llama_token>(tokens), extra);
}

// Helper: build a value with sequential cell indices and generation=1
static llama_radix_node_value make_value(int n, uint32_t start_cell = 0) {
    std::vector<uint32_t> cells(n);
    std::vector<uint64_t> gens(n, 1);
    for (int i = 0; i < n; i++) {
        cells[i] = start_cell + i;
    }
    llama_radix_node_value v;
    v.cell_indices     = llama_vector_view<uint32_t>(cells);
    v.cell_generations = llama_vector_view<uint64_t>(gens);
    return v;
}

TEST(tree_insert_single) {
    llama_radix_tree tree;
    auto key = make_key({1, 2, 3, 4, 5});
    auto val = make_value(5, 10);

    bool ok = tree.insert(key, val);
    ASSERT_TRUE(ok);

    auto result = tree.search(key);
    ASSERT_TRUE(result.success);
    ASSERT_EQ(result.matched_length, 5);
    ASSERT_EQ((int)result.cell_indices.size(), 5);
    ASSERT_EQ(result.cell_indices[0], 10u);
    ASSERT_EQ(result.cell_indices[4], 14u);
}

TEST(tree_insert_common_prefix) {
    llama_radix_tree tree;

    // Insert [1,2,3] → cells [10,11,12]
    tree.insert(make_key({1, 2, 3}), make_value(3, 10));

    // Insert [1,2,4] → cells [20,21,22]
    // Should split at position 2: [1,2] becomes parent, [3] and [4] become children
    tree.insert(make_key({1, 2, 4}), make_value(3, 20));

    // Search [1,2,3] should still work
    auto r1 = tree.search(make_key({1, 2, 3}));
    ASSERT_TRUE(r1.success);
    ASSERT_EQ(r1.matched_length, 3);
    ASSERT_EQ(r1.cell_indices[0], 10u);
    ASSERT_EQ(r1.cell_indices[1], 11u);
    ASSERT_EQ(r1.cell_indices[2], 12u);

    // Search [1,2,4] should also work
    auto r2 = tree.search(make_key({1, 2, 4}));
    ASSERT_TRUE(r2.success);
    ASSERT_EQ(r2.matched_length, 3);
    // First 2 cells come from the split parent [1,2] → should be [10,11]
    ASSERT_EQ(r2.cell_indices[0], 10u);
    ASSERT_EQ(r2.cell_indices[1], 11u);
    // Third cell comes from the new leaf [4] → should be [22] (offset 2 in the inserted value)
    ASSERT_EQ(r2.cell_indices[2], 22u);
}

TEST(tree_insert_duplicate) {
    llama_radix_tree tree;
    auto key = make_key({1, 2, 3});
    auto val = make_value(3, 10);

    tree.insert(key, val);
    bool ok = tree.insert(key, val); // duplicate
    ASSERT_TRUE(ok);

    // Should still find it
    auto r = tree.search(key);
    ASSERT_TRUE(r.success);
    ASSERT_EQ(r.matched_length, 3);
}

TEST(tree_insert_one_is_prefix_of_other) {
    llama_radix_tree tree;

    // Insert [1,2] first
    tree.insert(make_key({1, 2}), make_value(2, 10));

    // Insert [1,2,3,4] which extends the first
    tree.insert(make_key({1, 2, 3, 4}), make_value(4, 20));

    // Search [1,2] should match 2
    auto r1 = tree.search(make_key({1, 2}));
    ASSERT_TRUE(r1.success);
    ASSERT_EQ(r1.matched_length, 2);

    // Search [1,2,3,4] should match 4
    auto r2 = tree.search(make_key({1, 2, 3, 4}));
    ASSERT_TRUE(r2.success);
    ASSERT_EQ(r2.matched_length, 4);
}

TEST(tree_insert_reverse_order) {
    llama_radix_tree tree;

    // Insert longer first: [1,2,3,4]
    tree.insert(make_key({1, 2, 3, 4}), make_value(4, 10));

    // Insert shorter: [1,2] — should split the existing [1,2,3,4] node
    tree.insert(make_key({1, 2}), make_value(2, 20));

    auto r1 = tree.search(make_key({1, 2}));
    ASSERT_TRUE(r1.success);
    ASSERT_EQ(r1.matched_length, 2);

    auto r2 = tree.search(make_key({1, 2, 3, 4}));
    ASSERT_TRUE(r2.success);
    ASSERT_EQ(r2.matched_length, 4);
}

TEST(tree_search_prefix_match) {
    llama_radix_tree tree;

    // Tree has [1,2,3]
    tree.insert(make_key({1, 2, 3}), make_value(3, 10));

    // Search for [1,2,3,4,5] — should match first 3
    auto r = tree.search(make_key({1, 2, 3, 4, 5}));
    ASSERT_TRUE(r.success);
    ASSERT_EQ(r.matched_length, 3);
    ASSERT_EQ((int)r.cell_indices.size(), 3);
}

TEST(tree_search_no_match) {
    llama_radix_tree tree;

    tree.insert(make_key({1, 2, 3}), make_value(3, 10));

    // Search for [9, 8, 7] — no overlap
    auto r = tree.search(make_key({9, 8, 7}));
    ASSERT_FALSE(r.success);
    ASSERT_EQ(r.matched_length, 0);
}

TEST(tree_search_partial_edge_match) {
    llama_radix_tree tree;

    // Tree has edge [1,2,3]
    tree.insert(make_key({1, 2, 3}), make_value(3, 10));

    // Search for [1,2,9] — matches [1,2] but not [3]
    // Should report matched_length = 2 (the partial edge match)
    auto r = tree.search(make_key({1, 2, 9}));
    ASSERT_TRUE(r.success);
    ASSERT_EQ(r.matched_length, 2);
    ASSERT_EQ((int)r.cell_indices.size(), 2);
    ASSERT_EQ(r.cell_indices[0], 10u);
    ASSERT_EQ(r.cell_indices[1], 11u);
}

TEST(tree_search_empty_key) {
    llama_radix_tree tree;
    tree.insert(make_key({1, 2, 3}), make_value(3, 10));

    auto r = tree.search(make_key({}));
    ASSERT_FALSE(r.success);
    ASSERT_EQ(r.matched_length, 0);
}

TEST(tree_invalidate_cell) {
    llama_radix_tree tree;

    // Insert [1,2,3] → cells [10,11,12], gens [1,1,1]
    tree.insert(make_key({1, 2, 3}), make_value(3, 10));

    // Invalidate cell 11 (the middle one)
    tree.invalidate_cell(11);

    // Search should still return matched_length=3, but cell_generations[1] should be 0
    auto r = tree.search(make_key({1, 2, 3}));
    ASSERT_TRUE(r.success);
    ASSERT_EQ(r.matched_length, 3);
    // The generation for the invalidated cell should be 0
    ASSERT_EQ(r.cell_generations[1], 0u);
}

TEST(tree_invalidate_nonexistent_cell) {
    llama_radix_tree tree;
    tree.insert(make_key({1, 2, 3}), make_value(3, 10));

    // Invalidating a cell that doesn't exist in the tree should be a no-op
    tree.invalidate_cell(999);

    auto r = tree.search(make_key({1, 2, 3}));
    ASSERT_TRUE(r.success);
    ASSERT_EQ(r.matched_length, 3);
}

TEST(tree_clear) {
    llama_radix_tree tree;
    tree.insert(make_key({1, 2, 3}), make_value(3, 10));
    tree.insert(make_key({4, 5, 6}), make_value(3, 20));

    tree.clear();

    auto r1 = tree.search(make_key({1, 2, 3}));
    ASSERT_FALSE(r1.success);
    auto r2 = tree.search(make_key({4, 5, 6}));
    ASSERT_FALSE(r2.success);
}

TEST(tree_extra_key_isolation) {
    llama_radix_tree tree;

    // Same tokens, different extra_key
    tree.insert(make_key({1, 2, 3}, 0), make_value(3, 10));
    tree.insert(make_key({1, 2, 3}, 1), make_value(3, 20));

    // Search with extra_key=0 should find cells [10,11,12]
    auto r0 = tree.search(make_key({1, 2, 3}, 0));
    ASSERT_TRUE(r0.success);
    ASSERT_EQ(r0.matched_length, 3);
    ASSERT_EQ(r0.cell_indices[0], 10u);

    // Search with extra_key=1 should find cells [20,21,22]
    auto r1 = tree.search(make_key({1, 2, 3}, 1));
    ASSERT_TRUE(r1.success);
    ASSERT_EQ(r1.matched_length, 3);
    ASSERT_EQ(r1.cell_indices[0], 20u);

    // Search with extra_key=2 should find nothing
    auto r2 = tree.search(make_key({1, 2, 3}, 2));
    ASSERT_FALSE(r2.success);
}

TEST(tree_invalidate_after_split) {
    llama_radix_tree tree;

    // Insert [1,2,3,4] → cells [10,11,12,13]
    tree.insert(make_key({1, 2, 3, 4}), make_value(4, 10));

    // Insert [1,2,5] → triggers split at position 2
    tree.insert(make_key({1, 2, 5}), make_value(3, 20));

    // Invalidate cell 11 (in the shared prefix [1,2])
    tree.invalidate_cell(11);

    // Both paths should show the invalidation
    auto r1 = tree.search(make_key({1, 2, 3, 4}));
    ASSERT_TRUE(r1.success);
    ASSERT_EQ(r1.cell_generations[1], 0u);

    auto r2 = tree.search(make_key({1, 2, 5}));
    ASSERT_TRUE(r2.success);
    ASSERT_EQ(r2.cell_generations[1], 0u);
}

TEST(tree_multiple_branches) {
    llama_radix_tree tree;

    // Create a tree with multiple branches
    tree.insert(make_key({1, 2, 3}),    make_value(3, 10));
    tree.insert(make_key({1, 2, 4}),    make_value(3, 20));
    tree.insert(make_key({1, 2, 5}),    make_value(3, 30));
    tree.insert(make_key({1, 3, 6}),    make_value(3, 40));
    tree.insert(make_key({2, 7, 8}),    make_value(3, 50));

    // Verify all can be found
    ASSERT_EQ(tree.search(make_key({1, 2, 3})).matched_length, 3);
    ASSERT_EQ(tree.search(make_key({1, 2, 4})).matched_length, 3);
    ASSERT_EQ(tree.search(make_key({1, 2, 5})).matched_length, 3);
    ASSERT_EQ(tree.search(make_key({1, 3, 6})).matched_length, 3);
    ASSERT_EQ(tree.search(make_key({2, 7, 8})).matched_length, 3);

    // Partial match: [1,2] matches prefix of [1,2,3], [1,2,4], [1,2,5]
    auto r = tree.search(make_key({1, 2, 99}));
    ASSERT_TRUE(r.success);
    ASSERT_EQ(r.matched_length, 2);
}

TEST(tree_large_scale_insert) {
    llama_radix_tree tree;
    std::mt19937 rng(42);

    const int N = 500;
    std::vector<std::vector<llama_token>> sequences;

    // Generate 500 random sequences of length 5-20
    for (int i = 0; i < N; i++) {
        int len = 5 + (rng() % 16);
        std::vector<llama_token> seq(len);
        for (int j = 0; j < len; j++) {
            seq[j] = rng() % 100; // small vocab to force prefix sharing
        }
        sequences.push_back(seq);
    }

    // Insert all
    for (int i = 0; i < N; i++) {
        auto key = make_key(sequences[i]);
        auto val = make_value((int)sequences[i].size(), (uint32_t)(i * 100));
        tree.insert(key, val);
    }

    // Verify all can be found (at least prefix match)
    int found = 0;
    for (int i = 0; i < N; i++) {
        auto r = tree.search(make_key(sequences[i]));
        if (r.success && r.matched_length > 0) {
            found++;
        }
    }
    // All should have at least a partial match (they were inserted)
    ASSERT_EQ(found, N);
}

TEST(tree_single_token_sequences) {
    llama_radix_tree tree;

    tree.insert(make_key({1}), make_value(1, 10));
    tree.insert(make_key({2}), make_value(1, 20));
    tree.insert(make_key({3}), make_value(1, 30));

    auto r1 = tree.search(make_key({1}));
    ASSERT_TRUE(r1.success);
    ASSERT_EQ(r1.matched_length, 1);
    ASSERT_EQ(r1.cell_indices[0], 10u);

    auto r2 = tree.search(make_key({2}));
    ASSERT_TRUE(r2.success);
    ASSERT_EQ(r2.cell_indices[0], 20u);

    auto r3 = tree.search(make_key({4})); // not inserted
    ASSERT_FALSE(r3.success);
}

TEST(tree_insert_empty_key) {
    llama_radix_tree tree;
    auto key = make_key({});
    auto val = make_value(0);

    bool ok = tree.insert(key, val);
    ASSERT_TRUE(ok); // should be a no-op, not crash
}

TEST(tree_dot_visualization) {
    llama_radix_tree tree;
    tree.insert(make_key({1, 2, 3}), make_value(3, 10));
    tree.insert(make_key({1, 2, 4}), make_value(3, 20));

    std::string dot = tree.dot();
    // Should at least contain digraph header
    ASSERT_TRUE(dot.find("digraph") != std::string::npos);
    ASSERT_TRUE(dot.size() > 20);
}

TEST(tree_invalidate_all_cells_in_node) {
    llama_radix_tree tree;

    tree.insert(make_key({1, 2, 3}), make_value(3, 10));

    // Invalidate all 3 cells
    tree.invalidate_cell(10);
    tree.invalidate_cell(11);
    tree.invalidate_cell(12);

    auto r = tree.search(make_key({1, 2, 3}));
    ASSERT_TRUE(r.success); // tree structure still intact
    ASSERT_EQ(r.matched_length, 3);
    // But all generations should be 0
    ASSERT_EQ(r.cell_generations[0], 0u);
    ASSERT_EQ(r.cell_generations[1], 0u);
    ASSERT_EQ(r.cell_generations[2], 0u);
}

TEST(tree_hash_consistency) {
    // Verify that hash and operator== are consistent:
    // equal keys must have equal hashes
    llama_radix_node_key_hash hasher;

    auto k1 = make_key({1, 2, 3}, 0);
    auto k2 = make_key({1, 2, 3}, 0);
    auto k3 = make_key({1, 2, 3}, 1); // different extra_key

    ASSERT_TRUE(k1 == k2);
    ASSERT_EQ(hasher(k1), hasher(k2));

    ASSERT_TRUE(k1 != k3);
    // Different extra_key should (usually) produce different hash
    // Not guaranteed, but very likely with FNV-1a
}

TEST(tree_deep_chain) {
    // Test a deep chain: each insert extends by 1 token
    llama_radix_tree tree;

    for (int len = 1; len <= 50; len++) {
        std::vector<llama_token> tokens(len);
        for (int i = 0; i < len; i++) tokens[i] = i + 1;
        tree.insert(make_key(tokens), make_value(len, (uint32_t)(len * 100)));
    }

    // Search for the longest chain
    std::vector<llama_token> full(50);
    for (int i = 0; i < 50; i++) full[i] = i + 1;
    auto r = tree.search(make_key(full));
    ASSERT_TRUE(r.success);
    ASSERT_EQ(r.matched_length, 50);
}

TEST(tree_cell_reuse_different_nodes) {
    llama_radix_tree tree;

    // Two different keys pointing to the same cell indices
    tree.insert(make_key({1, 2}), make_value(2, 10));
    tree.insert(make_key({3, 4}), make_value(2, 10)); // same cells!

    // Invalidating cell 10 should affect both
    tree.invalidate_cell(10);

    auto r1 = tree.search(make_key({1, 2}));
    ASSERT_EQ(r1.cell_generations[0], 0u);

    auto r2 = tree.search(make_key({3, 4}));
    ASSERT_EQ(r2.cell_generations[0], 0u);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    printf("=== Radix Tree Test Suite ===\n\n");
    printf("Phase 1: VectorView\n");
    printf("Phase 2: Radix Tree Core\n\n");

    // Tests are auto-registered by static constructors above
    // They've already run by the time we reach main()

    printf("\n=== Results: %d passed, %d failed ===\n", n_pass, n_fail);

    return n_fail > 0 ? 1 : 0;
}
