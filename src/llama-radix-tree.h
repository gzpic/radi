// Radix Tree for KV Cache Prefix Matching
// Ported from mllm (https://github.com/mllm-team/mllm) prefix_cache/RadixTree
// Adapted for llama.cpp: uses llama_token + cell index instead of vp_addr_t

#pragma once

#include "llama.h"

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

//
// VectorView - zero-copy sub-range view over a shared vector
//
// When the radix tree splits a node, the token sequence and cell index array
// need to be divided into two parts. VectorView avoids O(n) copies by sharing
// the underlying storage and adjusting offset/length.
//
template<typename T>
class llama_vector_view {
    static_assert(!std::is_const_v<T>, "T must not be const.");

public:
    using value_type     = T;
    using size_type      = int32_t;
    using reference      = T &;
    using const_reference = const T &;
    using pointer        = T *;
    using const_pointer  = const T *;
    using iterator       = T *;
    using const_iterator = const T *;

    llama_vector_view() = default;

    explicit llama_vector_view(const std::vector<T> & vec)
        : offset_(0),
          length_(static_cast<size_type>(vec.size())),
          data_(std::make_shared<std::vector<T>>(vec)) {}

    explicit llama_vector_view(std::shared_ptr<std::vector<T>> vec_ptr)
        : offset_(0),
          length_(vec_ptr ? static_cast<size_type>(vec_ptr->size()) : 0),
          data_(vec_ptr) {}

    llama_vector_view(std::shared_ptr<std::vector<T>> vec_ptr, size_type off, size_type len)
        : data_(std::move(vec_ptr)) {
        if (!data_) {
            offset_ = 0;
            length_ = 0;
            return;
        }
        if (off < 0 || len < 0 || off + len > static_cast<size_type>(data_->size())) {
            throw std::out_of_range("llama_vector_view: invalid offset/length");
        }
        offset_ = off;
        length_ = len;
    }

    reference       operator[](size_type i)       { return (*data_)[offset_ + i]; }
    const_reference operator[](size_type i) const { return (*data_)[offset_ + i]; }

    pointer       data()       { return data_ ? data_->data() + offset_ : nullptr; }
    const_pointer data() const { return data_ ? data_->data() + offset_ : nullptr; }

    size_type size()  const noexcept { return length_; }
    bool      empty() const noexcept { return length_ == 0; }

    iterator       begin()       noexcept { return data(); }
    iterator       end()         noexcept { return data() ? data() + length_ : nullptr; }
    const_iterator begin() const noexcept { return data(); }
    const_iterator end()   const noexcept { return data() ? data() + length_ : nullptr; }

    llama_vector_view subview(size_type off, size_type len) const {
        if (off < 0 || len < 0 || off + len > length_) {
            throw std::out_of_range("subview: out of range");
        }
        return llama_vector_view(data_, offset_ + off, len);
    }

    friend bool operator==(const llama_vector_view & a, const llama_vector_view & b) noexcept {
        if (a.size() != b.size()) { return false; }
        for (size_type i = 0; i < a.size(); ++i) {
            if (a[i] != b[i]) { return false; }
        }
        return true;
    }

    friend bool operator!=(const llama_vector_view & a, const llama_vector_view & b) noexcept {
        return !(a == b);
    }

private:
    size_type offset_ = 0;
    size_type length_ = 0;
    std::shared_ptr<std::vector<T>> data_;
};

//
// Radix Tree Node Key
//
// Each edge in the tree is labeled with a variable-length token sequence.
// extra_key distinguishes different contexts (e.g. LoRA adapters).
//
struct llama_radix_node_key {
    llama_vector_view<llama_token> token_ids;
    int64_t extra_key = 0;

    llama_radix_node_key() = default;
    llama_radix_node_key(const llama_vector_view<llama_token> & tokens, int64_t extra = 0);

    bool operator==(const llama_radix_node_key & o) const noexcept;
    bool operator!=(const llama_radix_node_key & o) const noexcept;
};

struct llama_radix_node_key_hash {
    size_t operator()(const llama_radix_node_key & k) const noexcept;
};

//
// Radix Tree Node Value
//
// In llama.cpp, a single cell index determines the KV storage position across
// ALL layers (unlike mllm where each layer has independent addresses).
// We also store a generation counter per cell to detect invalidation.
//
struct llama_radix_node_value {
    llama_vector_view<uint32_t> cell_indices;    // cell index for each token in this edge
    llama_vector_view<uint64_t> cell_generations; // generation at insert time, for validity check
};

//
// Radix Tree Node
//
struct llama_radix_node {
    llama_radix_node_key   key;
    llama_radix_node_value value;

    llama_radix_node * parent = nullptr;
    std::unordered_map<llama_radix_node_key, llama_radix_node *, llama_radix_node_key_hash> children;

    // metadata for LRU eviction
    int32_t ref_count = 0;
    int32_t hit_count = 0;
    std::chrono::steady_clock::time_point last_accessed{std::chrono::steady_clock::now()};

    llama_radix_node();
    llama_radix_node(const llama_radix_node_key & k, const llama_radix_node_value & v, llama_radix_node * p = nullptr);
};

//
// Radix Tree Search Result
//
struct llama_radix_search_result {
    bool    success        = false;
    int32_t matched_length = 0;

    // path of (node, matched_count_in_this_node)
    std::vector<std::pair<llama_radix_node *, int32_t>> path;

    // flattened cell indices and generations for the matched prefix
    std::vector<uint32_t> cell_indices;
    std::vector<uint64_t> cell_generations;
};

//
// Radix Tree Options
//
struct llama_radix_tree_options {
    bool    enable_lru_eviction    = true;
    float   eviction_threshold     = 0.9f;
    bool    enable_path_compression = false;
    size_t  min_compression_length = 2;
};

//
// Radix Tree
//
// Manages a prefix index over token sequences. Each node stores the cell
// indices where the corresponding KV data resides in the llama_kv_cache.
//
// Thread safety: NOT thread-safe. Caller must synchronize.
//
class llama_radix_tree {
public:
    explicit llama_radix_tree(const llama_radix_tree_options & options = {});
    ~llama_radix_tree();

    void clear();

    // Insert a token sequence with its cell indices and generation counters.
    // Splits existing nodes if necessary.
    bool insert(const llama_radix_node_key & key, const llama_radix_node_value & value);

    // Search for the longest prefix match. Does not modify the tree.
    llama_radix_search_result search(const llama_radix_node_key & key);

    // Invalidate all tree entries that reference the given cell index.
    // Called when a cell is overwritten or freed.
    void invalidate_cell(uint32_t cell_idx);

    // GraphViz visualization
    std::string dot() const;

private:
    int32_t node_count_ = 0;
    llama_radix_tree_options options_;
    std::unique_ptr<llama_radix_node> root_;

    // cell_idx -> set of (node, token_offset_in_node) that reference this cell
    // enables O(1) invalidation when a cell is overwritten
    std::unordered_map<uint32_t, std::vector<std::pair<llama_radix_node *, int32_t>>> cell_to_nodes_;

    std::pair<llama_radix_node *, llama_radix_node *> split(llama_radix_node * node, size_t position);

    void delete_recursive(llama_radix_node * node);

    static size_t matched_length(const llama_radix_node_key & a, const llama_radix_node_key & b);

    // helpers for cell_to_nodes_ tracking
    void register_cells(llama_radix_node * node);
    void unregister_cells(llama_radix_node * node);

    static void dot_helper(const llama_radix_node * n, int32_t pid, int32_t & id_gen, std::ostringstream & os);
};
