// Radix Tree for KV Cache Prefix Matching
// Ported from mllm prefix_cache/RadixTree, adapted for llama.cpp

#include "llama-radix-tree.h"

#include <algorithm>
#include <cassert>

//
// llama_radix_node_key
//

llama_radix_node_key::llama_radix_node_key(
        const llama_vector_view<llama_token> & tokens, int64_t extra)
    : token_ids(tokens), extra_key(extra) {}

bool llama_radix_node_key::operator==(const llama_radix_node_key & o) const noexcept {
    return extra_key == o.extra_key && token_ids == o.token_ids;
}

bool llama_radix_node_key::operator!=(const llama_radix_node_key & o) const noexcept {
    return !(*this == o);
}

//
// llama_radix_node_key_hash
//
// Simple FNV-1a hash over token ids. The children map is typically small,
// so hash quality is not critical.
//

size_t llama_radix_node_key_hash::operator()(const llama_radix_node_key & k) const noexcept {
    size_t h = 14695981039346656037ULL; // FNV offset basis
    for (int32_t i = 0; i < k.token_ids.size(); ++i) {
        h ^= static_cast<size_t>(k.token_ids[i]);
        h *= 1099511628211ULL; // FNV prime
    }
    h ^= static_cast<size_t>(k.extra_key);
    h *= 1099511628211ULL;
    return h;
}

//
// llama_radix_node
//

llama_radix_node::llama_radix_node() = default;

llama_radix_node::llama_radix_node(
        const llama_radix_node_key & k,
        const llama_radix_node_value & v,
        llama_radix_node * p)
    : key(k), value(v), parent(p) {}

//
// llama_radix_tree
//

llama_radix_tree::llama_radix_tree(const llama_radix_tree_options & options)
    : options_(options), root_(std::make_unique<llama_radix_node>()) {
    std::vector<llama_token> root_tokens{-1}; // sentinel
    root_->key = llama_radix_node_key(llama_vector_view<llama_token>(root_tokens));
    node_count_ = 1;
}

llama_radix_tree::~llama_radix_tree() {
    clear();
}

void llama_radix_tree::clear() {
    for (auto & [key, child] : root_->children) {
        delete_recursive(child);
    }
    root_->children.clear();
    root_->parent = nullptr;
    node_count_ = 1;
    cell_to_nodes_.clear();
}

bool llama_radix_tree::insert(const llama_radix_node_key & key, const llama_radix_node_value & value) {
    if (key.token_ids.empty()) {
        return true;
    }

    llama_radix_node * cur = root_.get();
    size_t matched = 0;
    llama_radix_node * deepest = cur;
    size_t edge_matched = 0;

    // Walk down the tree, matching as many tokens as possible
    while (matched < (size_t)key.token_ids.size() && cur) {
        auto rest = key.token_ids.subview(matched, key.token_ids.size() - matched);
        llama_radix_node_key rest_key{rest, key.extra_key};

        bool found = false;
        for (auto & [child_key, child_ptr] : cur->children) {
            size_t ml = matched_length(rest_key, child_key);
            if (ml == 0) { continue; }
            deepest = child_ptr;
            edge_matched = ml;
            matched += ml;
            found = true;
            break;
        }
        if (!found) { break; }

        // If we only matched part of this edge, we cannot descend further
        // (the children expect tokens starting after the full edge, not mid-edge)
        if (edge_matched < (size_t)deepest->key.token_ids.size()) {
            break;
        }
        cur = deepest;
    }

    // If we matched all tokens in the key but only partially matched the last
    // edge, we need to split that edge first so the upper node exactly covers
    // the matched portion.
    if (matched == (size_t)key.token_ids.size()
            && edge_matched > 0
            && edge_matched < (size_t)deepest->key.token_ids.size()) {
        auto [upper, lower] = split(deepest, edge_matched);
        (void)lower;
        deepest = upper;
    }

    // Full match - update value and bump ref count
    if (matched == (size_t)key.token_ids.size()) {
        // Update value: the caller may be re-promoting with new cell indices
        // after the old ones were invalidated. We need to rebuild the value
        // across all nodes on the matched path.
        llama_radix_node * walk = root_.get();
        size_t walk_matched = 0;
        while (walk_matched < matched) {
            bool found_child = false;
            auto rest = key.token_ids.subview(walk_matched, key.token_ids.size() - walk_matched);
            llama_radix_node_key rest_key{rest, key.extra_key};
            for (auto & [child_key, child_ptr] : walk->children) {
                size_t ml = matched_length(rest_key, child_key);
                if (ml == 0) { continue; }
                // Update this child's value with the new cells/generations
                unregister_cells(child_ptr);
                size_t update_len = (size_t)child_ptr->key.token_ids.size();
                child_ptr->value.cell_indices    = value.cell_indices.subview(walk_matched, update_len);
                child_ptr->value.cell_generations = value.cell_generations.subview(walk_matched, update_len);
                register_cells(child_ptr);
                walk_matched += update_len;
                walk = child_ptr;
                found_child = true;
                break;
            }
            if (!found_child) { break; }
        }
        deepest->ref_count++;
        return true;
    }

    // Partial match within an edge - split the node
    if (matched < (size_t)key.token_ids.size() && edge_matched > 0
            && edge_matched < (size_t)deepest->key.token_ids.size()) {
        auto [upper, lower] = split(deepest, edge_matched);
        (void)lower;
        deepest = upper;
    }

    cur = deepest;

    // Create leaf for remaining tokens
    auto suffix = key.token_ids.subview(matched, key.token_ids.size() - matched);
    auto leaf_key = llama_radix_node_key(suffix, key.extra_key);
    llama_radix_node_value leaf_value;
    leaf_value.cell_indices    = value.cell_indices.subview(matched, key.token_ids.size() - matched);
    leaf_value.cell_generations = value.cell_generations.subview(matched, key.token_ids.size() - matched);

    llama_radix_node * leaf = new llama_radix_node(leaf_key, leaf_value, cur);
    auto [it, inserted] = cur->children.emplace(leaf_key, leaf);
    if (!inserted) {
        // key already exists (should not happen with correct logic, but guard against leaks)
        delete leaf;
        return true;
    }
    node_count_++;

    register_cells(leaf);

    return true;
}

llama_radix_search_result llama_radix_tree::search(const llama_radix_node_key & key) {
    llama_radix_search_result result;
    result.success = false;
    result.matched_length = 0;

    llama_radix_node * cur_node = root_.get();
    size_t cur_searched_len = 0;

    std::vector<std::pair<llama_radix_node *, int32_t>> path;
    path.emplace_back(cur_node, 0);

    while (cur_searched_len < (size_t)key.token_ids.size() && cur_node) {
        bool found_next = false;
        auto sub = key.token_ids.subview(cur_searched_len, key.token_ids.size() - cur_searched_len);

        for (auto & [child_key, child_node] : cur_node->children) {
            auto ml = matched_length(llama_radix_node_key{sub, key.extra_key}, child_key);
            if (ml) {
                cur_node = child_node;
                cur_searched_len += ml;
                path.emplace_back(cur_node, static_cast<int32_t>(ml));
                found_next = true;
                break;
            }
        }

        if (!found_next) { break; }

        // If we only partially matched this edge, stop (cannot descend into children)
        if (cur_searched_len < (size_t)key.token_ids.size()) {
            const auto & last = path.back();
            if ((size_t)last.second < (size_t)last.first->key.token_ids.size()) {
                break;
            }
        }
    }

    if (cur_searched_len) {
        result.success = true;
        result.path = path;
        result.matched_length = static_cast<int32_t>(cur_searched_len);

        // Update access metadata for LRU eviction on matched nodes
        auto now = std::chrono::steady_clock::now();
        for (auto & [node, len] : path) {
            if (len > 0) {
                node->last_accessed = now;
                node->hit_count++;
            }
        }

        // Flatten cell indices and generations from the path
        for (auto & [node, len] : path) {
            for (int32_t i = 0; i < len; ++i) {
                result.cell_indices.push_back(node->value.cell_indices[i]);
                result.cell_generations.push_back(node->value.cell_generations[i]);
            }
        }
    } else {
        result.path = path;
    }

    return result;
}

void llama_radix_tree::invalidate_cell(uint32_t cell_idx) {
    auto it = cell_to_nodes_.find(cell_idx);
    if (it == cell_to_nodes_.end()) {
        return;
    }

    // For each node referencing this cell, mark the generation as invalid (0)
    for (auto & [node, offset] : it->second) {
        if (offset < node->value.cell_generations.size()) {
            node->value.cell_generations[offset] = 0;
        }
    }

    cell_to_nodes_.erase(it);
}

std::string llama_radix_tree::dot() const {
    std::ostringstream os;
    os << "digraph Radix {\n"
       << "  node [shape=box, fontname=\"Mono\"];\n"
       << "  edge [arrowhead=vee];\n";
    int32_t id_cnt = 0;
    dot_helper(root_.get(), -1, id_cnt, os);
    os << "}\n";
    return os.str();
}

//
// Private methods
//

std::pair<llama_radix_node *, llama_radix_node *>
llama_radix_tree::split(llama_radix_node * node, size_t position) {
    const size_t old_sz = node->key.token_ids.size();
    if (position == 0)       { return {nullptr, node}; }
    if (position >= old_sz)  { return {node, nullptr}; }

    // Unregister cells of this node before restructuring
    unregister_cells(node);

    // Create lower node with the tail portion
    auto lower_tokens = node->key.token_ids.subview(position, old_sz - position);
    llama_radix_node_key lower_key(lower_tokens, node->key.extra_key);

    llama_radix_node_value lower_value;
    lower_value.cell_indices    = node->value.cell_indices.subview(position, old_sz - position);
    lower_value.cell_generations = node->value.cell_generations.subview(position, old_sz - position);

    auto * lower = new llama_radix_node(lower_key, lower_value, node);

    // Move children from upper to lower
    lower->children.swap(node->children);
    for (auto & [k, c] : lower->children) {
        c->parent = lower;
    }

    // Add lower as child of upper
    node->children.emplace(lower_key, lower);

    // Truncate upper node to the head portion
    auto old_node_key = node->key;
    node->key.token_ids    = node->key.token_ids.subview(0, position);
    node->value.cell_indices    = node->value.cell_indices.subview(0, position);
    node->value.cell_generations = node->value.cell_generations.subview(0, position);

    node_count_++;

    // Update parent's children map (key changed)
    if (node->parent) {
        auto & parent_children = node->parent->children;
        auto it = parent_children.find(old_node_key);
        if (it != parent_children.end()) {
            parent_children.erase(it);
            parent_children.emplace(node->key, node);
        }
    }

    // Re-register cells for both nodes
    register_cells(node);
    register_cells(lower);

    return {node, lower};
}

void llama_radix_tree::delete_recursive(llama_radix_node * node) {
    if (!node) { return; }
    for (auto & [key, child] : node->children) {
        delete_recursive(child);
    }
    unregister_cells(node);
    delete node;
}

size_t llama_radix_tree::matched_length(
        const llama_radix_node_key & a,
        const llama_radix_node_key & b) {
    if (a.extra_key != b.extra_key) { return 0; }
    size_t n = std::min<size_t>(a.token_ids.size(), b.token_ids.size());
    for (size_t i = 0; i < n; ++i) {
        if (a.token_ids[i] != b.token_ids[i]) { return i; }
    }
    return n;
}

void llama_radix_tree::register_cells(llama_radix_node * node) {
    for (int32_t i = 0; i < node->value.cell_indices.size(); ++i) {
        uint32_t idx = node->value.cell_indices[i];
        cell_to_nodes_[idx].emplace_back(node, i);
    }
}

void llama_radix_tree::unregister_cells(llama_radix_node * node) {
    for (int32_t i = 0; i < node->value.cell_indices.size(); ++i) {
        uint32_t idx = node->value.cell_indices[i];
        auto it = cell_to_nodes_.find(idx);
        if (it != cell_to_nodes_.end()) {
            auto & vec = it->second;
            vec.erase(
                std::remove_if(vec.begin(), vec.end(),
                    [node, i](const std::pair<llama_radix_node *, int32_t> & p) {
                        return p.first == node && p.second == i;
                    }),
                vec.end());
            if (vec.empty()) {
                cell_to_nodes_.erase(it);
            }
        }
    }
}

void llama_radix_tree::dot_helper(
        const llama_radix_node * n,
        int32_t pid,
        int32_t & id_gen,
        std::ostringstream & os) {
    if (!n) { return; }
    int32_t my_id = id_gen++;
    os << "  n" << my_id << " [label=\"";
    for (int32_t i = 0; i < n->key.token_ids.size(); ++i) {
        os << n->key.token_ids[i] << " ";
    }
    os << "\\nref=" << n->ref_count << "\"];\n";
    if (pid >= 0) {
        os << "  n" << pid << " -> n" << my_id << ";\n";
    }
    for (auto & [k, c] : n->children) {
        dot_helper(c, my_id, id_gen, os);
    }
}

//
// LRU eviction (A6)
//

void llama_radix_tree::collect_leaves(llama_radix_node * node, std::vector<llama_radix_node *> & leaves) {
    if (!node) { return; }
    if (node->children.empty()) {
        // leaf node (but not root)
        if (node->parent != nullptr) {
            leaves.push_back(node);
        }
        return;
    }
    for (auto & [k, c] : node->children) {
        collect_leaves(c, leaves);
    }
}

void llama_radix_tree::remove_leaf(llama_radix_node * leaf) {
    if (!leaf || !leaf->parent) { return; }

    // unregister cells from reverse index
    unregister_cells(leaf);

    // remove from parent's children
    llama_radix_node * parent = leaf->parent;
    for (auto it = parent->children.begin(); it != parent->children.end(); ++it) {
        if (it->second == leaf) {
            parent->children.erase(it);
            break;
        }
    }

    delete leaf;
    node_count_--;

    // if parent now has exactly one child and is not root, merge parent with child
    if (parent->parent != nullptr && parent->children.size() == 1) {
        // optional: could merge parent+child to maintain compact tree
        // for now, leave as-is (correctness first)
    }
}

int32_t llama_radix_tree::evict_lru(int32_t max_nodes) {
    if (node_count_ <= max_nodes) {
        return 0;
    }

    int32_t evicted = 0;

    while (node_count_ > max_nodes) {
        // collect current leaves
        std::vector<llama_radix_node *> leaves;
        collect_leaves(root_.get(), leaves);

        if (leaves.empty()) {
            break;  // nothing to evict
        }

        // find the least-recently-accessed leaf
        auto oldest = std::min_element(leaves.begin(), leaves.end(),
            [](const llama_radix_node * a, const llama_radix_node * b) {
                return a->last_accessed < b->last_accessed;
            });

        remove_leaf(*oldest);
        evicted++;
    }

    return evicted;
}
