#include "llama-session.h"

#include <algorithm>
#include <cmath>

llama_session_params llama_session::default_params() {
    llama_session_params params = {};
    params.evict_threshold = 0.8f;
    params.pressure_warn = 0.9f;
    params.n_ctx = 0;
    return params;
}

llama_session::llama_session(llama_context * ctx, llama_session_params params)
    : ctx_(ctx)
    , mem_(ctx ? llama_get_memory(ctx) : nullptr)
    , params_(params) {
    if (params_.evict_threshold <= 0.0f) {
        params_.evict_threshold = 0.8f;
    }
    if (params_.pressure_warn <= 0.0f) {
        params_.pressure_warn = 0.9f;
    }
    if (params_.pressure_warn > 1.0f) {
        params_.pressure_warn = 1.0f;
    }
}

llama_session_turn * llama_session::find_turn(int32_t turn_id) {
    for (auto & turn : turns_) {
        if (turn.turn_id == turn_id) {
            return &turn;
        }
    }
    return nullptr;
}

const llama_session_turn * llama_session::find_turn(int32_t turn_id) const {
    for (const auto & turn : turns_) {
        if (turn.turn_id == turn_id) {
            return &turn;
        }
    }
    return nullptr;
}

bool llama_session::has_open_turn() const {
    for (const auto & turn : turns_) {
        if (!turn.ended) {
            return true;
        }
    }
    return false;
}

uint32_t llama_session::effective_n_ctx() const {
    if (params_.n_ctx > 0) {
        return params_.n_ctx;
    }
    if (!ctx_) {
        return 0;
    }
    return llama_n_ctx(ctx_);
}

int32_t llama_session::turn_begin() {
    if (!mem_ || has_open_turn()) {
        return -1;
    }

    const llama_pos cur = llama_memory_seq_pos_max(mem_, 0);
    const llama_pos p0 = cur >= 0 ? cur + 1 : 0;

    llama_session_turn turn;
    turn.turn_id = next_turn_id_++;
    turn.p0 = p0;
    turn.p1 = p0;
    turn.ended = false;
    turns_.push_back(std::move(turn));
    return turns_.back().turn_id;
}

bool llama_session::turn_add_tokens(int32_t turn_id, const llama_token * tokens, int32_t n_tokens) {
    if (!mem_ || !tokens || n_tokens <= 0) {
        return false;
    }

    auto * turn = find_turn(turn_id);
    if (!turn || turn->ended) {
        return false;
    }

    turn->tokens.insert(turn->tokens.end(), tokens, tokens + n_tokens);
    turn->p1 += n_tokens;
    return true;
}

bool llama_session::turn_end(int32_t turn_id) {
    if (!mem_) {
        return false;
    }

    auto * turn = find_turn(turn_id);
    if (!turn || turn->ended) {
        return false;
    }

    const llama_pos cur = llama_memory_seq_pos_max(mem_, 0);
    if (cur >= turn->p0 - 1) {
        turn->p1 = cur + 1;
    }

    const int32_t expected = std::max<int32_t>(0, turn->p1 - turn->p0);
    if ((int32_t) turn->tokens.size() > expected) {
        turn->tokens.resize(expected);
    }

    turn->ended = true;
    return true;
}

int32_t llama_session::checkpoint_save() {
    if (!mem_) {
        return -1;
    }
    return llama_memory_checkpoint_save(mem_);
}

void llama_session::rollback_turns_to_pos(llama_pos new_end_exclusive) {
    for (size_t i = 0; i < turns_.size();) {
        auto & turn = turns_[i];

        if (turn.p0 >= new_end_exclusive) {
            turns_.erase(turns_.begin() + i);
            continue;
        }

        if (turn.p1 > new_end_exclusive) {
            turn.p1 = new_end_exclusive;
            const int32_t expected = std::max<int32_t>(0, turn.p1 - turn.p0);
            if ((int32_t) turn.tokens.size() > expected) {
                turn.tokens.resize(expected);
            }
            turn.ended = true;
            if (turn.p1 <= turn.p0) {
                turns_.erase(turns_.begin() + i);
                continue;
            }
        }

        ++i;
    }
}

bool llama_session::checkpoint_rollback(int32_t checkpoint_id) {
    if (!mem_) {
        return false;
    }

    if (!llama_memory_checkpoint_rollback(mem_, checkpoint_id)) {
        return false;
    }

    const llama_pos pos_max = llama_memory_seq_pos_max(mem_, 0);
    const llama_pos new_end = pos_max >= 0 ? pos_max + 1 : 0;
    rollback_turns_to_pos(new_end);
    return true;
}

llama_seq_id llama_session::fork(llama_seq_id parent_seq) {
    if (!mem_) {
        return -1;
    }
    return llama_memory_fork(mem_, parent_seq);
}

bool llama_session::merge(llama_seq_id winner) {
    if (!mem_) {
        return false;
    }
    const bool ok = llama_memory_merge(mem_, winner);
    if (ok) {
        // Merge rewrites seq 0 state from winner. Existing turn metadata is no longer trustworthy.
        turns_.clear();
    }
    return ok;
}

void llama_session::shift_turn_positions(llama_pos pivot, llama_pos delta) {
    for (auto & turn : turns_) {
        if (turn.p0 >= pivot) {
            turn.p0 = std::max<llama_pos>(0, turn.p0 - delta);
            turn.p1 = std::max<llama_pos>(turn.p0, turn.p1 - delta);
        }
    }
}

int32_t llama_session::trim_turn(int32_t turn_id) {
    if (!mem_) {
        return -1;
    }

    int idx = -1;
    for (int i = 0; i < (int) turns_.size(); ++i) {
        if (turns_[i].turn_id == turn_id) {
            idx = i;
            break;
        }
    }
    if (idx < 0) {
        return -1;
    }

    const llama_pos p0 = turns_[idx].p0;
    const llama_pos p1 = turns_[idx].p1;
    if (p1 <= p0) {
        turns_.erase(turns_.begin() + idx);
        return 0;
    }

    const int32_t removed = llama_memory_selective_trim(mem_, p0, p1, nullptr, nullptr, 0);
    if (removed < 0) {
        return -1;
    }

    turns_.erase(turns_.begin() + idx);
    if (removed > 0) {
        shift_turn_positions(p1, removed);
    }
    return removed;
}

int32_t llama_session::trim_turns(int32_t first_turn_id, int32_t last_turn_id) {
    if (!mem_) {
        return -1;
    }

    if (first_turn_id > last_turn_id) {
        std::swap(first_turn_id, last_turn_id);
    }

    bool found = false;
    llama_pos p0 = 0;
    llama_pos p1 = 0;
    for (const auto & turn : turns_) {
        if (turn.turn_id >= first_turn_id && turn.turn_id <= last_turn_id) {
            if (!found) {
                p0 = turn.p0;
                p1 = turn.p1;
                found = true;
            } else {
                p0 = std::min(p0, turn.p0);
                p1 = std::max(p1, turn.p1);
            }
        }
    }

    if (!found || p1 <= p0) {
        return -1;
    }

    const int32_t removed = llama_memory_selective_trim(mem_, p0, p1, nullptr, nullptr, 0);
    if (removed < 0) {
        return -1;
    }

    turns_.erase(
        std::remove_if(turns_.begin(), turns_.end(), [first_turn_id, last_turn_id](const llama_session_turn & turn) {
            return turn.turn_id >= first_turn_id && turn.turn_id <= last_turn_id;
        }),
        turns_.end());

    if (removed > 0) {
        shift_turn_positions(p1, removed);
    }
    return removed;
}

int32_t llama_session::after_promote() {
    if (!mem_ || params_.evict_threshold <= 0.0f) {
        return 0;
    }

    const uint32_t n_ctx = effective_n_ctx();
    if (n_ctx == 0) {
        return 0;
    }

    const int32_t max_nodes = std::max<int32_t>(1, (int32_t) std::floor(params_.evict_threshold * n_ctx));
    const int32_t node_count = llama_memory_prefix_node_count(mem_);
    if (node_count > max_nodes) {
        return llama_memory_prefix_evict_lru(mem_, max_nodes);
    }

    return 0;
}

bool llama_session::check_overflow() {
    if (!mem_) {
        return false;
    }

    const uint32_t n_ctx = effective_n_ctx();
    if (n_ctx == 0) {
        return false;
    }

    const llama_pos pos_max = llama_memory_seq_pos_max(mem_, 0);
    const uint32_t used = pos_max >= 0 ? (uint32_t) (pos_max + 1) : 0;
    if (used >= n_ctx) {
        llama_memory_clear(mem_, false);
        turns_.clear();
        return true;
    }
    return false;
}

float llama_session::kv_usage_ratio() const {
    if (!mem_) {
        return 0.0f;
    }

    const uint32_t n_ctx = effective_n_ctx();
    if (n_ctx == 0) {
        return 0.0f;
    }

    const llama_pos pos_max = llama_memory_seq_pos_max(mem_, 0);
    const uint32_t used = pos_max >= 0 ? (uint32_t) (pos_max + 1) : 0;
    return (float) used / (float) n_ctx;
}

int32_t llama_session::n_turns() const {
    return (int32_t) turns_.size();
}

llama_pos llama_session::current_pos() const {
    if (!mem_) {
        return -1;
    }
    return llama_memory_seq_pos_max(mem_, 0);
}

bool llama_session::get_turn(int32_t index, int32_t * turn_id_out, llama_pos * p0_out, llama_pos * p1_out) const {
    if (index < 0 || index >= (int32_t) turns_.size()) {
        return false;
    }

    const auto & turn = turns_[index];
    if (turn_id_out) {
        *turn_id_out = turn.turn_id;
    }
    if (p0_out) {
        *p0_out = turn.p0;
    }
    if (p1_out) {
        *p1_out = turn.p1;
    }
    return true;
}

