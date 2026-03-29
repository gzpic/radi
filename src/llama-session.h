#pragma once

#include "llama.h"

#include <vector>

struct llama_session_turn {
    int32_t turn_id = -1;
    llama_pos p0 = 0; // inclusive
    llama_pos p1 = 0; // exclusive
    std::vector<llama_token> tokens;
    bool ended = false;
};

class llama_session {
public:
    static llama_session_params default_params();

    llama_session(llama_context * ctx, llama_session_params params);

    int32_t turn_begin();
    bool turn_add_tokens(int32_t turn_id, const llama_token * tokens, int32_t n_tokens);
    bool turn_end(int32_t turn_id);

    int32_t checkpoint_save();
    bool checkpoint_rollback(int32_t checkpoint_id);
    llama_seq_id fork(llama_seq_id parent_seq);
    bool merge(llama_seq_id winner);
    int32_t trim_turn(int32_t turn_id);
    int32_t trim_turns(int32_t first_turn_id, int32_t last_turn_id);

    int32_t after_promote();
    bool check_overflow();

    float kv_usage_ratio() const;
    int32_t n_turns() const;
    llama_pos current_pos() const;
    bool get_turn(int32_t index, int32_t * turn_id_out, llama_pos * p0_out, llama_pos * p1_out) const;

private:
    llama_session_turn * find_turn(int32_t turn_id);
    const llama_session_turn * find_turn(int32_t turn_id) const;
    bool has_open_turn() const;
    uint32_t effective_n_ctx() const;
    void shift_turn_positions(llama_pos pivot, llama_pos delta);
    void rollback_turns_to_pos(llama_pos new_end_exclusive);

private:
    llama_context * ctx_ = nullptr;
    llama_memory_t mem_ = nullptr;
    llama_session_params params_ = default_params();
    std::vector<llama_session_turn> turns_;
    int32_t next_turn_id_ = 0;
};

