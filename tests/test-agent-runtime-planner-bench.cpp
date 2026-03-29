// Agent runtime planner benchmark (Baseline vs Enhanced)
//
// Simulates a realistic planning workflow with branching, tool failures,
// rollback, branch pruning, and final route selection.
//
// Modes:
//   - Baseline: stateless linear workflow (clear + full prefill each attempt)
//   - Enhanced: session runtime workflow (checkpoint/rollback + fork/merge + trim)
//
// Outputs:
//   - run-XX-events.jsonl
//   - planner-comparison-summary.json
//   - planner-comparison-summary.md
//
// Usage:
//   test-agent-runtime-planner-bench <model.gguf> [--runs N] [--out DIR]

#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

using hrclock = std::chrono::high_resolution_clock;

struct runtime_snapshot {
    int32_t   n_turns         = -1;
    llama_pos current_pos     = -1;
    llama_pos seq0_pos_max    = -1;
    float     kv_usage        = 0.0f;
    int32_t   prefix_nodes    = -1;
    int32_t   active_branches = 0;
};

struct mode_metrics {
    double  decode_ms_total    = 0.0;
    int64_t decode_tokens      = 0;

    int32_t rollback_count     = 0;
    int32_t fork_count         = 0;
    int32_t merge_count        = 0;
    int32_t prune_count        = 0;
    int32_t trim_count         = 0;

    int32_t checks_pass        = 0;
    int32_t checks_fail        = 0;

    float   peak_kv_usage      = 0.0f;
    int32_t peak_prefix_nodes  = 0;
};

struct run_metrics {
    mode_metrics baseline;
    mode_metrics enhanced;
};

struct planner_texts {
    std::string base_context;
    std::string path_a_fail;
    std::string path_b_eval;
    std::string path_c_eval;
    std::string scratch_note;
    std::string final_commit;
};

struct planner_tokens {
    // baseline: cumulative full prompts
    std::vector<llama_token> baseline_a;
    std::vector<llama_token> baseline_b;
    std::vector<llama_token> baseline_c;
    std::vector<llama_token> baseline_final;

    // enhanced: incremental segments
    std::vector<llama_token> base;
    std::vector<llama_token> a_fail;
    std::vector<llama_token> b_eval;
    std::vector<llama_token> c_eval;
    std::vector<llama_token> scratch_note;
    std::vector<llama_token> final_commit;
};

static double elapsed_ms(hrclock::time_point a, hrclock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

static std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 16);
    for (char c : s) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"':  out += "\\\""; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;      break;
        }
    }
    return out;
}

class event_logger {
public:
    explicit event_logger(const std::filesystem::path & path) : out(path) {}

    void write(
            int run,
            const std::string & mode,
            const std::string & step,
            const std::string & action,
            bool ok,
            const runtime_snapshot & s,
            double decode_ms,
            int32_t decoded_tokens,
            const std::string & note) {
        if (!out.good()) {
            return;
        }
        out
            << "{"
            << "\"run\":" << run << ","
            << "\"mode\":\"" << json_escape(mode) << "\","
            << "\"step\":\"" << json_escape(step) << "\","
            << "\"action\":\"" << json_escape(action) << "\","
            << "\"ok\":" << (ok ? "true" : "false") << ","
            << "\"decode_ms\":" << std::fixed << std::setprecision(3) << decode_ms << ","
            << "\"decoded_tokens\":" << decoded_tokens << ","
            << "\"n_turns\":" << s.n_turns << ","
            << "\"current_pos\":" << s.current_pos << ","
            << "\"seq0_pos_max\":" << s.seq0_pos_max << ","
            << "\"kv_usage\":" << std::setprecision(6) << s.kv_usage << ","
            << "\"prefix_nodes\":" << s.prefix_nodes << ","
            << "\"active_branches\":" << s.active_branches << ","
            << "\"note\":\"" << json_escape(note) << "\""
            << "}\n";
    }

private:
    std::ofstream out;
};

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text, bool add_special) {
    int n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), nullptr, 0, add_special, true);
    if (n < 0) {
        n = -n;
    }

    std::vector<llama_token> out(n);
    int r = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), out.data(), (int32_t) out.size(), add_special, true);
    if (r < 0) {
        return {};
    }
    out.resize(r);
    return out;
}

static bool decode_seq0(llama_context * ctx, const std::vector<llama_token> & toks) {
    if (toks.empty()) {
        return true;
    }
    llama_batch b = llama_batch_get_one(const_cast<llama_token *>(toks.data()), (int32_t) toks.size());
    return llama_decode(ctx, b) == 0;
}

static bool decode_on_seq(
        llama_context * ctx,
        const std::vector<llama_token> & toks,
        llama_seq_id seq_id,
        llama_pos pos0) {
    if (toks.empty()) {
        return true;
    }

    llama_batch b = llama_batch_init((int32_t) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) {
        b.token[i] = toks[i];
        b.pos[i] = pos0 + (llama_pos) i;
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = seq_id;
        b.logits[i] = (i + 1 == toks.size()) ? 1 : 0;
    }
    b.n_tokens = (int32_t) toks.size();

    const int rc = llama_decode(ctx, b);
    llama_batch_free(b);
    return rc == 0;
}

static llama_context * make_ctx(llama_model * model, uint32_t n_ctx, uint32_t n_seq_max) {
    auto cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx;
    cparams.n_batch = n_ctx;
    cparams.n_seq_max = n_seq_max;
    return llama_init_from_model(model, cparams);
}

static runtime_snapshot collect_snapshot(
        llama_memory_t mem,
        llama_session_t session,
        int32_t active_branches) {
    runtime_snapshot s;
    s.seq0_pos_max = llama_memory_seq_pos_max(mem, 0);
    s.prefix_nodes = llama_memory_prefix_node_count(mem);
    s.active_branches = active_branches;

    if (session != nullptr) {
        s.n_turns = llama_session_n_turns(session);
        s.current_pos = llama_session_current_pos(session);
        s.kv_usage = llama_session_kv_usage(session);
    }

    return s;
}

static void update_peaks(mode_metrics & m, const runtime_snapshot & s) {
    m.peak_kv_usage = std::max(m.peak_kv_usage, s.kv_usage);
    m.peak_prefix_nodes = std::max(m.peak_prefix_nodes, s.prefix_nodes);
}

static planner_texts build_planner_texts() {
    planner_texts t;

    t.base_context =
        "System: You are RoutePlannerAgent for a same-day city logistics task. "
        "You must plan a path from Hub-H0 to Site-S9 before 09:30, minimizing expected delay and cost. "
        "You can call tools: map_route(), traffic_probe(), weather_now(), toll_estimator(), safety_check(). "
        "Output final decision with route id, ETA, and fallback.\n"
        "User: Plan a robust route now. Constraints: avoid unsafe zones, budget <= 48, "
        "prefer arrival <= 09:20, keep one fallback branch.\n"
        "Context: Candidate trunks = Route-A(highway), Route-B(metro+walk), Route-C(ring-road+shuttle).";

    t.path_a_fail =
        "\nPlanner[Path-A]: Start with Route-A. "
        "Call traffic_probe(A12,A13) and toll_estimator(A12 corridor). "
        "Observation: A12 incident severity=high, probe timeout on segment A13, "
        "ETA uncertainty exceeds policy threshold. "
        "Decision: mark Path-A as invalid and rollback to pre-Path-A checkpoint.";

    t.path_b_eval =
        "\nPlanner[Path-B]: Evaluate metro transfer strategy. "
        "Tool results: map_route(B)=ETA 09:18, toll=0, transfer_risk=0.42, "
        "weather impact on walk segment=moderate rain, safety_check near station=pass. "
        "Decision: keep Path-B as feasible but penalize for transfer variance.";

    t.path_c_eval =
        "\nPlanner[Path-C]: Evaluate ring-road + shuttle strategy. "
        "Tool results: map_route(C)=ETA 09:12, toll=18, transfer_risk=0.12, "
        "weather impact=low, safety_check=pass, congestion trend stable. "
        "Decision: Path-C dominates Path-B on ETA and reliability.";

    t.scratch_note =
        "\nPlanner[Scratch]: remove discarded branch narrative from retained context.";

    t.final_commit =
        "\nFinalPlan: choose Route-C as primary path, keep Route-B as fallback only if shuttle delay > 8 min. "
        "Expected arrival 09:12, budget 18, reliability high. "
        "Execute Route-C now.";

    return t;
}

static planner_tokens build_planner_tokens(const llama_vocab * vocab, const planner_texts & t) {
    planner_tokens p;

    // Baseline cumulative prompts: each attempt replays full history from scratch.
    std::string b1 = t.base_context + t.path_a_fail;
    std::string b2 = b1 + t.path_b_eval;
    std::string b3 = b2 + t.path_c_eval;
    std::string b4 = b3 + t.final_commit;

    p.baseline_a = tokenize(vocab, b1, true);
    p.baseline_b = tokenize(vocab, b2, true);
    p.baseline_c = tokenize(vocab, b3, true);
    p.baseline_final = tokenize(vocab, b4, true);

    // Enhanced incremental segments.
    p.base = tokenize(vocab, t.base_context, true);
    p.a_fail = tokenize(vocab, t.path_a_fail, false);
    p.b_eval = tokenize(vocab, t.path_b_eval, false);
    p.c_eval = tokenize(vocab, t.path_c_eval, false);
    p.scratch_note = tokenize(vocab, t.scratch_note, false);
    p.final_commit = tokenize(vocab, t.final_commit, false);

    return p;
}

static bool check_and_log(
        mode_metrics & m,
        event_logger & lg,
        int run,
        const std::string & mode,
        const std::string & step,
        const std::string & action,
        bool cond,
        const runtime_snapshot & snap,
        const std::string & note) {
    if (cond) {
        m.checks_pass++;
    } else {
        m.checks_fail++;
    }
    lg.write(run, mode, step, action, cond, snap, 0.0, 0, note);
    return cond;
}

static mode_metrics run_baseline_once(
        llama_model * model,
        const planner_tokens & toks,
        int run,
        event_logger & lg,
        bool & ok) {
    mode_metrics m;
    ok = true;

    llama_context * ctx = make_ctx(model, 4096, 1);
    if (!ctx) {
        ok = false;
        return m;
    }

    llama_memory_t mem = llama_get_memory(ctx);

    auto run_step = [&](const char * step, const std::vector<llama_token> & v) {
        llama_memory_clear(mem, true);

        auto t0 = hrclock::now();
        bool d_ok = decode_seq0(ctx, v);
        auto t1 = hrclock::now();
        const double ms = elapsed_ms(t0, t1);

        runtime_snapshot s = collect_snapshot(mem, nullptr, 0);
        update_peaks(m, s);

        lg.write(run, "baseline", step, "clear+full_prefill", d_ok, s, ms, (int32_t) v.size(),
            d_ok ? "decoded cumulative prompt from scratch" : "decode failed");

        if (!d_ok) {
            ok = false;
        }

        m.decode_ms_total += ms;
        m.decode_tokens += (int64_t) v.size();
    };

    run_step("attempt_path_a_fail", toks.baseline_a);
    run_step("attempt_path_b", toks.baseline_b);
    run_step("attempt_path_c", toks.baseline_c);
    run_step("final_commit", toks.baseline_final);

    llama_free(ctx);
    return m;
}

static mode_metrics run_enhanced_once(
        llama_model * model,
        const planner_tokens & toks,
        int run,
        event_logger & lg,
        bool & ok) {
    mode_metrics m;
    ok = true;

    llama_context * ctx = make_ctx(model, 2048, 4);
    if (!ctx) {
        ok = false;
        return m;
    }

    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_clear(mem, true);
    llama_memory_prefix_cache_enable(mem);

    llama_session_params sp = llama_session_default_params();
    sp.n_ctx = llama_n_ctx(ctx);
    llama_session_t s = llama_session_init(ctx, sp);
    if (!s) {
        llama_free(ctx);
        ok = false;
        return m;
    }

    int32_t active_branches = 0;
    llama_pos root_pos = -1;

    auto add_decode_metric = [&](double ms, int32_t n_tokens) {
        m.decode_ms_total += ms;
        m.decode_tokens += n_tokens;
    };

    // Step 1: base turn on seq0
    int32_t t_base = llama_session_turn_begin(s);
    llama_session_turn_add_tokens(s, t_base, toks.base.data(), (int32_t) toks.base.size());
    auto t0 = hrclock::now();
    bool d0_ok = decode_seq0(ctx, toks.base);
    auto t1 = hrclock::now();
    const double d0_ms = elapsed_ms(t0, t1);
    add_decode_metric(d0_ms, (int32_t) toks.base.size());
    llama_session_turn_end(s, t_base);
    root_pos = llama_session_current_pos(s);

    runtime_snapshot snap = collect_snapshot(mem, s, active_branches);
    update_peaks(m, snap);
    lg.write(run, "enhanced", "base_context", "decode_seq0+turn_end", d0_ok, snap, d0_ms, (int32_t) toks.base.size(),
        "seed planning root context");
    if (!d0_ok) {
        ok = false;
    }

    check_and_log(m, lg, run, "enhanced", "base_context", "state_check",
        snap.n_turns == 1, snap, "n_turns should be 1 after base turn");
    check_and_log(m, lg, run, "enhanced", "base_context", "state_check",
        snap.current_pos == snap.seq0_pos_max, snap, "session current_pos should match seq0 pos max");

    // Step 2: checkpoint root
    const int32_t cp_root = llama_session_checkpoint_save(s);
    snap = collect_snapshot(mem, s, active_branches);
    update_peaks(m, snap);
    check_and_log(m, lg, run, "enhanced", "checkpoint_root", "checkpoint_save",
        cp_root >= 0, snap, "checkpoint id must be valid");

    // Step 3: path A failed attempt on seq0, then rollback
    int32_t t_fail = llama_session_turn_begin(s);
    llama_session_turn_add_tokens(s, t_fail, toks.a_fail.data(), (int32_t) toks.a_fail.size());
    t0 = hrclock::now();
    bool d_fail_ok = decode_seq0(ctx, toks.a_fail);
    t1 = hrclock::now();
    const double d_fail_ms = elapsed_ms(t0, t1);
    add_decode_metric(d_fail_ms, (int32_t) toks.a_fail.size());
    llama_session_turn_end(s, t_fail);

    snap = collect_snapshot(mem, s, active_branches);
    update_peaks(m, snap);
    lg.write(run, "enhanced", "path_a_failed", "decode_seq0+turn_end", d_fail_ok, snap, d_fail_ms, (int32_t) toks.a_fail.size(),
        "simulate tool failure on path A");
    if (!d_fail_ok) {
        ok = false;
    }

    check_and_log(m, lg, run, "enhanced", "path_a_failed", "state_check",
        snap.current_pos > root_pos, snap, "failed attempt should advance seq0 position");

    const bool rb_ok = llama_session_checkpoint_rollback(s, cp_root);
    if (rb_ok) {
        m.rollback_count++;
    }
    snap = collect_snapshot(mem, s, active_branches);
    update_peaks(m, snap);
    check_and_log(m, lg, run, "enhanced", "rollback_path_a", "checkpoint_rollback",
        rb_ok, snap, "rollback to root checkpoint should succeed");
    check_and_log(m, lg, run, "enhanced", "rollback_path_a", "state_check",
        snap.current_pos == root_pos, snap, "current_pos should return to root after rollback");
    check_and_log(m, lg, run, "enhanced", "rollback_path_a", "state_check",
        snap.n_turns == 1, snap, "rollback should remove failed turn metadata");

    // Step 4: fork path B and decode branch B
    llama_seq_id b = llama_session_fork(s, 0);
    if (b >= 0) {
        m.fork_count++;
        active_branches++;
    }
    snap = collect_snapshot(mem, s, active_branches);
    update_peaks(m, snap);
    check_and_log(m, lg, run, "enhanced", "fork_path_b", "fork",
        b >= 0, snap, "fork path B should return valid seq id");
    check_and_log(m, lg, run, "enhanced", "fork_path_b", "state_check",
        (b >= 0) && (llama_memory_seq_pos_max(mem, b) == root_pos), snap,
        "forked branch B should start at root position");

    llama_pos pos_b0 = llama_memory_seq_pos_max(mem, b) + 1;
    t0 = hrclock::now();
    bool d_b_ok = decode_on_seq(ctx, toks.b_eval, b, pos_b0);
    t1 = hrclock::now();
    const double d_b_ms = elapsed_ms(t0, t1);
    add_decode_metric(d_b_ms, (int32_t) toks.b_eval.size());
    snap = collect_snapshot(mem, s, active_branches);
    update_peaks(m, snap);
    lg.write(run, "enhanced", "decode_path_b", "decode_branch_seq", d_b_ok, snap, d_b_ms, (int32_t) toks.b_eval.size(),
        "evaluate feasible but weaker candidate");
    if (!d_b_ok) {
        ok = false;
    }

    // Step 5: fork path C and decode branch C
    llama_seq_id c = llama_session_fork(s, 0);
    if (c >= 0) {
        m.fork_count++;
        active_branches++;
    }
    snap = collect_snapshot(mem, s, active_branches);
    update_peaks(m, snap);
    check_and_log(m, lg, run, "enhanced", "fork_path_c", "fork",
        c >= 0, snap, "fork path C should return valid seq id");
    check_and_log(m, lg, run, "enhanced", "fork_path_c", "state_check",
        (c >= 0) && (llama_memory_seq_pos_max(mem, c) == root_pos), snap,
        "forked branch C should start at root position");

    llama_pos pos_c0 = llama_memory_seq_pos_max(mem, c) + 1;
    t0 = hrclock::now();
    bool d_c_ok = decode_on_seq(ctx, toks.c_eval, c, pos_c0);
    t1 = hrclock::now();
    const double d_c_ms = elapsed_ms(t0, t1);
    add_decode_metric(d_c_ms, (int32_t) toks.c_eval.size());
    const llama_pos c_pos_end_before_merge = llama_memory_seq_pos_max(mem, c);
    snap = collect_snapshot(mem, s, active_branches);
    update_peaks(m, snap);
    lg.write(run, "enhanced", "decode_path_c", "decode_branch_seq", d_c_ok, snap, d_c_ms, (int32_t) toks.c_eval.size(),
        "evaluate dominant candidate");
    if (!d_c_ok) {
        ok = false;
    }

    // Step 6: prune invalid/weak path B
    const bool prune_b_ok = llama_memory_seq_rm(mem, b, 0, -1);
    if (prune_b_ok) {
        m.prune_count++;
        active_branches = std::max(0, active_branches - 1);
    }
    snap = collect_snapshot(mem, s, active_branches);
    update_peaks(m, snap);
    check_and_log(m, lg, run, "enhanced", "prune_path_b", "seq_rm_branch",
        prune_b_ok, snap, "pruning path B should succeed");
    check_and_log(m, lg, run, "enhanced", "prune_path_b", "state_check",
        llama_memory_seq_pos_max(mem, b) == -1, snap, "pruned path B must be empty");

    // Step 7: merge winner path C into seq0
    const bool merge_ok = llama_session_merge(s, c);
    if (merge_ok) {
        m.merge_count++;
        active_branches = 0;
    }
    snap = collect_snapshot(mem, s, active_branches);
    update_peaks(m, snap);
    check_and_log(m, lg, run, "enhanced", "merge_path_c", "merge",
        merge_ok, snap, "merge winner C should succeed");
    check_and_log(m, lg, run, "enhanced", "merge_path_c", "state_check",
        llama_memory_seq_pos_max(mem, 0) >= c_pos_end_before_merge, snap,
        "seq0 should include winner C state after merge");
    check_and_log(m, lg, run, "enhanced", "merge_path_c", "state_check",
        llama_memory_seq_pos_max(mem, c) == -1, snap,
        "winner branch id should be normalized away after merge");
    check_and_log(m, lg, run, "enhanced", "merge_path_c", "state_check",
        snap.n_turns == 0, snap,
        "current MVP semantics: merge clears turn metadata");

    // Step 8: add and trim scratch turn
    const llama_pos pos_before_scratch = snap.current_pos;
    int32_t t_scratch = llama_session_turn_begin(s);
    llama_session_turn_add_tokens(s, t_scratch, toks.scratch_note.data(), (int32_t) toks.scratch_note.size());
    t0 = hrclock::now();
    bool d_scratch_ok = decode_seq0(ctx, toks.scratch_note);
    t1 = hrclock::now();
    const double d_scratch_ms = elapsed_ms(t0, t1);
    add_decode_metric(d_scratch_ms, (int32_t) toks.scratch_note.size());
    llama_session_turn_end(s, t_scratch);
    snap = collect_snapshot(mem, s, active_branches);
    update_peaks(m, snap);
    lg.write(run, "enhanced", "scratch_turn", "decode_seq0+turn_end", d_scratch_ok, snap, d_scratch_ms, (int32_t) toks.scratch_note.size(),
        "temporary planning note to be trimmed");
    if (!d_scratch_ok) {
        ok = false;
    }

    const int32_t removed = llama_session_trim_turn(s, t_scratch);
    if (removed > 0) {
        m.trim_count++;
    }
    snap = collect_snapshot(mem, s, active_branches);
    update_peaks(m, snap);
    check_and_log(m, lg, run, "enhanced", "trim_scratch", "trim_turn",
        removed > 0, snap, "scratch turn should be removed");
    check_and_log(m, lg, run, "enhanced", "trim_scratch", "state_check",
        snap.current_pos <= pos_before_scratch, snap,
        "current_pos should not grow after trimming scratch turn");

    // Step 9: final commit turn on seq0
    int32_t t_final = llama_session_turn_begin(s);
    llama_session_turn_add_tokens(s, t_final, toks.final_commit.data(), (int32_t) toks.final_commit.size());
    t0 = hrclock::now();
    bool d_final_ok = decode_seq0(ctx, toks.final_commit);
    t1 = hrclock::now();
    const double d_final_ms = elapsed_ms(t0, t1);
    add_decode_metric(d_final_ms, (int32_t) toks.final_commit.size());
    llama_session_turn_end(s, t_final);
    snap = collect_snapshot(mem, s, active_branches);
    update_peaks(m, snap);
    lg.write(run, "enhanced", "final_commit", "decode_seq0+turn_end", d_final_ok, snap, d_final_ms, (int32_t) toks.final_commit.size(),
        "final selected route is Route-C");
    if (!d_final_ok) {
        ok = false;
    }
    check_and_log(m, lg, run, "enhanced", "final_commit", "state_check",
        snap.n_turns == 1, snap, "final state should keep one committed turn");

    // Optional maintenance hooks
    llama_session_after_promote(s);
    llama_session_check_overflow(s);

    llama_session_free(s);
    llama_free(ctx);
    return m;
}

static double mean(const std::vector<double> & v) {
    if (v.empty()) return 0.0;
    double s = 0.0;
    for (double x : v) s += x;
    return s / v.size();
}

static double stddev(const std::vector<double> & v) {
    if (v.size() < 2) return 0.0;
    const double m = mean(v);
    double acc = 0.0;
    for (double x : v) {
        const double d = x - m;
        acc += d*d;
    }
    return std::sqrt(acc / v.size());
}

static std::string ts_now() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tmv{};
#if defined(_WIN32)
    localtime_s(&tmv, &t);
#else
    localtime_r(&t, &tmv);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tmv, "%Y%m%d-%H%M%S");
    return oss.str();
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: %s <model.gguf> [--runs N] [--out DIR]\n", argv[0]);
        return 1;
    }

    const char * model_path = argv[1];
    int runs = 5;
    std::filesystem::path out_dir = std::filesystem::path("artifacts") / "agent-runtime-comparison" / ts_now();

    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--runs" && i + 1 < argc) {
            runs = std::max(1, std::atoi(argv[++i]));
        } else if (arg == "--out" && i + 1 < argc) {
            out_dir = argv[++i];
        } else {
            std::fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            return 1;
        }
    }

    std::filesystem::create_directories(out_dir);

    std::printf("=== Agent Runtime Planner Benchmark ===\n");
    std::printf("Model: %s\n", model_path);
    std::printf("Runs:  %d\n", runs);
    std::printf("Out:   %s\n\n", out_dir.string().c_str());

    llama_backend_init();

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        std::fprintf(stderr, "Failed to load model: %s\n", model_path);
        llama_backend_free();
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const planner_texts texts = build_planner_texts();
    const planner_tokens toks = build_planner_tokens(vocab, texts);

    if (toks.base.empty() || toks.baseline_final.empty()) {
        std::fprintf(stderr, "Tokenization failed; abort.\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    std::vector<double> base_ms_all, enh_ms_all, base_tok_all, enh_tok_all;
    int32_t total_checks_pass = 0;
    int32_t total_checks_fail = 0;

    for (int run = 1; run <= runs; ++run) {
        std::ostringstream evname;
        evname << "run-" << std::setw(2) << std::setfill('0') << run << "-events.jsonl";
        event_logger lg(out_dir / evname.str());

        std::printf("[Run %d/%d] baseline...\n", run, runs);
        bool base_ok = true;
        mode_metrics base = run_baseline_once(model, toks, run, lg, base_ok);

        std::printf("[Run %d/%d] enhanced...\n", run, runs);
        bool enh_ok = true;
        mode_metrics enh = run_enhanced_once(model, toks, run, lg, enh_ok);

        base_ms_all.push_back(base.decode_ms_total);
        enh_ms_all.push_back(enh.decode_ms_total);
        base_tok_all.push_back((double) base.decode_tokens);
        enh_tok_all.push_back((double) enh.decode_tokens);

        total_checks_pass += enh.checks_pass;
        total_checks_fail += enh.checks_fail;

        const double tok_gain = (base.decode_tokens > 0)
            ? 100.0 * (1.0 - (double) enh.decode_tokens / (double) base.decode_tokens)
            : 0.0;
        const double ms_gain = (base.decode_ms_total > 0.0)
            ? 100.0 * (1.0 - enh.decode_ms_total / base.decode_ms_total)
            : 0.0;

        std::printf(
            "[Run %d] baseline=%.1f ms (%lld tok), enhanced=%.1f ms (%lld tok), "
            "token_reduction=%.2f%%, time_reduction=%.2f%%, checks=%d/%d\n\n",
            run,
            base.decode_ms_total, (long long) base.decode_tokens,
            enh.decode_ms_total, (long long) enh.decode_tokens,
            tok_gain, ms_gain,
            enh.checks_pass, enh.checks_pass + enh.checks_fail);

        if (!base_ok || !enh_ok) {
            std::fprintf(stderr, "Run %d had decode failures (base_ok=%d, enh_ok=%d)\n", run, base_ok ? 1 : 0, enh_ok ? 1 : 0);
        }
    }

    const double base_ms_mean = mean(base_ms_all);
    const double enh_ms_mean  = mean(enh_ms_all);
    const double base_ms_std  = stddev(base_ms_all);
    const double enh_ms_std   = stddev(enh_ms_all);

    const double base_tok_mean = mean(base_tok_all);
    const double enh_tok_mean  = mean(enh_tok_all);
    const double base_tok_std  = stddev(base_tok_all);
    const double enh_tok_std   = stddev(enh_tok_all);

    const double token_reduction_pct = (base_tok_mean > 0.0)
        ? 100.0 * (1.0 - enh_tok_mean / base_tok_mean) : 0.0;
    const double time_reduction_pct = (base_ms_mean > 0.0)
        ? 100.0 * (1.0 - enh_ms_mean / base_ms_mean) : 0.0;

    // JSON summary
    {
        std::ofstream js(out_dir / "planner-comparison-summary.json");
        js << "{\n";
        js << "  \"model\": \"" << json_escape(model_path) << "\",\n";
        js << "  \"runs\": " << runs << ",\n";
        js << "  \"baseline\": {\n";
        js << "    \"decode_ms_mean\": " << std::fixed << std::setprecision(3) << base_ms_mean << ",\n";
        js << "    \"decode_ms_std\": " << base_ms_std << ",\n";
        js << "    \"decode_tokens_mean\": " << base_tok_mean << ",\n";
        js << "    \"decode_tokens_std\": " << base_tok_std << "\n";
        js << "  },\n";
        js << "  \"enhanced\": {\n";
        js << "    \"decode_ms_mean\": " << enh_ms_mean << ",\n";
        js << "    \"decode_ms_std\": " << enh_ms_std << ",\n";
        js << "    \"decode_tokens_mean\": " << enh_tok_mean << ",\n";
        js << "    \"decode_tokens_std\": " << enh_tok_std << ",\n";
        js << "    \"checks_pass\": " << total_checks_pass << ",\n";
        js << "    \"checks_fail\": " << total_checks_fail << "\n";
        js << "  },\n";
        js << "  \"improvement\": {\n";
        js << "    \"token_reduction_pct\": " << token_reduction_pct << ",\n";
        js << "    \"time_reduction_pct\": " << time_reduction_pct << "\n";
        js << "  }\n";
        js << "}\n";
    }

    // Markdown summary
    {
        std::ofstream md(out_dir / "planner-comparison-summary.md");
        md << "# Agent Runtime Planner Comparison\n\n";
        md << "- Model: `" << model_path << "`\n";
        md << "- Runs: `" << runs << "`\n\n";

        md << "## Aggregate Metrics\n\n";
        md << "| Metric | Baseline | Enhanced | Improvement |\n";
        md << "|---|---:|---:|---:|\n";
        md << "| Decode time mean (ms) | " << std::fixed << std::setprecision(3) << base_ms_mean
           << " ± " << base_ms_std << " | " << enh_ms_mean << " ± " << enh_ms_std
           << " | " << time_reduction_pct << "% |\n";
        md << "| Decoded tokens mean | " << base_tok_mean << " ± " << base_tok_std
           << " | " << enh_tok_mean << " ± " << enh_tok_std
           << " | " << token_reduction_pct << "% |\n";
        md << "| Enhanced state checks | - | " << total_checks_pass << " pass / " << total_checks_fail
           << " fail | - |\n\n";

        md << "## Scenario Notes\n\n";
        md << "- Baseline replays full cumulative prompt from scratch for each path attempt.\n";
        md << "- Enhanced executes: base context -> path-A failure -> rollback -> fork(B/C) -> prune B -> merge C -> trim scratch -> final commit.\n";
        md << "- Per-step trace and state snapshots are in `run-XX-events.jsonl` files.\n";
    }

    std::printf("=== Summary ===\n");
    std::printf("Baseline decode: %.1f ± %.1f ms, %.1f ± %.1f tokens\n", base_ms_mean, base_ms_std, base_tok_mean, base_tok_std);
    std::printf("Enhanced decode: %.1f ± %.1f ms, %.1f ± %.1f tokens\n", enh_ms_mean, enh_ms_std, enh_tok_mean, enh_tok_std);
    std::printf("Token reduction: %.2f%%\n", token_reduction_pct);
    std::printf("Time reduction:  %.2f%%\n", time_reduction_pct);
    std::printf("Enhanced checks: %d pass / %d fail\n", total_checks_pass, total_checks_fail);
    std::printf("Artifacts: %s\n", out_dir.string().c_str());

    llama_model_free(model);
    llama_backend_free();
    return (total_checks_fail == 0) ? 0 : 2;
}

