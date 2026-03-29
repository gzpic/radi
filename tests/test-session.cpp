// Session Runtime MVP smoke/unit tests (model-backed)
//
// Usage:
//   test-session <model.gguf>

#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

static int g_pass = 0;
static int g_fail = 0;

#define ASSERT_TRUE(x) do { if (!(x)) { throw std::runtime_error(std::string("ASSERT_TRUE failed: ") + #x + " at line " + std::to_string(__LINE__)); } } while (0)
#define ASSERT_EQ(a, b) do { auto _a = (a); auto _b = (b); if (_a != _b) { throw std::runtime_error(std::string("ASSERT_EQ failed at line ") + std::to_string(__LINE__)); } } while (0)
#define ASSERT_GE(a, b) do { auto _a = (a); auto _b = (b); if (!(_a >= _b)) { throw std::runtime_error(std::string("ASSERT_GE failed at line ") + std::to_string(__LINE__)); } } while (0)
#define ASSERT_LE(a, b) do { auto _a = (a); auto _b = (b); if (!(_a <= _b)) { throw std::runtime_error(std::string("ASSERT_LE failed at line ") + std::to_string(__LINE__)); } } while (0)

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text, bool add_special) {
    int n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), nullptr, 0, add_special, true);
    if (n < 0) n = -n;
    std::vector<llama_token> out(n);
    int r = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), out.data(), (int32_t) out.size(), add_special, true);
    if (r < 0) {
        return {};
    }
    out.resize(r);
    return out;
}

static bool decode_tokens(llama_context * ctx, const std::vector<llama_token> & toks) {
    if (toks.empty()) {
        return true;
    }
    llama_batch batch = llama_batch_get_one(const_cast<llama_token *>(toks.data()), (int32_t) toks.size());
    return llama_decode(ctx, batch) == 0;
}

static bool decode_tokens_on_seq(
        llama_context * ctx,
        const std::vector<llama_token> & toks,
        llama_seq_id seq,
        llama_pos pos0) {
    if (toks.empty()) {
        return true;
    }

    llama_batch batch = llama_batch_init((int32_t) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) {
        batch.token[i] = toks[i];
        batch.pos[i] = pos0 + (llama_pos) i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = seq;
        batch.logits[i] = (i + 1 == toks.size()) ? 1 : 0;
    }
    batch.n_tokens = (int32_t) toks.size();

    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    return rc == 0;
}

static llama_context * make_ctx(llama_model * model, uint32_t n_ctx, uint32_t n_seq_max) {
    auto cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx;
    cparams.n_batch = n_ctx;
    cparams.n_seq_max = n_seq_max;
    return llama_init_from_model(model, cparams);
}

static void run_test(const char * name, const std::function<void()> & fn) {
    std::printf("  TEST %-44s ", name);
    try {
        fn();
        std::printf("[PASS]\n");
        ++g_pass;
    } catch (const std::exception & e) {
        std::printf("[FAIL] %s\n", e.what());
        ++g_fail;
    } catch (...) {
        std::printf("[FAIL] unknown exception\n");
        ++g_fail;
    }
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    const char * model_path = argv[1];
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

    std::printf("=== Session Runtime MVP Tests ===\n");

    run_test("turn + checkpoint + rollback", [&]() {
        llama_context * ctx = make_ctx(model, 512, 4);
        ASSERT_TRUE(ctx != nullptr);

        llama_memory_t mem = llama_get_memory(ctx);
        llama_memory_clear(mem, true);

        llama_session_params sp = llama_session_default_params();
        sp.n_ctx = llama_n_ctx(ctx);
        llama_session_t s = llama_session_init(ctx, sp);
        ASSERT_TRUE(s != nullptr);

        const auto t0_tokens = tokenize(vocab, "System: You are a helpful agent.", true);
        ASSERT_TRUE(!t0_tokens.empty());

        int32_t t0 = llama_session_turn_begin(s);
        ASSERT_GE(t0, 0);
        llama_session_turn_add_tokens(s, t0, t0_tokens.data(), (int32_t) t0_tokens.size());
        ASSERT_TRUE(decode_tokens(ctx, t0_tokens));
        llama_session_turn_end(s, t0);

        ASSERT_EQ(llama_session_n_turns(s), 1);

        int32_t cp = llama_session_checkpoint_save(s);
        ASSERT_GE(cp, 0);

        const auto t1_tokens = tokenize(vocab, "User: calculate 1 + 1.", false);
        ASSERT_TRUE(!t1_tokens.empty());
        int32_t t1 = llama_session_turn_begin(s);
        ASSERT_GE(t1, 0);
        llama_session_turn_add_tokens(s, t1, t1_tokens.data(), (int32_t) t1_tokens.size());
        ASSERT_TRUE(decode_tokens(ctx, t1_tokens));
        llama_session_turn_end(s, t1);

        ASSERT_EQ(llama_session_n_turns(s), 2);
        ASSERT_TRUE(llama_session_checkpoint_rollback(s, cp));
        ASSERT_EQ(llama_session_n_turns(s), 1);

        int32_t turn_id = -1;
        llama_pos p0 = -1, p1 = -1;
        ASSERT_TRUE(llama_session_get_turn(s, 0, &turn_id, &p0, &p1));
        ASSERT_EQ(turn_id, t0);
        ASSERT_EQ(llama_session_current_pos(s), p1 - 1);

        llama_session_free(s);
        llama_free(ctx);
    });

    run_test("trim_turn + trim_turns", [&]() {
        llama_context * ctx = make_ctx(model, 512, 4);
        ASSERT_TRUE(ctx != nullptr);

        llama_memory_t mem = llama_get_memory(ctx);
        llama_memory_clear(mem, true);

        llama_session_params sp = llama_session_default_params();
        sp.n_ctx = llama_n_ctx(ctx);
        llama_session_t s = llama_session_init(ctx, sp);
        ASSERT_TRUE(s != nullptr);

        const auto a = tokenize(vocab, "A: first turn.", true);
        const auto b = tokenize(vocab, "B: second turn.", false);
        const auto c = tokenize(vocab, "C: third turn.", false);
        ASSERT_TRUE(!a.empty() && !b.empty() && !c.empty());

        int32_t t0 = llama_session_turn_begin(s);
        llama_session_turn_add_tokens(s, t0, a.data(), (int32_t) a.size());
        ASSERT_TRUE(decode_tokens(ctx, a));
        llama_session_turn_end(s, t0);

        int32_t t1 = llama_session_turn_begin(s);
        llama_session_turn_add_tokens(s, t1, b.data(), (int32_t) b.size());
        ASSERT_TRUE(decode_tokens(ctx, b));
        llama_session_turn_end(s, t1);

        int32_t t2 = llama_session_turn_begin(s);
        llama_session_turn_add_tokens(s, t2, c.data(), (int32_t) c.size());
        ASSERT_TRUE(decode_tokens(ctx, c));
        llama_session_turn_end(s, t2);

        ASSERT_EQ(llama_session_n_turns(s), 3);

        int32_t removed = llama_session_trim_turn(s, t1);
        ASSERT_TRUE(removed > 0);
        ASSERT_EQ(llama_session_n_turns(s), 2);

        int32_t id0 = -1, id1 = -1;
        llama_pos p00 = -1, p01 = -1, p10 = -1, p11 = -1;
        ASSERT_TRUE(llama_session_get_turn(s, 0, &id0, &p00, &p01));
        ASSERT_TRUE(llama_session_get_turn(s, 1, &id1, &p10, &p11));
        ASSERT_EQ(id0, t0);
        ASSERT_EQ(id1, t2);
        ASSERT_LE(p01, p10);

        ASSERT_TRUE(llama_session_trim_turns(s, t0, t2) > 0);
        ASSERT_EQ(llama_session_n_turns(s), 0);

        llama_session_free(s);
        llama_free(ctx);
    });

    run_test("fork + merge (n_seq_max > 1)", [&]() {
        llama_context * ctx = make_ctx(model, 512, 4);
        ASSERT_TRUE(ctx != nullptr);

        llama_memory_t mem = llama_get_memory(ctx);
        llama_memory_clear(mem, true);

        llama_session_params sp = llama_session_default_params();
        sp.n_ctx = llama_n_ctx(ctx);
        llama_session_t s = llama_session_init(ctx, sp);
        ASSERT_TRUE(s != nullptr);

        const auto prefix = tokenize(vocab, "Plan: evaluate two options.", true);
        const auto suffix = tokenize(vocab, "Option branch extra reasoning.", false);
        ASSERT_TRUE(!prefix.empty() && !suffix.empty());

        int32_t t0 = llama_session_turn_begin(s);
        ASSERT_GE(t0, 0);
        llama_session_turn_add_tokens(s, t0, prefix.data(), (int32_t) prefix.size());
        ASSERT_TRUE(decode_tokens(ctx, prefix));
        llama_session_turn_end(s, t0);

        llama_seq_id branch = llama_session_fork(s, 0);
        ASSERT_GE(branch, 0);

        llama_pos pos0 = llama_memory_seq_pos_max(mem, 0) + 1;
        ASSERT_TRUE(decode_tokens_on_seq(ctx, suffix, branch, pos0));

        ASSERT_TRUE(llama_session_merge(s, branch));
        ASSERT_EQ(llama_session_n_turns(s), 0); // current MVP clears metadata after merge
        ASSERT_GE(llama_memory_seq_pos_max(mem, 0), pos0 + (llama_pos) suffix.size() - 1);

        llama_session_free(s);
        llama_free(ctx);
    });

    run_test("after_promote + overflow", [&]() {
        llama_context * ctx = make_ctx(model, 512, 4);
        ASSERT_TRUE(ctx != nullptr);

        llama_memory_t mem = llama_get_memory(ctx);
        llama_memory_clear(mem, true);
        llama_memory_prefix_cache_enable(mem);

        // Build some prefix entries first
        const auto tok = tokenize(vocab, "Prefix cache population turn.", true);
        ASSERT_TRUE(!tok.empty());
        ASSERT_TRUE(decode_tokens(ctx, tok));

        llama_session_params sp = llama_session_default_params();
        sp.evict_threshold = 0.01f; // aggressive, should cap around floor(0.01 * 16) => 1 (with n_ctx override)
        sp.n_ctx = 16;
        llama_session_t s = llama_session_init(ctx, sp);
        ASSERT_TRUE(s != nullptr);

        const int32_t before = llama_memory_prefix_node_count(mem);
        llama_session_after_promote(s);
        const int32_t after = llama_memory_prefix_node_count(mem);
        ASSERT_LE(after, before);
        ASSERT_LE(after, 1);

        // Force overflow by using a tiny session n_ctx override
        llama_memory_clear(mem, true);
        llama_session_free(s);

        llama_session_params sp2 = llama_session_default_params();
        sp2.n_ctx = 1;
        llama_session_t s2 = llama_session_init(ctx, sp2);
        ASSERT_TRUE(s2 != nullptr);

        const auto one = tokenize(vocab, "X", true);
        ASSERT_TRUE(!one.empty());
        int32_t t = llama_session_turn_begin(s2);
        ASSERT_GE(t, 0);
        llama_session_turn_add_tokens(s2, t, one.data(), (int32_t) one.size());
        ASSERT_TRUE(decode_tokens(ctx, one));
        llama_session_turn_end(s2, t);
        ASSERT_TRUE(llama_session_n_turns(s2) >= 1);

        llama_session_check_overflow(s2);
        ASSERT_EQ(llama_session_n_turns(s2), 0);
        ASSERT_EQ(llama_session_current_pos(s2), -1);

        llama_session_free(s2);
        llama_free(ctx);
    });

    std::printf("\nSummary: pass=%d fail=%d\n", g_pass, g_fail);

    llama_model_free(model);
    llama_backend_free();

    return g_fail == 0 ? 0 : 1;
}
