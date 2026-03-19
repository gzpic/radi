// Inference Consistency Test for Radix Tree Prefix Cache
//
// Verifies that enabling the prefix cache does NOT change inference results.
// Compares logits (and greedy-decoded tokens) between:
//   - Baseline: fresh decode without any prefix cache reuse
//   - Cached:   decode that reuses prefix from a prior computation
//
// If prefix cache reuse changes logits, the KV data being reused is stale or
// incorrectly mapped — this is a critical correctness bug.
//
// Usage: test-radix-tree-consistency <model.gguf>

#include "llama.h"
#include "llama-kv-cache.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

// ============================================================================
// Helpers
// ============================================================================

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text, bool add_special) {
    int n = llama_tokenize(vocab, text.c_str(), text.length(), nullptr, 0, add_special, true);
    if (n < 0) { n = -n; }
    std::vector<llama_token> tokens(n);
    int ret = llama_tokenize(vocab, text.c_str(), text.length(), tokens.data(), tokens.size(), add_special, true);
    if (ret < 0) {
        fprintf(stderr, "tokenize failed: %d\n", ret);
        return {};
    }
    tokens.resize(ret);
    return tokens;
}

struct logits_snapshot {
    std::vector<float> data;
    int32_t n_vocab = 0;

    void capture(llama_context * ctx, int32_t idx) {
        const llama_model * model = llama_get_model(ctx);
        const llama_vocab * vocab = llama_model_get_vocab(model);
        n_vocab = llama_vocab_n_tokens(vocab);
        float * logits = llama_get_logits_ith(ctx, idx);
        if (logits) {
            data.assign(logits, logits + n_vocab);
        }
    }

    int32_t argmax() const {
        if (data.empty()) return -1;
        return (int32_t)(std::max_element(data.begin(), data.end()) - data.begin());
    }
};

struct logits_comparison {
    float   max_abs_diff = 0.0f;
    float   mean_abs_diff = 0.0f;
    double  cosine_sim = 0.0;
    bool    argmax_match = false;
    int32_t argmax_a = -1;
    int32_t argmax_b = -1;

    void compute(const logits_snapshot & a, const logits_snapshot & b) {
        if (a.data.size() != b.data.size() || a.data.empty()) return;

        double sum_diff = 0.0;
        double dot = 0.0, norm_a = 0.0, norm_b = 0.0;

        for (size_t i = 0; i < a.data.size(); i++) {
            float diff = std::fabs(a.data[i] - b.data[i]);
            if (diff > max_abs_diff) max_abs_diff = diff;
            sum_diff += diff;
            dot    += (double)a.data[i] * (double)b.data[i];
            norm_a += (double)a.data[i] * (double)a.data[i];
            norm_b += (double)b.data[i] * (double)b.data[i];
        }

        mean_abs_diff = (float)(sum_diff / a.data.size());
        cosine_sim = (norm_a > 0 && norm_b > 0) ? dot / (std::sqrt(norm_a) * std::sqrt(norm_b)) : 0.0;

        argmax_a = a.argmax();
        argmax_b = b.argmax();
        argmax_match = (argmax_a == argmax_b);
    }

    void print(const char * label) const {
        printf("    %s:\n", label);
        printf("      max_abs_diff  = %.6e\n", max_abs_diff);
        printf("      mean_abs_diff = %.6e\n", mean_abs_diff);
        printf("      cosine_sim    = %.10f\n", cosine_sim);
        printf("      argmax_match  = %s (A=%d, B=%d)\n",
               argmax_match ? "YES" : "NO", argmax_a, argmax_b);
    }

    bool pass(float max_diff_thresh = 1e-5f, double cos_thresh = 0.99999) const {
        return max_abs_diff < max_diff_thresh && cosine_sim > cos_thresh && argmax_match;
    }
};

// Decode a token sequence and capture the last-token logits.
// Returns true on success.
static bool decode_and_capture(llama_context * ctx, const std::vector<llama_token> & tokens, logits_snapshot & out) {
    llama_batch batch = llama_batch_get_one(
        const_cast<llama_token *>(tokens.data()), tokens.size());
    int ret = llama_decode(ctx, batch);
    if (ret != 0) {
        fprintf(stderr, "decode failed: %d\n", ret);
        return false;
    }
    // Capture logits for the last token (index = n_tokens - 1)
    out.capture(ctx, tokens.size() - 1);
    return true;
}

// Greedy decode n_tokens starting from the current state
static std::vector<llama_token> greedy_generate(llama_context * ctx, int n_tokens) {
    std::vector<llama_token> generated;
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    for (int i = 0; i < n_tokens; i++) {
        float * logits = llama_get_logits_ith(ctx, -1);
        int32_t n_vocab = llama_vocab_n_tokens(vocab);
        int32_t best = (int32_t)(std::max_element(logits, logits + n_vocab) - logits);
        generated.push_back(best);

        llama_batch batch = llama_batch_get_one(&generated.back(), 1);
        int ret = llama_decode(ctx, batch);
        if (ret != 0) {
            fprintf(stderr, "greedy_generate decode failed at step %d: %d\n", i, ret);
            break;
        }
    }
    return generated;
}

// ============================================================================
// Test cases
// ============================================================================

static int n_pass = 0;
static int n_fail = 0;

static void report(const char * name, bool passed) {
    printf("  %-50s %s\n", name, passed ? "[PASS]" : "[FAIL]");
    if (passed) n_pass++; else n_fail++;
}

// C1: Same prompt decoded twice — logits should be identical
static void test_C1_same_prompt_twice(llama_context * ctx, llama_kv_cache * kv,
                                       const std::vector<llama_token> & tokens) {
    printf("\n--- C1: Same prompt decoded twice ---\n");

    // First decode (cold)
    llama_memory_clear(llama_get_memory(ctx), true);
    logits_snapshot snap_a;
    if (!decode_and_capture(ctx, tokens, snap_a)) { report("C1", false); return; }

    // Second decode (prefix cache should have the full sequence)
    llama_memory_clear(llama_get_memory(ctx), true);
    logits_snapshot snap_b;
    if (!decode_and_capture(ctx, tokens, snap_b)) { report("C1", false); return; }

    logits_comparison cmp;
    cmp.compute(snap_a, snap_b);
    cmp.print("logits A vs B");

    // Same prompt, same model, same context → should be bit-exact
    report("C1: same prompt logits match", cmp.argmax_match && cmp.max_abs_diff < 1e-6f);
}

// C2: Shared prefix + different suffix — baseline vs cached
static void test_C2_shared_prefix(llama_context * ctx, llama_kv_cache * kv,
                                   const std::vector<llama_token> & tokens_a,
                                   const std::vector<llama_token> & tokens_b) {
    printf("\n--- C2: Shared prefix + different suffix ---\n");

    // Count shared prefix
    size_t shared = 0;
    for (size_t i = 0; i < std::min(tokens_a.size(), tokens_b.size()); i++) {
        if (tokens_a[i] == tokens_b[i]) shared++; else break;
    }
    printf("    Shared prefix: %zu tokens\n", shared);

    // Baseline: decode B from scratch (no cache reuse)
    llama_memory_clear(llama_get_memory(ctx), true);
    logits_snapshot snap_baseline;
    if (!decode_and_capture(ctx, tokens_b, snap_baseline)) { report("C2", false); return; }

    // Cached: decode A first, then clear and decode B (cache should reuse prefix)
    llama_memory_clear(llama_get_memory(ctx), true);
    logits_snapshot snap_a;
    if (!decode_and_capture(ctx, tokens_a, snap_a)) { report("C2", false); return; }

    // Now clear KV but tree entries remain (generations will invalidate them though
    // because clear bumps generations). So we need a different approach:
    // Decode A, then decode B in a fresh seq but same context (seq_rm A, then decode B).
    llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);

    logits_snapshot snap_cached;
    if (!decode_and_capture(ctx, tokens_b, snap_cached)) { report("C2", false); return; }

    logits_comparison cmp;
    cmp.compute(snap_baseline, snap_cached);
    cmp.print("baseline vs cached");

    report("C2: shared prefix logits match", cmp.pass(1e-4f, 0.9999));
}

// C3: Long prefix (repeated text) + short suffix
static void test_C3_long_prefix(llama_context * ctx, llama_kv_cache * kv,
                                 const llama_vocab * vocab) {
    printf("\n--- C3: Long prefix + short suffix ---\n");

    // Build a long prefix by repeating text
    std::string long_text = "";
    for (int i = 0; i < 5; i++) {
        long_text += "The quick brown fox jumps over the lazy dog. ";
    }
    std::string suffix_a = "What color is the fox?";
    std::string suffix_b = "What animal is lazy?";

    auto tokens_a = tokenize(vocab, long_text + suffix_a, true);
    auto tokens_b = tokenize(vocab, long_text + suffix_b, true);

    printf("    Long prefix + suffix A: %zu tokens\n", tokens_a.size());
    printf("    Long prefix + suffix B: %zu tokens\n", tokens_b.size());

    // Baseline: decode B from scratch
    llama_memory_clear(llama_get_memory(ctx), true);
    logits_snapshot snap_baseline;
    if (!decode_and_capture(ctx, tokens_b, snap_baseline)) { report("C3", false); return; }

    // Cached: decode A first, then seq_rm, then decode B
    llama_memory_clear(llama_get_memory(ctx), true);
    logits_snapshot snap_a;
    if (!decode_and_capture(ctx, tokens_a, snap_a)) { report("C3", false); return; }

    llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);

    logits_snapshot snap_cached;
    if (!decode_and_capture(ctx, tokens_b, snap_cached)) { report("C3", false); return; }

    logits_comparison cmp;
    cmp.compute(snap_baseline, snap_cached);
    cmp.print("baseline vs cached (long prefix)");

    report("C3: long prefix logits match", cmp.pass(1e-4f, 0.9999));
}

// C4: Minimal prefix (1 shared token)
static void test_C4_minimal_prefix(llama_context * ctx, llama_kv_cache * kv,
                                    const llama_vocab * vocab) {
    printf("\n--- C4: Minimal prefix (single shared token) ---\n");

    // Two prompts that share only the BOS token (if any) or first word
    auto tokens_a = tokenize(vocab, "Hello world", true);
    auto tokens_b = tokenize(vocab, "Hello everyone", true);

    size_t shared = 0;
    for (size_t i = 0; i < std::min(tokens_a.size(), tokens_b.size()); i++) {
        if (tokens_a[i] == tokens_b[i]) shared++; else break;
    }
    printf("    Shared tokens: %zu\n", shared);
    printf("    Tokens A: %zu, Tokens B: %zu\n", tokens_a.size(), tokens_b.size());

    // Baseline
    llama_memory_clear(llama_get_memory(ctx), true);
    logits_snapshot snap_baseline;
    if (!decode_and_capture(ctx, tokens_b, snap_baseline)) { report("C4", false); return; }

    // Cached
    llama_memory_clear(llama_get_memory(ctx), true);
    logits_snapshot snap_a;
    if (!decode_and_capture(ctx, tokens_a, snap_a)) { report("C4", false); return; }

    llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);

    logits_snapshot snap_cached;
    if (!decode_and_capture(ctx, tokens_b, snap_cached)) { report("C4", false); return; }

    logits_comparison cmp;
    cmp.compute(snap_baseline, snap_cached);
    cmp.print("baseline vs cached (minimal prefix)");

    report("C4: minimal prefix logits match", cmp.pass(1e-4f, 0.9999));
}

// C5: Greedy generation consistency
static void test_C5_greedy_generation(llama_context * ctx, llama_kv_cache * kv,
                                       const std::vector<llama_token> & tokens_a,
                                       const std::vector<llama_token> & tokens_b) {
    printf("\n--- C5: Greedy generation consistency ---\n");

    const int gen_len = 20;

    // Baseline: decode B from scratch, then greedy generate
    llama_memory_clear(llama_get_memory(ctx), true);
    {
        llama_batch batch = llama_batch_get_one(
            const_cast<llama_token *>(tokens_b.data()), tokens_b.size());
        if (llama_decode(ctx, batch) != 0) { report("C5", false); return; }
    }
    auto gen_baseline = greedy_generate(ctx, gen_len);
    printf("    Baseline generated %zu tokens\n", gen_baseline.size());

    // Cached: decode A first, seq_rm, then decode B, then greedy generate
    llama_memory_clear(llama_get_memory(ctx), true);
    {
        llama_batch batch = llama_batch_get_one(
            const_cast<llama_token *>(tokens_a.data()), tokens_a.size());
        if (llama_decode(ctx, batch) != 0) { report("C5", false); return; }
    }
    llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
    {
        llama_batch batch = llama_batch_get_one(
            const_cast<llama_token *>(tokens_b.data()), tokens_b.size());
        if (llama_decode(ctx, batch) != 0) { report("C5", false); return; }
    }
    auto gen_cached = greedy_generate(ctx, gen_len);
    printf("    Cached generated %zu tokens\n", gen_cached.size());

    // Compare generated sequences
    bool match = (gen_baseline.size() == gen_cached.size());
    if (match) {
        for (size_t i = 0; i < gen_baseline.size(); i++) {
            if (gen_baseline[i] != gen_cached[i]) {
                printf("    Mismatch at position %zu: baseline=%d, cached=%d\n",
                       i, gen_baseline[i], gen_cached[i]);
                match = false;
                break;
            }
        }
    }

    if (match) {
        printf("    All %d generated tokens match.\n", gen_len);
    }
    report("C5: greedy generation matches", match);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    const char * model_path = argv[1];
    printf("=== Inference Consistency Test ===\n");
    printf("Model: %s\n\n", model_path);

    // Load model
    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;

    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }

    // Create context
    auto cparams = llama_context_default_params();
    cparams.n_ctx   = 512;
    cparams.n_batch = 512;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "Failed to create context\n"); llama_model_free(model); return 1; }

    // Enable prefix cache
    llama_memory_t mem = llama_get_memory(ctx);
    llama_kv_cache * kv = dynamic_cast<llama_kv_cache *>(mem);
    if (!kv) { fprintf(stderr, "Not a KV cache\n"); llama_free(ctx); llama_model_free(model); return 1; }
    kv->prefix_cache_enable();

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // Prepare test prompts
    std::string prefix = "You are a helpful assistant. Please answer the following question carefully and concisely.";
    std::string suffix_a = " What is the capital of France?";
    std::string suffix_b = " What is the largest planet in the solar system?";

    auto tokens_a = tokenize(vocab, prefix + suffix_a, true);
    auto tokens_b = tokenize(vocab, prefix + suffix_b, true);

    printf("Tokens A: %zu, Tokens B: %zu\n", tokens_a.size(), tokens_b.size());

    // Run tests
    test_C1_same_prompt_twice(ctx, kv, tokens_a);
    test_C2_shared_prefix(ctx, kv, tokens_a, tokens_b);
    test_C3_long_prefix(ctx, kv, vocab);
    test_C4_minimal_prefix(ctx, kv, vocab);
    test_C5_greedy_generation(ctx, kv, tokens_a, tokens_b);

    // Summary
    printf("\n=== Results: %d passed, %d failed ===\n", n_pass, n_fail);

    llama_free(ctx);
    llama_model_free(model);

    return n_fail > 0 ? 1 : 0;
}
