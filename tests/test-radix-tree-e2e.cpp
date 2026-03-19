// Phase 4: End-to-end test for radix tree prefix cache
//
// Loads a real GGUF model, enables the prefix cache on the KV cache,
// performs two decode passes that share a common prefix, and verifies
// that the prefix cache correctly identifies and reuses the shared portion.
//
// Usage: test-radix-tree-e2e <model.gguf>

#include "llama.h"
#include "llama-kv-cache.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

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

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    const char * model_path = argv[1];

    printf("=== Radix Tree Prefix Cache End-to-End Test ===\n\n");

    // --- Load model ---
    printf("[1] Loading model: %s\n", model_path);
    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0; // CPU only for testing

    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    printf("    Model loaded successfully.\n");

    // --- Create context ---
    printf("[2] Creating context...\n");
    auto cparams = llama_context_default_params();
    cparams.n_ctx   = 512;
    cparams.n_batch = 512;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "Failed to create context\n");
        llama_model_free(model);
        return 1;
    }
    printf("    Context created (n_ctx=%u).\n", llama_n_ctx(ctx));

    // --- Enable prefix cache ---
    printf("[3] Enabling prefix cache on KV cache...\n");
    llama_memory_t mem = llama_get_memory(ctx);
    llama_kv_cache * kv = dynamic_cast<llama_kv_cache *>(mem);
    if (!kv) {
        fprintf(stderr, "Failed to get KV cache (memory is not llama_kv_cache)\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    kv->prefix_cache_enable();
    printf("    Prefix cache enabled.\n");

    // --- Tokenize ---
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // Shared prefix (system prompt)
    std::string shared_prefix = "You are a helpful assistant. Please answer the following question carefully and concisely.";
    // Two different suffixes
    std::string question_a = " What is the capital of France?";
    std::string question_b = " What is the largest planet in the solar system?";

    auto tokens_full_a = tokenize(vocab, shared_prefix + question_a, true);
    auto tokens_full_b = tokenize(vocab, shared_prefix + question_b, true);
    auto tokens_prefix = tokenize(vocab, shared_prefix, true);

    printf("[4] Tokenized:\n");
    printf("    Shared prefix:  %zu tokens\n", tokens_prefix.size());
    printf("    Full query A:   %zu tokens\n", tokens_full_a.size());
    printf("    Full query B:   %zu tokens\n", tokens_full_b.size());

    // --- First decode (query A) ---
    printf("[5] Decoding query A (first pass, no cache)...\n");
    {
        llama_batch batch = llama_batch_get_one(tokens_full_a.data(), tokens_full_a.size());
        int ret = llama_decode(ctx, batch);
        if (ret != 0) {
            fprintf(stderr, "    Decode A failed: %d\n", ret);
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        printf("    Decode A succeeded.\n");
    }

    // --- Check prefix cache after first decode ---
    printf("[6] Checking prefix cache after decode A...\n");
    {
        std::vector<uint32_t> out_cells;
        int32_t matched = kv->prefix_cache_find(tokens_full_a, out_cells);
        printf("    prefix_cache_find(query_A): matched=%d\n", matched);

        // Also check if the shared prefix matches
        std::vector<uint32_t> prefix_cells;
        int32_t prefix_matched = kv->prefix_cache_find(tokens_full_b, prefix_cells);
        printf("    prefix_cache_find(query_B): matched=%d (shared prefix)\n", prefix_matched);

        if (prefix_matched > 0) {
            printf("    SUCCESS: Prefix cache found %d shared tokens for query B\n", prefix_matched);
        } else {
            printf("    NOTE: No prefix match yet (auto-promote may not have triggered)\n");
        }
    }

    // --- Manually promote the tokens from decode A ---
    printf("[7] Manually promoting query A tokens into prefix tree...\n");
    {
        // Build cell indices: for llama_batch_get_one with seq_id=0,
        // tokens are placed at positions 0..n-1, which map to cell indices 0..n-1
        std::vector<uint32_t> cell_indices(tokens_full_a.size());
        for (size_t i = 0; i < tokens_full_a.size(); i++) {
            cell_indices[i] = (uint32_t)i;
        }
        kv->prefix_cache_promote(tokens_full_a, cell_indices);
        printf("    Promoted %zu tokens.\n", tokens_full_a.size());
    }

    // --- Check if query B can find the shared prefix ---
    printf("[8] Checking prefix match for query B...\n");
    {
        std::vector<uint32_t> out_cells;
        int32_t matched = kv->prefix_cache_find(tokens_full_b, out_cells);
        printf("    prefix_cache_find(query_B): matched=%d\n", matched);

        // The shared prefix tokens should match
        size_t expected_prefix = 0;
        for (size_t i = 0; i < std::min(tokens_full_a.size(), tokens_full_b.size()); i++) {
            if (tokens_full_a[i] == tokens_full_b[i]) {
                expected_prefix++;
            } else {
                break;
            }
        }
        printf("    Expected shared prefix: %zu tokens\n", expected_prefix);

        if (matched > 0 && (size_t)matched >= expected_prefix) {
            printf("    PASS: Prefix cache correctly identified %d shared tokens\n", matched);
        } else if (matched > 0) {
            printf("    PARTIAL: Prefix cache found %d tokens (expected >= %zu)\n", matched, expected_prefix);
        } else {
            printf("    FAIL: No prefix match found\n");
        }
    }

    // --- Test generation invalidation ---
    printf("[9] Testing generation invalidation...\n");
    {
        // Clear KV cache (simulates context reset)
        llama_memory_clear(mem, true);
        printf("    KV cache cleared.\n");

        // Now the tree entries should be invalid (cells were freed)
        std::vector<uint32_t> out_cells;
        int32_t matched = kv->prefix_cache_find(tokens_full_a, out_cells);
        printf("    prefix_cache_find after clear: matched=%d\n", matched);

        if (matched == 0) {
            printf("    PASS: Cache correctly invalidated after clear\n");
        } else {
            printf("    FAIL: Cache should be invalid after clear, but matched=%d\n", matched);
        }
    }

    // --- Cleanup ---
    printf("\n[10] Cleanup...\n");
    llama_free(ctx);
    llama_model_free(model);
    printf("    Done.\n");

    printf("\n=== End-to-End Test Complete ===\n");
    return 0;
}
