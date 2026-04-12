// Agent Scenario A/B/C Benchmark
//
// Three-group comparison for prefix cache overhead and benefit:
//   Group A (baseline):  Original llama.cpp, no modifications
//   Group B (disabled):  Modified llama.cpp, prefix cache NOT enabled
//   Group C (enabled):   Modified llama.cpp, prefix cache enabled
//
// Compile three separate binaries:
//   g++ ... -DBENCH_GROUP=1 → baseline (link original libllama)
//   g++ ... -DBENCH_GROUP=2 → disabled (link modified libllama)
//   g++ ... -DBENCH_GROUP=3 → enabled  (link modified libllama)
//
// Usage: test-agent-bench-X <model.gguf>

#include "llama.h"

#if BENCH_GROUP >= 2
#include "llama-kv-cache.h"
#endif

#include <chrono>
#include <cstdio>
#include <cstdlib>
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
    if (ret < 0) { return {}; }
    tokens.resize(ret);
    return tokens;
}

using hrclock = std::chrono::high_resolution_clock;

static double elapsed_ms(hrclock::time_point a, hrclock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// ============================================================================
// Agent conversation (same across all groups)
// ============================================================================

static const char * SYSTEM_PROMPT =
    "You are a helpful AI assistant with access to tools. "
    "When the user asks a question, think step by step, "
    "decide which tool to use, and call it with the correct arguments. "
    "Available tools:\n"
    "1. search(query: string) - Search the web for information\n"
    "2. calculator(expression: string) - Evaluate a math expression\n"
    "3. weather(city: string) - Get current weather for a city\n"
    "4. database(sql: string) - Query the user database\n"
    "Always respond in JSON format for tool calls: "
    "{\"tool\": \"name\", \"args\": {\"key\": \"value\"}}";

static const char * TURN_CONTENTS[] = {
    "\n\nUser: What is the population of Tokyo and how does it compare to New York City?",

    "\n\nAssistant: I need to search for the population of both cities. Let me start with Tokyo."
    "\n{\"tool\": \"search\", \"args\": {\"query\": \"population of Tokyo 2024\"}}"
    "\n\nObservation: Tokyo has a population of approximately 13.96 million in the city proper "
    "and 37.4 million in the greater metropolitan area, making it the most populous metropolitan area in the world."
    "\n\nUser: Good, now compare with New York.",

    "\n\nAssistant: Now let me search for New York City's population for comparison."
    "\n{\"tool\": \"search\", \"args\": {\"query\": \"population of New York City 2024\"}}"
    "\n\nObservation: New York City has a population of approximately 8.34 million in the city proper "
    "and 20.1 million in the metropolitan area."
    "\n\nUser: Can you calculate the ratio?",

    "\n\nAssistant: Let me calculate the population ratio between Tokyo and NYC."
    "\n{\"tool\": \"calculator\", \"args\": {\"expression\": \"13.96 / 8.34\"}}"
    "\n\nObservation: Result: 1.6738..."
    "\n\nUser: What about the weather in both cities right now?",

    "\n\nAssistant: I'll check the weather in both cities."
    "\n{\"tool\": \"weather\", \"args\": {\"city\": \"Tokyo\"}}"
    "\n\nObservation: Tokyo: 18C, partly cloudy, humidity 65%."
    "\n{\"tool\": \"weather\", \"args\": {\"city\": \"New York\"}}"
    "\n\nObservation: New York: 12C, sunny, humidity 45%."
    "\n\nUser: Summarize everything you found.",
};

static const int N_TURNS = 5;

// ============================================================================
// Main
// ============================================================================

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    const char * model_path = argv[1];

#if BENCH_GROUP == 1
    const char * group_name = "A (Original llama.cpp - baseline)";
#elif BENCH_GROUP == 2
    const char * group_name = "B (Modified llama.cpp - prefix cache DISABLED)";
#elif BENCH_GROUP == 3
    const char * group_name = "C (Modified llama.cpp - prefix cache ENABLED)";
#else
    #error "Define BENCH_GROUP=1, 2, or 3"
#endif

    printf("================================================================\n");
    printf("  Agent Benchmark — Group %s\n", group_name);
    printf("================================================================\n\n");

    // --- Load model ---
    printf("[Init] Loading model: %s\n", model_path);
    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;

    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }

    auto cparams = llama_context_default_params();
    cparams.n_ctx   = 4096;
    cparams.n_batch = 4096;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "Failed to create context\n"); llama_model_free(model); return 1; }

    llama_memory_t mem = llama_get_memory(ctx);

#if BENCH_GROUP == 3
    // Enable prefix cache (Group C only)
    llama_kv_cache * kv = dynamic_cast<llama_kv_cache *>(mem);
    if (!kv) { fprintf(stderr, "Failed to get KV cache\n"); llama_free(ctx); llama_model_free(model); return 1; }
    kv->prefix_cache_enable();
    printf("[Init] Prefix cache ENABLED.\n");
#elif BENCH_GROUP == 2
    llama_kv_cache * kv = dynamic_cast<llama_kv_cache *>(mem);
    printf("[Init] Prefix cache DISABLED (modified code, cache not enabled).\n");
#else
    printf("[Init] Original llama.cpp (no prefix cache code).\n");
#endif

    const llama_vocab * vocab = llama_model_get_vocab(model);
    printf("[Init] n_ctx=%u, n_batch=%u, CPU only\n\n", llama_n_ctx(ctx), cparams.n_batch);

    // --- Build cumulative prompts ---
    std::string system_str(SYSTEM_PROMPT);
    std::vector<std::string> cumulative_prompts(N_TURNS);
    {
        std::string accum = system_str;
        for (int t = 0; t < N_TURNS; t++) {
            accum += TURN_CONTENTS[t];
            cumulative_prompts[t] = accum;
        }
    }

    std::vector<std::vector<llama_token>> turn_tokens(N_TURNS);
    for (int t = 0; t < N_TURNS; t++) {
        turn_tokens[t] = tokenize(vocab, cumulative_prompts[t], true);
    }

    // Print token counts
    printf("Turn  Tokens  Delta\n");
    printf("────  ──────  ─────\n");
    for (int t = 0; t < N_TURNS; t++) {
        int prev = (t > 0) ? (int)turn_tokens[t-1].size() : 0;
        printf("  %d    %4zu    +%d\n", t+1, turn_tokens[t].size(), (int)turn_tokens[t].size() - prev);
    }
    printf("\n");

    // --- Warmup: 1 decode to stabilize timings ---
    printf("[Warmup] Decoding turn 1 once to stabilize...\n");
    {
        llama_memory_clear(mem, true);
        llama_batch batch = llama_batch_get_one(turn_tokens[0].data(), turn_tokens[0].size());
        llama_decode(ctx, batch);
    }
    printf("[Warmup] Done.\n\n");

    // --- Benchmark ---
    double ttft[N_TURNS] = {};
    double promote_time[N_TURNS] = {};
    double find_time[N_TURNS] = {};
    int    cache_hits[N_TURNS] = {};

    for (int t = 0; t < N_TURNS; t++) {
        auto & tokens = turn_tokens[t];

        // Clear KV cache (each turn starts fresh — current llama.cpp behavior)
        llama_memory_clear(mem, true);

        // Decode (measure TTFT)
        auto t0 = hrclock::now();
        llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
        int ret = llama_decode(ctx, batch);
        auto t1 = hrclock::now();

        if (ret != 0) {
            fprintf(stderr, "Decode failed at turn %d: %d\n", t+1, ret);
            break;
        }
        ttft[t] = elapsed_ms(t0, t1);

#if BENCH_GROUP == 3
        // Promote into prefix cache
        {
            std::vector<uint32_t> cell_indices(tokens.size());
            for (size_t i = 0; i < tokens.size(); i++) cell_indices[i] = (uint32_t)i;

            auto p0 = hrclock::now();
            kv->prefix_cache_promote(tokens, cell_indices);
            auto p1 = hrclock::now();
            promote_time[t] = elapsed_ms(p0, p1);
        }

        // Find for next turn (measure cache hits)
        if (t + 1 < N_TURNS) {
            auto & next = turn_tokens[t + 1];
            std::vector<uint32_t> out;

            auto f0 = hrclock::now();
            int32_t matched = kv->prefix_cache_find(next, out);
            auto f1 = hrclock::now();

            find_time[t] = elapsed_ms(f0, f1);
            cache_hits[t] = matched;
        } else {
            std::vector<uint32_t> out;
            auto f0 = hrclock::now();
            int32_t matched = kv->prefix_cache_find(tokens, out);
            auto f1 = hrclock::now();
            find_time[t] = elapsed_ms(f0, f1);
            cache_hits[t] = matched;
        }
#endif
    }

    // --- Results ---
    printf("================================================================\n");
    printf("  RESULTS — Group %s\n", group_name);
    printf("================================================================\n\n");

    printf("Turn  Tokens   TTFT(ms)   ms/tok");
#if BENCH_GROUP == 3
    printf("   Hits  HitRate  Promote(ms) Find(ms)");
#endif
    printf("\n");

    printf("────  ──────   ────────   ──────");
#if BENCH_GROUP == 3
    printf("   ────  ───────  ─────────── ────────");
#endif
    printf("\n");

    double total_ttft = 0;
    for (int t = 0; t < N_TURNS; t++) {
        int n = (int)turn_tokens[t].size();
        printf("  %d    %4d    %7.1f    %5.2f",
               t+1, n, ttft[t], ttft[t] / n);
#if BENCH_GROUP == 3
        int ref_n = (t + 1 < N_TURNS) ? (int)turn_tokens[t+1].size() : n;
        printf("   %4d  %5.1f%%    %7.3f     %6.3f",
               cache_hits[t],
               100.0 * cache_hits[t] / ref_n,
               promote_time[t], find_time[t]);
#endif
        printf("\n");
        total_ttft += ttft[t];
    }

    printf("\n");
    printf("Total TTFT:  %.1f ms\n", total_ttft);
    printf("Avg ms/tok:  %.2f\n", total_ttft / 1536.0);  // approximate

#if BENCH_GROUP == 3
    // Calculate projected savings
    double projected_total = ttft[0]; // turn 1 always full
    for (int t = 0; t < N_TURNS - 1; t++) {
        int next_n = (int)turn_tokens[t+1].size();
        int new_tok = next_n - cache_hits[t];
        double ms_per_tok = ttft[t] / turn_tokens[t].size();
        projected_total += ms_per_tok * new_tok;
    }
    printf("\nProjected TTFT (with A1 skip-prefill): %.1f ms\n", projected_total);
    printf("Projected savings: %.1f ms (%.0f%%)\n",
           total_ttft - projected_total,
           100.0 * (1.0 - projected_total / total_ttft));

    double avg_overhead = 0;
    for (int t = 0; t < N_TURNS; t++) avg_overhead += promote_time[t] + find_time[t];
    printf("Avg cache overhead: %.3f ms/turn\n", avg_overhead / N_TURNS);
#endif

    printf("\n");

    // --- Cleanup ---
    llama_free(ctx);
    llama_model_free(model);

    return 0;
}
