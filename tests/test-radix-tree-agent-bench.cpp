// Agent Scenario Benchmark for Radix Tree Prefix Cache
//
// Simulates a 5-turn agent conversation (System + Tools + multi-turn ReAct)
// and measures:
//   1. TTFT (prefill latency) per turn
//   2. Prefix cache hit count and hit rate per turn
//   3. Prefix cache operation overhead (promote + find)
//   4. Projected TTFT savings when skip-prefill (A1) is implemented
//
// Usage: test-radix-tree-agent-bench <model.gguf>

#include "llama.h"
#include "llama-kv-cache.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

using hrclock = std::chrono::high_resolution_clock;

static double elapsed_ms(hrclock::time_point start, hrclock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// ============================================================================
// Turn result
// ============================================================================

struct turn_result {
    int         turn_id;
    int         total_tokens;
    int         cache_hits;
    double      hit_rate_pct;
    double      prefill_ms;         // actual full prefill time
    double      find_ms;            // prefix_cache_find latency
    double      promote_ms;         // prefix_cache_promote latency
    int         new_tokens;         // total - cache_hits
    double      projected_ttft_ms;  // estimated TTFT after A1 (skip prefill)
};

// ============================================================================
// Agent conversation content
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
    // Turn 1: User query
    "\n\nUser: What is the population of Tokyo and how does it compare to New York City?",

    // Turn 2: Agent think + tool call + observation
    "\n\nAssistant: I need to search for the population of both cities. Let me start with Tokyo."
    "\n{\"tool\": \"search\", \"args\": {\"query\": \"population of Tokyo 2024\"}}"
    "\n\nObservation: Tokyo has a population of approximately 13.96 million in the city proper "
    "and 37.4 million in the greater metropolitan area, making it the most populous metropolitan area in the world."
    "\n\nUser: Good, now compare with New York.",

    // Turn 3: Agent think + tool call + observation
    "\n\nAssistant: Now let me search for New York City's population for comparison."
    "\n{\"tool\": \"search\", \"args\": {\"query\": \"population of New York City 2024\"}}"
    "\n\nObservation: New York City has a population of approximately 8.34 million in the city proper "
    "and 20.1 million in the metropolitan area."
    "\n\nUser: Can you calculate the ratio?",

    // Turn 4: Agent think + tool call + observation
    "\n\nAssistant: Let me calculate the population ratio between Tokyo and NYC."
    "\n{\"tool\": \"calculator\", \"args\": {\"expression\": \"13.96 / 8.34\"}}"
    "\n\nObservation: Result: 1.6738..."
    "\n\nUser: What about the weather in both cities right now?",

    // Turn 5: Agent think + tool call + observation
    "\n\nAssistant: I'll check the weather in both cities."
    "\n{\"tool\": \"weather\", \"args\": {\"city\": \"Tokyo\"}}"
    "\n\nObservation: Tokyo: 18°C, partly cloudy, humidity 65%."
    "\n{\"tool\": \"weather\", \"args\": {\"city\": \"New York\"}}"
    "\n\nObservation: New York: 12°C, sunny, humidity 45%."
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

    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║   Agent Scenario Benchmark — Prefix Cache Performance      ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    // --- Load model ---
    printf("[Init] Loading model: %s\n", model_path);
    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0; // CPU only for reproducibility

    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    auto cparams = llama_context_default_params();
    cparams.n_ctx   = 4096;
    cparams.n_batch = 4096;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "Failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    // Enable prefix cache
    llama_memory_t mem = llama_get_memory(ctx);
    llama_kv_cache * kv = dynamic_cast<llama_kv_cache *>(mem);
    if (!kv) {
        fprintf(stderr, "Failed to get KV cache\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    kv->prefix_cache_enable();

    const llama_vocab * vocab = llama_model_get_vocab(model);

    printf("[Init] Context: n_ctx=%u, n_batch=%u\n", llama_n_ctx(ctx), cparams.n_batch);
    printf("[Init] Prefix cache enabled.\n\n");

    // --- Build cumulative prompts for each turn ---
    // Agent scenario: each turn appends to the conversation history.
    // In the current llama.cpp (without A3 multi-turn append), we clear and
    // re-decode the full context each turn. This is the baseline behavior.

    std::string system_str(SYSTEM_PROMPT);
    std::vector<std::string> cumulative_prompts(N_TURNS);
    {
        std::string accum = system_str;
        for (int t = 0; t < N_TURNS; t++) {
            accum += TURN_CONTENTS[t];
            cumulative_prompts[t] = accum;
        }
    }

    // Tokenize all turns
    std::vector<std::vector<llama_token>> turn_tokens(N_TURNS);
    for (int t = 0; t < N_TURNS; t++) {
        turn_tokens[t] = tokenize(vocab, cumulative_prompts[t], true);
    }

    printf("┌─────────────────────────────────────────────────────────────┐\n");
    printf("│ Turn │ Cumulative Tokens │ New Tokens (text)               │\n");
    printf("├─────────────────────────────────────────────────────────────┤\n");
    for (int t = 0; t < N_TURNS; t++) {
        int prev = (t > 0) ? (int)turn_tokens[t-1].size() : 0;
        int delta = (int)turn_tokens[t].size() - prev;
        printf("│  %d   │       %4zu        │     +%3d (turn content)        │\n",
               t+1, turn_tokens[t].size(), delta);
    }
    printf("└─────────────────────────────────────────────────────────────┘\n\n");

    // --- Run benchmark ---
    std::vector<turn_result> results(N_TURNS);

    for (int t = 0; t < N_TURNS; t++) {
        printf("── Turn %d/%d ─────────────────────────────────────────────\n", t+1, N_TURNS);

        auto & tokens = turn_tokens[t];
        turn_result & res = results[t];
        res.turn_id = t + 1;
        res.total_tokens = (int)tokens.size();

        // Step 1: Clear KV cache (simulates current behavior without A3)
        llama_memory_clear(mem, true);

        // Step 2: Measure prefix_cache_find (before decode)
        // After clear, generations are bumped, so all entries are invalid.
        // But after turn 1's promote, entries exist in the tree.
        // We need to decode first, then measure find for the NEXT turn's perspective.

        // Step 3: Full prefill (baseline TTFT)
        {
            auto t0 = hrclock::now();
            llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
            int ret = llama_decode(ctx, batch);
            auto t1 = hrclock::now();

            if (ret != 0) {
                fprintf(stderr, "  Decode failed at turn %d: %d\n", t+1, ret);
                break;
            }
            res.prefill_ms = elapsed_ms(t0, t1);
        }

        // Step 4: Measure promote latency
        {
            std::vector<uint32_t> cell_indices(tokens.size());
            for (size_t i = 0; i < tokens.size(); i++) {
                cell_indices[i] = (uint32_t)i;
            }

            auto t0 = hrclock::now();
            kv->prefix_cache_promote(tokens, cell_indices);
            auto t1 = hrclock::now();
            res.promote_ms = elapsed_ms(t0, t1);
        }

        // Step 5: Measure find latency for NEXT turn's tokens (if not last turn)
        if (t + 1 < N_TURNS) {
            auto & next_tokens = turn_tokens[t + 1];
            std::vector<uint32_t> out_cells;

            auto t0 = hrclock::now();
            int32_t matched = kv->prefix_cache_find(next_tokens, out_cells);
            auto t1 = hrclock::now();

            res.find_ms = elapsed_ms(t0, t1);
            res.cache_hits = matched;
            res.new_tokens = (int)next_tokens.size() - matched;
            res.hit_rate_pct = 100.0 * matched / next_tokens.size();

            // Project TTFT savings: assume prefill time scales linearly with token count
            double ms_per_token = res.prefill_ms / tokens.size();
            res.projected_ttft_ms = ms_per_token * res.new_tokens;
        } else {
            // Last turn: measure find for the same tokens (self-match)
            std::vector<uint32_t> out_cells;

            auto t0 = hrclock::now();
            int32_t matched = kv->prefix_cache_find(tokens, out_cells);
            auto t1 = hrclock::now();

            res.find_ms = elapsed_ms(t0, t1);
            res.cache_hits = matched;
            res.new_tokens = (int)tokens.size() - matched;
            res.hit_rate_pct = 100.0 * matched / tokens.size();
            res.projected_ttft_ms = 0; // full match, no new tokens
        }

        printf("  Total tokens:    %d\n", res.total_tokens);
        printf("  Prefill time:    %.1f ms  (%.2f ms/tok)\n",
               res.prefill_ms, res.prefill_ms / res.total_tokens);
        printf("  Cache hits:      %d / %d  (%.1f%%)\n",
               res.cache_hits,
               (t + 1 < N_TURNS) ? (int)turn_tokens[t+1].size() : res.total_tokens,
               res.hit_rate_pct);
        printf("  New tokens:      %d\n", res.new_tokens);
        printf("  Promote time:    %.3f ms\n", res.promote_ms);
        printf("  Find time:       %.3f ms\n", res.find_ms);
        if (t + 1 < N_TURNS) {
            printf("  Projected TTFT:  %.1f ms  (after A1 skip-prefill)\n", res.projected_ttft_ms);
            printf("  Projected save:  %.1f ms  (%.0f%% reduction)\n",
                   res.prefill_ms - res.projected_ttft_ms,
                   100.0 * (1.0 - res.projected_ttft_ms / res.prefill_ms));
        }
        printf("\n");
    }

    // --- Summary table ---
    printf("╔══════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║                          BENCHMARK SUMMARY                                 ║\n");
    printf("╠══════╦════════╦══════════╦══════════╦═══════╦══════════╦════════╦═══════════╣\n");
    printf("║ Turn ║ Tokens ║ TTFT(ms) ║ ms/tok   ║ Hits  ║ Hit Rate ║ New Tk ║ Proj TTFT║\n");
    printf("╠══════╬════════╬══════════╬══════════╬═══════╬══════════╬════════╬═══════════╣\n");

    double total_prefill_ms = 0;
    double total_projected_ms = 0;
    int total_hits = 0;
    int total_tokens_all = 0;

    for (int t = 0; t < N_TURNS; t++) {
        auto & r = results[t];
        printf("║  %d   ║  %4d  ║ %7.1f  ║  %5.2f   ║ %4d  ║  %5.1f%%  ║  %4d  ║  %6.1f   ║\n",
               r.turn_id, r.total_tokens, r.prefill_ms,
               r.prefill_ms / r.total_tokens,
               r.cache_hits, r.hit_rate_pct, r.new_tokens,
               r.projected_ttft_ms);

        total_prefill_ms += r.prefill_ms;
        total_projected_ms += (t + 1 < N_TURNS) ? r.projected_ttft_ms : 0;
        total_hits += r.cache_hits;
        total_tokens_all += r.total_tokens;
    }

    printf("╠══════╬════════╬══════════╬══════════╬═══════╬══════════╬════════╬═══════════╣\n");
    printf("║ Sum  ║ %5d ║ %7.1f  ║  %5.2f   ║ %4d  ║          ║        ║  %6.1f   ║\n",
           total_tokens_all, total_prefill_ms,
           total_prefill_ms / total_tokens_all,
           total_hits, total_projected_ms);
    printf("╚══════╩════════╩══════════╩══════════╩═══════╩══════════╩════════╩═══════════╝\n");

    printf("\n");
    printf("┌─────────────────────────────────────────────────────────────┐\n");
    printf("│ KEY METRICS                                                 │\n");
    printf("├─────────────────────────────────────────────────────────────┤\n");
    printf("│ Total prefill time (baseline):  %7.1f ms                  │\n", total_prefill_ms);
    printf("│ Projected total (with A1):      %7.1f ms                  │\n",
           results[0].prefill_ms + total_projected_ms);
    printf("│ Projected savings:              %7.1f ms  (%.0f%%)          │\n",
           total_prefill_ms - results[0].prefill_ms - total_projected_ms,
           100.0 * (1.0 - (results[0].prefill_ms + total_projected_ms) / total_prefill_ms));
    printf("│ Avg cache overhead (find+promote): %.3f ms/turn            │\n",
           (results[0].find_ms + results[0].promote_ms +
            results[1].find_ms + results[1].promote_ms +
            results[2].find_ms + results[2].promote_ms +
            results[3].find_ms + results[3].promote_ms +
            results[4].find_ms + results[4].promote_ms) / N_TURNS);
    printf("└─────────────────────────────────────────────────────────────┘\n");

    printf("\n");
    printf("NOTE: \"Projected TTFT\" assumes linear scaling of prefill time with token\n");
    printf("count. Actual savings after implementing A1 (skip-prefill) may differ due\n");
    printf("to GPU batch efficiency and attention mask overhead.\n");
    printf("\n");

    // --- Cleanup ---
    llama_free(ctx);
    llama_model_free(model);

    printf("=== Benchmark Complete ===\n");
    return 0;
}
