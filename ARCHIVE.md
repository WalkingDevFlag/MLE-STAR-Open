# Ollama Integration Deferral Note

This archive is intentionally scoped down to only document why the Ollama (local model) adapter was deferred.

## Summary
Initial local model (Ollama) integration was postponed to prioritize core pipeline stability and upcoming structured tool usage (e.g., formal search/tool invocation and artifact caching). The deferral prevents rework while the provider abstraction, evaluation harness, and tool layer are still in motion.

## Key Reasons

1. Provider Interface Still Evolving – Caching hooks, richer response metadata, and error normalization are being added; locking a second backend early would multiply refactors.
2. Tool & Retrieval Layer In Flight – Incorporating structured tool usage (search, potential retrieval augmentation) changes prompt shapes and provider call patterns; better to finalize those first.
3. Deterministic Baseline First – A single remote backend (OpenRouter) simplifies debugging model candidate generation and evaluation variance.
4. Onboarding Friction – Requiring local model downloads / GPU setup raises the barrier for early testers; API‑only start keeps feedback loop fast.
5. Test Surface Control – Each additional provider expands CI coverage needs (rate limits, streaming behaviors, error edge cases) before the current suite is mature.
6. Caching Prerequisite – Local inference benefits most after a prompt / artifact cache exists to avoid redundant generations; that cache isn’t merged yet.

## Revisit Criteria

- Prompt / artifact caching layer merged
- Structured tool invocation integrated (search & any retrieval utilities)
- Provider interface stable (no breaking changes for 2+ weeks)
- Candidate ranking & evaluation artifacts persisted and reviewed

## Interim Guidance

- Users eager for local models can prototype a subclass of `LLMProvider` (see `shared_libraries/llm_provider.py`) and set an environment switch locally; upstream inclusion waits for the criteria above.

## Next Step After Revisit
 
Implement minimal `OllamaProvider` with: health check, model name mapping, unified response schema (latency + token usage placeholders), retry/backoff, and optional local-only flag. Add basic tests mirroring the OpenRouter provider contract.

---

End of archive (scoped to Ollama deferral only).
