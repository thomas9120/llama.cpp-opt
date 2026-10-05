# llama.cpp/examples/lookup

Demonstration of Prompt Lookup Decoding

https://github.com/apoorvumang/prompt-lookup-decoding

The key parameters for lookup decoding are `ngram_min`, `ngram_max` and `n_draft`. The first two determine the size of the ngrams to search for in the prompt for a match. The latter specifies how many subsequent tokens to draft if a match is found.

More info:

https://github.com/ggml-org/llama.cpp/pull/4484
https://github.com/ggml-org/llama.cpp/issues/4226

## Cache formats

`llama-lookup-create` writes a compact static cache. Static caches are shared across
server sequences. Existing legacy caches are still accepted and are converted in
memory at load time; rebuild them with `llama-lookup-create` to avoid that conversion.

`llama-lookup-merge` accepts both legacy caches and this fork's compact static caches.
It writes the legacy format, preserving support for dynamic caches and mixed n-gram
lengths. The static loader can read that merged output.

Compact files retain a table of original n-gram keys for merging. Inference skips
this table, so it increases disk space but not the resident cache size. Files from
the original constmap PR (magic `ggngcmap`) can be used for inference, but cannot be
merged because they do not contain the original keys. Rebuild those files from
their source corpus with this fork's `llama-lookup-create` before merging.

This optimization applies to `--spec-type ngram-cache` and the lookup examples.
It does not change `ngram-mod`, MTP, GPU attention, or lazy model loading.
