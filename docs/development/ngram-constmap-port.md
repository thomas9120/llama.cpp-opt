# Static n-gram cache port

## Source and scope

Ported from [jadidbourbaki/llama.cpp PR #7](https://github.com/jadidbourbaki/llama.cpp/pull/7), open at review time on 2026-09-27, head `b90b28145bbb0f720789fd7191c3a03c689af240`, base `f46522c4102f9643c186bad2601ff213996ebfaf`.

The stack includes PRs [#2](https://github.com/jadidbourbaki/llama.cpp/pull/2), [#5](https://github.com/jadidbourbaki/llama.cpp/pull/5), and [#10](https://github.com/jadidbourbaki/llama.cpp/pull/10). Its base is `84e76d8a23162eca70490da131945ebec1f09bf4`. The eight imported revisions are `1d4372408`, `f911df50f`, `58abc4a5a`, `e5e1d6b44`, `9bec231d8`, `f46522c41`, `51acb8d73`, and `b90b28145`. This worktree starts at Windows fork commit `11277c2c4268ca3072f713248a46d7c016da9095`; the merge base with the PR is `718f7b4175bf8b6af6f5eac09fee10754b3ecddd`.

The stack avoids copying cache parts during drafting, uses segmented outer maps and sorted token-count vectors, and shares a compact immutable static cache across sequences. Dependencies are unordered_dense v5.0.1 and fastconstmap `990afd04148949300dc00b2206b63af9374c6334`, with their upstream licenses. The local cpp-httplib version pin and vendor patches are preserved.

Only the ngram-cache speculative implementation changes. MTP, ngram-mod, QSA masks, shared prefill inputs, pinned-host exclusions, Windows HC16 exclusions, and positioned lazy reads are unchanged. Upstream performance results were measured on Apple M4 Pro; they are not gfx1151 measurements.

## Compatibility adaptations

- Legacy static files are converted into the compact representation in memory without rewriting the original file.
- Original PR static files (`ggngcmap`) remain readable for inference. Their hash map does not retain keys, so they cannot be enumerated for merging; the loader gives an explicit rebuild instruction when used as merge input.
- New static files (`ggngcmp2`) append a key table for merging. Inference reads only the entries and constmap, skipping the key table. The table costs 16 bytes per n-gram on disk.
- The version 2 layout is three uint64 header fields (magic, entry count, map size), a uint64 key count, token/count entries, serialized constmap bytes, then key records (two int32 tokens and a uint64 packed span). Like the upstream format, it targets little-endian hosts.
- lookup-merge accepts legacy and version 2 inputs and emits the existing legacy format, retaining all n-gram lengths. Static inference converts this output on load.
- File sizes, map dimensions, and returned entry spans are checked. File output failures are reported. Merge input errors return a diagnostic and nonzero status.

## Validation

Completed on 2026-09-27:

- Windows source guards and `git diff --check` passed.
- `build-windows.ps1 -Jobs 8` and the full `test-windows.ps1 -Jobs 8` passed, including allocator recovery, direct lazy reads, server recovery, QSA visibility, shared prefill input views, pooled keys, overlapping MMB contexts, and ROCm F16/Q8 attention.
- `build-windows-vulkan.ps1 -VulkanSdkPath C:/VulkanSDK/1.4.350.0 -Jobs 8` passed. All four lookup tools also built with MSVC.
- The new cache regression passed against both production libraries (Clang/ROCm and MSVC/Vulkan). It checks binary search against the standard library, legacy and new format round trips, shared ownership, draft equivalence, mixed-format merge counts, upstream version 1 loading, corrupt files, write failures, and empty caches.
- The actual lookup-create executable created a version 2 cache from a synthetic corpus. lookup-merge read two copies and produced a legacy file; subsequent inference successfully loaded that merged file.
- Both backends completed three configurations: speculation disabled, ngram-cache with the version 2 file, and ngram-cache with the merged legacy file. Each configuration processed two concurrent slots and a cached repeat. All 18 completions returned 96 tokens and matched the corresponding non-speculative content exactly. Every speculative completion accepted all 84 drafted tokens. The repeat reused 500 prompt tokens and processed 4 new prompt tokens.

The model smoke used the locally available Qwen3.5 9B Nikusui v1 Uncensored Heretic Q6_K model. GGUF metadata confirms architecture `qwen35`, file type 18, quantization version 2. Settings: context 8192, batch 512, microbatch 256, two slots, eight CPU threads, full GPU offload, Flash Attention, F16 K/V, draft maximum 7, greedy sampling, seed 4242, and a 504-token repetitive numeric prompt. Lazy mode and MTP were not enabled. These short synthetic runs validate integration; they do not establish coding performance or long-context screenshot stability. No speedup is claimed relative to the previous ngram-cache implementation.

Hardware/runtime: Radeon 8060S (gfx1151), Windows display driver 32.0.31041.1004, TheRock at `C:/TheRock/build`, AMD Clang 23.0.0git revision `8f497e0992fb7513f7f78a6f6b6f1056c375e961`, Visual Studio 2022 MSVC 14.44, Vulkan SDK 1.4.350.0. Both built servers reported the expected ROCm0/Vulkan0 device. No test server remained running afterward.

Artifacts are in the worktree's ignored `build-pr7-checks/` directory: binary hashes, model commands, process IDs, logs, responses, summary, and reproduction harness. Full regression logs are under `build-rocm10-gfx1151/windows-regression/`. Build logs are `build-pr7-rocm.log`, `build-pr7-vulkan.log`, `build-pr7-lookup.log`, and `build-pr7-regression.log`.

The primary checkout and launcher binaries were not changed. This port is isolated in the ngram-constmap-port worktree and branch.


## PR #12 follow-up

Also imported [lemire's PR #12](https://github.com/jadidbourbaki/llama.cpp/pull/12), open at review time on 2026-09-27: commit `f764f3238ce98e1bf0b8a3f7d900fd360bf9b9f6`, based directly on `b90b28145bbb0f720789fd7191c3a03c689af240`. The production change is 15 added and 7 removed lines in common/ngram-cache.cpp. It applies unchanged and needs no additional dependency or format change.

The primary cache is first scanned for total and largest counts. If even the largest count cannot meet the sample/percentage thresholds, the code skips static-weight lookups for that part. The final percentage check remains necessary: static weighting can select a token with fewer primary observations than the largest count. The port preserves this final check, tie order, and fallback to dynamic/static caches.

Thirteen focused cases passed against the pre-follow-up MSVC library and the rebuilt MSVC and Clang libraries. They cover context/dynamic sample limits, exact percentage boundaries, diffuse distributions, weighted selection, post-weight rejection, tie order, and static fallback. Existing cache-format and merge regressions also passed.

Both backend builds, all four Vulkan lookup targets, and the full Windows regression suite passed again. The same 18 model completions were repeated on both backends: content and drafted/accepted token counts matched the saved pre-follow-up results exactly, including concurrent slots and prompt reuse. This is a correctness check, not a representative performance benchmark; upstream drafting-latency improvements are not claimed as Windows decode speedups.

Follow-up logs: build-pr12-regression.log, build-pr12-vulkan.log, build-pr12-lookup.log, build-pr12-model.log. Previous responses are retained in build-pr7-checks/pr12-before-model; current responses and timing data are in build-pr7-checks/model. Updated binary hashes are in build-pr7-checks/pr12-binary-hashes.json. The main checkout and launcher binaries remain unchanged. The complete port is prepared on the ngram-constmap-port branch for review.
