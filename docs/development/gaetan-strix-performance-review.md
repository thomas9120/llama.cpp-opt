# gaetan-puleo Strix Halo performance review

Reviewed 2026-09-26 against local `strix-halo-beta` at `cf6c4ee276138b78df390bae50e4e3e6ca5e32cf` (includes the shared QSA input-view memory fix). This is a source and model-metadata review, not a performance result or an imported patch set.

## Source revisions and scope

Repository: [gaetan-puleo/llama-cpp-strix-halo](https://github.com/gaetan-puleo/llama-cpp-strix-halo). The GitHub API returned no open or closed pull requests or issues in that repository at review time. Commit messages therefore provide most of its experiment descriptions; their measurements have not been reproduced on this Windows system.

| Branch | Reviewed head | Relevance |
| --- | --- | --- |
| `master` | `860c828363988d3e4b3d5c2b701dcba3d7b9f26c` | Performance work on an older base; no `src/models/qwen4exp.cpp` at this tip. The optimization series follows `a30273376`. |
| `long-context-improvements` | `f7eed71e60a6e4b6442f3fe2576944dd70886336` | Ancestor of master; not an additional newer patch series. |
| `import/fork-master-optimizations` | `ff9c3b29987f75398a43d1e2464378639fa9808e` | Most useful comparison. Contains qwen4exp and 24 commits after halo-box's `732484c20`, repackaging and extending the optimization work. |
| `poc` | `e0fdcae0986e442d0203e7e7ca2beb29c116b527` | Older experiments, including graph-owned activation caching. Not a replacement for our context-safe buffers. |
| `ds4-experiment` | `cdcae7ba6a262db95d7c3158167734ed5a5aa1ac` | DeepSeek-V4-specific sparse attention and indexing, not the Qwen sparse-attention implementation. |

The local repository is shallow, with boundaries at `0636c9ae` and `866322481`; `git merge-base HEAD gaetan-review/master` finds no common ancestor in the available history. This does not establish unrelated ancestry. Findings use exact commit patches and direct tip-to-tip file comparisons, not a presumed merge base. Comparison refs were fetched under `refs/remotes/gaetan-review/`; configured remote URLs and the working branch were not changed.

## Actual AtomicChat tensor layout

Read the metadata of all 33 local AD-4.27bpw-Q4_K_M-M64 shards without loading the model onto the GPU. The filename does not describe every tensor's quantization.

| Tensor family | Type and dimensions, GGML order | Count |
| --- | --- | ---: |
| Routed gate and up, each | IQ2_S, `[2560,640,512]` | 36 each |
| Routed gate and up, each | IQ3_S, `[2560,640,512]` | 12 each |
| Routed down | IQ4_NL, `[640,2560,512]` | 48 |
| Shared gate and up, each | Q8_0, `[2560,640]` | 48 each |
| Shared down | Q8_0, `[640,2560]` | 48 |
| SSM alpha and beta, each | Q8_0, `[2560,48]` | 36 each |

The saved matching-model startup log records 512 experts, 10 selected experts, 24 attention heads / 2 KV heads (GQA12), and GDN state size 128 with 48 value heads and 16 key heads. This excludes many of the other fork's exact-shape tuning gates.

## Recommended candidates

### 1. Share MMQ activation quantization between paired projections

Sources: [routed pairs, 441bec16a](https://github.com/gaetan-puleo/llama-cpp-strix-halo/commit/441bec16af850463e8013bcd34e68d24fa3715f3), [dense Q8 pairs, 35e450a8d](https://github.com/gaetan-puleo/llama-cpp-strix-halo/commit/35e450a8d97f746001ef57f4697e924dab664643), and their implementations in the import branch's `ggml-cuda.cu` / `mmq.cu`.

The current local MMQ path prepares expert IDs and Q8_1 activations separately for each matrix multiplication. Their paired path shares those preparations between adjacent compatible operations while keeping separate weight matrices and outputs. The local MoE graph has separate gate/up projections, and the actual AtomicChat pairs match in shape and quantization. The dense shared experts are Q8_0 and fit the dense pair restriction. Runtime graph tracing must still establish which pairs are adjacent after graph construction and which reach MMQ rather than existing decode fusions.

This is the strongest direct Qwen3.8 candidate, primarily for prompt processing and possibly verification batches that select MMQ. It is not a claimed batch-one decode gain. Their earlier Qwen3.6 routed-pair experiment reports roughly 1-4% prompt-processing gains, not measurements for our model or baseline.

Adaptation requirements:

- Retain our `mmq_args::ncols_opt`, 512-expert descriptor capacity, IQ4_NL compact selection, and guarded partial-tile ID reads. Their import branch uses a different 256-expert compact selector and lacks these local adaptations.
- Preserve backend placement, graph-capture buffer lifetimes, compute-weight padding rules, and overlapping-context isolation. Reuse per-context temporary allocation; do not introduce global shared storage.
- Keep the ordinary dense path restricted to Q8_0 initially. Their dense-pair commit explicitly excluded ordinary IQ weights after an exactness failure; routed IQ pairs are a separate path and need actual IQ2_S/IQ3_S coverage here.
- Begin with already-adjacent pairs. Scheduling alpha/beta together is a subsequent graph change inspired by [3d3096790](https://github.com/gaetan-puleo/llama-cpp-strix-halo/commit/3d30967904c8121d82da066783565084e34fbdd1), which modifies qwen35moe rather than qwen4exp. The actual AtomicChat alpha/beta weights are compatible Q8_0, but adjacency, allocations, and outputs need separate validation.

### 2. Two activation-quantization chunks per block

Source: [4d74d746d](https://github.com/gaetan-puleo/llama-cpp-strix-halo/commit/4d74d746d0cc0a5883cf99305eeac43df20c1292), including the scatter adaptation visible at the import tip.

This is absent locally. Each block handles two chunks instead of one, retaining each quantization group's arithmetic. It can reduce launch-grid overhead across MMQ consumers without modifying model or cache state. Test independently of paired MMQ so the benefit can be attributed. Preserve scatter indexing, channel offsets, padded tails, and all Q8_1 metadata layouts. Initially limit new scheduling to the intended HIP/gfx1151 path; the source change is broader. No local throughput estimate is justified yet.

### 3. Small conversion and elementwise kernels

Sources: [F16/F32 conversion pairs, e9f38f77b](https://github.com/gaetan-puleo/llama-cpp-strix-halo/commit/e9f38f77bdf883f4a077d9bdcc9140c307b4eb5a) and [contiguous F32 ADD/MUL, 8ee14b01f](https://github.com/gaetan-puleo/llama-cpp-strix-halo/commit/8ee14b01f5fac6391d16262417f2f41a1d07eb37).

Both are absent locally and avoid model, lazy-reader, and checkpoint changes. Conversion handles aligned pairs and scalar odd tails; ADD/MUL bypasses general broadcasting for identical contiguous F32 shapes. They are suitable small first experiments, with likely modest overall gains. The conversion author's KAT-Coder result saved about 1.9 ms per pp2048 pass. That is not a Qwen3.8 gain. ADD/MUL must be compared against our existing packed broadcast implementation; specialization alone does not prove it is faster.

Test aligned and misaligned inputs, odd tails, multidimensional/noncontiguous fallbacks, in-place outputs, and exact rounding. Gate architecture-specific dispatch rather than silently changing every CUDA device.

### 4. Vulkan verification-batch column splitting

This is inherited halo-box work, not a new gaetan-only optimization: merged [halo-box PR #1](https://github.com/halo-box/strix-llama.cpp/pull/1), `ca94157f70a2776e8da6b6849b50b45a083d0478`, with merged correctness follow-up [PR #10](https://github.com/halo-box/strix-llama.cpp/pull/10), `732484c20c8c7a61885c2e48ea28f90bdd7bd1e1`.

The reviewed import branch splits slow small matvec batch widths into better-performing widths, including Qwen3.8 speculative verification. The local Vulkan code lacks this chunking helper, but has newer row tuning and substantial backend restructuring. Measurements in the patch are from Linux RADV/Mesa 26.1; Windows AMD Vulkan may have different crossover points. Treat this as a Windows profiling experiment, after confirming a slow-width problem locally.

The correctness follow-up is mandatory: splitting must not bypass or improperly reuse whole-tensor Q8_1 integer-dot activation caches. Preserve the existing arithmetic path. Port the small dispatch idea into current code, not the old Vulkan backend file.

## Already present, mismatched, or deferred

| Source work | Current assessment |
| --- | --- |
| Q6_K fused activation-quantize decode (`37f02eb14`), Q8 FQ, F32 matvec prefetch, short BF16 matvec | Already represented locally. Our FQ support extends to additional IQ types; replacing the source would remove work. |
| MMQ skip-empty-tile setup (`51148f72d`) | Already present: local `mul_mat_q` checks routed bounds before shared-ID initialization. |
| MMQ prefetch and base RDNA3.5 configuration | Already represented locally; the import branch's `mmq-config-rdna3-5.cuh` is identical to ours. |
| H32/H64 GDN tiling and widening | The AtomicChat model uses H48; local code already has an H48-specific tiled path. Wholesale replacement would remove that selection. Other models can be assessed separately. |
| Dual L2 norm (`27cbd9d00`) and full GDN decode fusion (`1cf1594e6`) | Not directly compatible. Their matcher and arithmetic use L2 norm with `max(sum, eps*eps)`. Current `build_gdn_l2_norm` uses RMS norm with `eps/n`, then scales by `1/sqrt(n)`. Replacing this would change normalization. A future fusion must preserve the current operations and state-write ordering. |
| SwiGLU-to-Q8 quantization (`e1d11b93b`) | Import-tip dispatch requires width 512 -> 2048, Q8_0 down weights, and 256/8 routed experts. AtomicChat is width 640 -> 2560, with IQ4_NL routed down weights. Shared down is Q8_0 but still misses the shape gate. Adaptation is a separate experiment, not a direct enabled win. |
| Weighted expert/shared-gate fusions | Their retained gates target width 2048 and 8 selected experts. Preserve our 2560/10-compatible weighted-reduction and HC paths. |
| Qwen122 Q4/Q5 and 256-expert compact MMQ | Different types and shapes from the actual AtomicChat experts. Their fixed descriptor capacity must not replace ours. |
| D256/GQA8 mask reuse and register limits | Our F16 Qwen path uses dedicated direct-index QSA kernels with GQA12. These generic FA improvements do not imply a gain for that path. Other models or fallback paths require separate profiling. |
| Device speculative checkpoints (`f169e0b7d`) | Potentially useful but deferred. It also changes MTP from recurrent snapshots to checkpoint rollback. Their earlier recorded gain was only about 0.55% in one short MTP3 comparison. Revisit separately with memory accounting, rejected-draft rollback, screenshots, and long-context tests; do not bundle it with kernel tuning. |
| BF16 exact-batch hint (`8a22f52e5`) | A numerical-consistency change applied to qwen35 projections, not demonstrated acceleration for this qwen4exp quant. |
| POC graph-owned activation cache (`86bbe4968`) | Cache lifetime changes overlap the class of failures already fixed locally. Do not substitute it for our per-context MMB/replay ownership. |
| DeepSeek-V4 native sparse attention | Different graph/indexing contract. Do not replace Qwen QSA with it. |

The paired-MMQ commit cites canonical [ggml-org/llama.cpp PR #27233](https://github.com/ggml-org/llama.cpp/pull/27233) as inspiration. At review time that PR is closed and unmerged, and contains a much broader evolving integration. It is not an approved dependency to import wholesale.

## Safe implementation order and validation

Start with conversion/ADD-MUL as small independent experiments, then two-chunk quantization, then paired MMQ. Paired MMQ has the strongest model-specific opportunity but a larger integration surface. Profile Vulkan column widths separately. Leave GDN arithmetic, speculative checkpoint policy, and global cache ownership outside this first series.

Preserve the Windows host-buffer compatibility restriction, all HC16 exclusions, per-context MMB ownership/replay, allocator recovery, lazy positioned reads/prefetch, QSA indexer invalidation, screenshot visibility masks, and `qwen4exp_shared_input_view`. Avoid replacing entire backend or model files.

For each implementation: retain a baseline binary set; use the existing backend-ops infrastructure with actual IQ2_S/IQ3_S/IQ4_NL/Q8_0 shapes; run the Windows regression suite and Vulkan build for shared-code changes; check normalized generation/logits, cache reuse, overlapping contexts, MTP on/off, and screenshots. Measure prompt processing and delivered decode separately at filled 8K/32K/64K contexts, then repeat the screenshot memory test beyond 140K. Keep model files, sampling seed, F16 KV, offload, DIO/lazy mode, and batch sizes fixed. Track private memory and system commit alongside throughput; `-ub 1024` is the established long-context comparison setting.

Completed for this review: branch/API inventory, commit/diff inspection, local CodeGraph/source comparison, AtomicChat tensor-header inspection, all 14 Windows source guards, and `git diff --check`. No production source, binaries, launcher files, or GPU workloads were changed. Builds, runtime correctness, and performance of the candidates remain untested.
