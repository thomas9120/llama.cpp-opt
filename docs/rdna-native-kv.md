# Native quantized KV attention on Strix Halo

## Source and scope

The quantized chunk loaders are adapted from [stew675/llama-cpp-rdna-boosts](https://github.com/stew675/llama-cpp-rdna-boosts/tree/ce69ef4e45d4a24a111ee7d3082110d4d4e54985), release `v16-84e76d8a2-r17`. The source patch uses base `84e76d8a2`; its SHA-256 is `6731f93c3daabebe40a432337b176961855500a133b08fe01f59b8d2e1f5c0a9`. This is a selective adaptation onto this fork's `7c6fac4e19a5e935525cc27c9c05e20dcbc0a5de`, not an upstream merge.

HIP on RDNA3.5 can read Q4_0, Q4_1, Q5_0, Q5_1, and Q8_0 cache blocks directly in the tile attention kernels and in WMMA attention with K head dimensions at most 128. K and V are selected independently. The loaders reproduce the F16 staging arithmetic, retain byte strides, and support cache views and aliased K/V storage. The allocator and launcher use the same eligibility predicate, so native operands need no full-cache F16 scratch buffer.

The existing single-token Q8 tile specialization remains in use. The vector attention path is unchanged. The 256-wide WMMA path retains staging: initial gfx1151 measurements showed a regression from the extra native-loader register pressure. IQ4_NL KV, BF16 native loading, graph changes, and other GPU architectures are outside this port.

`GGML_CUDA_FA_KV_NATIVE=0` disables the new loaders for an A/B comparison. Set the variable before starting the process; the value is cached. The pre-existing Q8 decode specialization remains active in both arms.

This does not remove the F16-only requirement of Qwen4exp's direct sparse-attention and block-selection paths. Quantized KV can still select a slower QSA route. Use F16 K/V as a separate model-level baseline.

## Validation

Hardware: Ryzen AI Max+ 395 / Radeon 8060S, gfx1151. Windows driver `32.0.31041.1004`, TheRock AMD Clang `23.0.0git` at LLVM commit `8f497e0992fb7513f7f78a6f6b6f1056c375e961`, Visual Studio 2022 x64 environment, Release, HIP graphs enabled, default FA quant instantiations.

- Full `test-windows.ps1 -BuildDir build-rdna -Jobs 8`, including GPU QSA visibility, MMB context, and F16/Q8 attention.
- 120 added cases in the existing backend-ops suite against CPU references: independent K/V types, permuted cache views, K/V aliases, partial KV tiles, and query widths 1, 8, and 32.
- A deterministic A/B probe covered 1,008 cases with head dimensions 64/128/256, KV lengths 256/257, widths 1/2/3/7/8/9/32, six cache types, mixed K/V, aliases, and masks with excluded cells. Every output matched the baseline bit for bit.
- A further 162 attention cases used actual KV lengths 8,192, 32,768, and 65,536. Native and staged outputs matched bit for bit before the conservative 256-wide WMMA exclusion.

The probe measures attention, including backend submission overhead; it is not a model tokens/second result. Native KV loading primarily removes cache conversion traffic and temporary storage. Single-token model decode can remain dominated by weight reads or already use a native vector kernel. No end-to-end decode speedup is implied by attention-only measurements.

The CPU, Vulkan, and NVIDIA implementations are not changed by the host eligibility gate. GPU validation above is specific to gfx1151; a CUDA toolchain was not available for cross-compilation.

Final checks also passed six existing 64/128-wide WMMA cases against CPU references. A 96-token greedy coding generation on the local Qwen3.5 9B Q6_K model, Q4_0 K / Q8_0 V, FA on, full GPU offload, context allocation 8,192, batch/ubatch 512, seed 123, and temperature 0 produced identical text before and after. This short generation is a correctness smoke test, not a representative coding benchmark.

Repeated attention timings after tuning, with 128-wide heads, 3 KV heads, GQA 4, 8 query tokens, 32,768 populated cache rows, and masks containing excluded cells:

| KV type | Staged (us) | Native (us) | Ratio |
| --- | ---: | ---: | ---: |
| Q8_0 | 751.3 | 321.0 | 2.34x |
| Q4_0 | 670.4 | 294.6 | 2.28x |
| Q4_1 | 681.9 | 292.5 | 2.33x |
| Q5_0 | 775.3 | 310.1 | 2.50x |
| Q5_1 | 771.9 | 328.3 | 2.35x |

These are means from two passes per setting in staged/native/native/staged order, each timing 100 warmed graph evaluations. F16 was an unchanged control (381.1/369.3 us); system timing noise remains. The arrays were resident in GPU buffers, with no lazy file reads, speculative drafter, sampler, or server slots in this kernel probe.
