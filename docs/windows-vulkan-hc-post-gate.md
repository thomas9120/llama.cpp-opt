# Vulkan HC post gate fusion on Windows

## Source and scope

Port of [ggml-org/llama.cpp#29520](https://github.com/ggml-org/llama.cpp/pull/29520), merged upstream as `03a667aa304f2a8e02a9a02b2e3fb45d64bcae7f`. Local base: `1d13fa1c5d6ed9f8cfdc924744b204070c32466d`.

The five Vulkan production files match the upstream patch. The existing `test_dsv4_hc_post` gains the upstream gated cases plus two single-token cases at embedding width 2560, with and without combination weights. Existing Xing and HC consumer tests are retained.

The backend fuses `SCALE -> SIGMOID -> SCALE -> DSV4_HC_POST`, computing scatter weights inside the POST shader. Matching requires sigmoid, zero bias on both scales, F32 input, matching shapes, and the expected graph edges. The graph optimizer preserves the chain and registers allocation dependencies; the existing overlap check can still reject fusion. The original input's strides and offsets are passed to the shader.

This changes Vulkan only. It does not change model graphs, HIP HC kernels, Windows HC16 exclusions, lazy reads, or QSA/cache handling.

## Reproduction

Use the existing scripts with Visual Studio 2022 and a compatible Vulkan SDK:

```powershell
.\build-windows-vulkan.ps1 -Jobs 8
cmake --build build-vulkan --target test-backend-ops --parallel 8
.\build-vulkan\bin\test-backend-ops.exe test -b Vulkan0 -o DSV4_HC_POST,DSV4_HC_PRE,HC_F32_CONSUMER
.\test-windows.ps1 -Jobs 8
```

Set `GGML_VK_PERF_LOGGER=1` for a diagnostic run to verify `HC_POST_GATE`. Remove it before measuring model throughput. `GGML_VK_DISABLE_FUSION=1` exercises the unfused fallback but disables other Vulkan fusions too, so use a saved pre-port build for performance comparisons.

## Validation environment

Windows, Radeon 8060S (gfx1151), AMD proprietary driver 26.8.1 (LLPC), Vulkan driver version 2.0.395, Vulkan SDK 1.4.350.0, VS 2022 x64 Release. The HIP regression build uses TheRock at `C:/TheRock/build` and targets gfx1151.

The baseline executable and its DLLs were rebuilt at the local base and saved together under `build-pr-review/baseline-bin`. The patched Vulkan executables are under `build-vulkan/bin`. Logs and temporary harnesses are in ignored `build-pr-review/`.

Targeted Vulkan checks pass 11/11 POST, 10/10 supported PRE, and 27/27 HC consumer cases against CPU. Twelve unsupported PRE stream-count cases are skipped. The separate Xing POST probe is unsupported on Vulkan and is not counted as a correctness pass. POST also passes 11/11 with fusion disabled. Timing traces confirm four unfused dispatches become one `HC_POST_GATE` dispatch for the single-token identity case.

The full `test-windows.ps1 -Jobs 8` suite passes, including all 14 source guards, the HIP rebuild, allocator failure recovery, lazy-reader checks, server recovery, QSA metadata/GPU masks, KV pool checks, decode consistency, MMB context isolation, and F16/Q8 attention. Additional HIP checks pass 11/11 POST and 27/27 HC consumer cases. `git diff --check` passes.

## Model comparison

The supplied three-shard `Qwen3.8-Flash-Next-UD-IQ4_XS` model reports architecture `qwen4exp`, 48 layers, and embedding width 2560. Its actual tensor mix is 557 F32, 502 Q8_0, 94 IQ3_S, 44 IQ4_NL, 24 BF16, two IQ4_XS, and one Q6_K tensors. The PLE table is IQ4_NL. The filename does not imply uniformly IQ4_XS weights.

Both servers used `-c 9216 -np 1 -ngl 99 -fa on -ctk f16 -ctv f16 -b 512 -ub 512 -t 8 -tb 8 --lazy-mode on-direct --fit off --reasoning off`. KV offload was enabled. Speculation and multimodal projection were not enabled. Requests used temperature 0, seed 1234, 128 predicted tokens, `cache_prompt=false`, and five returned token probabilities. The baseline and patched runs used identical tokenized prompts and request order. The compiler was MSVC 19.44.35228.0.

The first request at each prompt length was discarded as warmup. The table contains medians of three remaining short requests and two remaining 8K requests. These are warm file-cache measurements; the OS cache was not flushed. The baseline ran first, then the patched build. This small sequential sample cannot establish a sub-percent performance difference or a general coding-workload speedup.

| Filled prompt | Build | Prompt tokens/s | Decode tokens/s | HTTP completion time (s) |
| --- | --- | ---: | ---: | ---: |
| 49 tokens | Baseline | 46.57 | 25.58 | 6.770 |
| 49 tokens | Patched | 46.38 | 25.32 | 6.799 |
| 8192 tokens | Baseline | 444.73 | 22.07 | 24.190 |
| 8192 tokens | Patched | 445.22 | 21.94 | 24.219 |

Measured decode changes are -1.0% and -0.6%, respectively. This test does not demonstrate a throughput improvement on this quant. All seven paired requests, including warmups, produced identical token sequences and identical reported sampled-token log probabilities (896 generated tokens compared). Separate repeats within the baseline varied, so this is a comparison of corresponding request sequences, not a claim that repeated greedy generation is always deterministic. Both builds retrieved the marker at the start of the filled 8K prompt.

A separate fixed-text check uses `llama-perplexity` with two 256-token chunks, F16 KV, Flash Attention, batch/ubatch 256, eight threads, and direct lazy reads. The baseline saves reference logits; the patched build uses `--kl-divergence --kl-divergence-base`. The comparison reports 100% top-token agreement and probability deltas rounding to 0.000%. Mean PPL is 4.402799 versus reference 4.402842. Reported mean KL is -0.000010; the tool stores clipped, quantized 16-bit reference log probabilities, so this near-zero negative estimate is not a literal negative KL divergence or proof of identical full-precision logits. Some uncertainty fields print NaN for these near-identical results. This is a small numerical smoke test, not a broad perplexity evaluation.

A separate verbose model trace, excluded from the timing table, confirms `HC_POST_GATE` executes in Qwen inference (93-95 fused POST chains per recorded graph). It also confirms 49/49 layers offloaded, a 61222.07 MiB Vulkan model buffer, F16 KV, and active Windows direct reads for the IQ4_NL PLE table. All temporary test servers were stopped after their checks.

No MTP, image input, quantized KV model run, or filled 32K/64K model workload was tested for this port. The targeted operator and Windows regression results do not cover those model-level cases.
