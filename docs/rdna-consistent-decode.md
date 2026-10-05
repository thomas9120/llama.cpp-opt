# Consistent decode and verification on Strix Halo

## Enable

Set this before starting the HIP process:

```powershell
$env:GGML_CUDA_CONSISTENT_DECODE = '1'
```

The setting is opt-in and cached at process startup. Unset it or set it to `0` to retain the usual speed-oriented dispatch. It only applies on RDNA3.5. For speculative comparisons, use one slot and draft at most seven tokens so the target's draft-plus-one verification batch stays within eight tokens. Larger batches retain the existing prefill choices.

Native quantized KV loading remains independently controlled by `GGML_CUDA_FA_KV_NATIVE`; see [native KV notes](rdna-native-kv.md). Qwen4exp's direct QSA paths still require F16 K/V, Flash Attention, and KV offload. This option does not make quantized QSA equivalent to that F16 route.

## Changes

The approach is adapted from the width-consistency findings in [stew675's GREEDY-PURITY notes](https://github.com/stew675/llama-cpp-rdna-boosts/blob/ce69ef4e45d4a24a111ee7d3082110d4d4e54985/GREEDY-PURITY.md). It is a small dispatch adaptation to this fork, not the collection's full matvec rewrite.

- Attention with 64/128/256-wide heads uses the decode tile configuration for query widths 1-8. The KV split heuristic also uses the width-one tile count. The existing native Q8 tile implementation covers the whole range.
- Supported floating-point matvecs stay on MMVF through width eight, after the existing alignment checks.
- Quantized routed matvecs stay on MMVQ through width eight, with device launch bounds large enough for that range. Single-token routed calls use the same MoE reduction as verification, except for the retained, already equivalent fused-quantize fast paths.
- Dense GLU fusions in this range and the specialized single-token weighted MoE down projection are bypassed. Other fusions, four-column weight reuse, prefill dispatch, Windows HC16 exclusions, and QSA visibility rules remain in place.

This controls the selected attention and matvec paths. It is not a general deterministic-inference switch for every architecture, head dimension, operator, device, precision, or batch size. It can trade verification throughput for numerical consistency, so compare delivered tokens/second and completion time on the intended model before making it a launcher default.

## Validation

Same gfx1151 Windows/TheRock/driver setup as the native-KV tests. The Windows regression runner compiles `scripts/windows-decode-consistency.cpp`, enables the mode for that executable, and restores the caller's environment afterward.

- Full `test-windows.ps1 -BuildDir build-rdna -Jobs 8` passed, including the new consistency check and existing QSA visibility, cache reuse, reader, allocator, server recovery, MMB, and attention checks.
- The consistency test covers 108 shape/type combinations at every width 1-8 (864 GPU graphs). It checks every output column against the width-one output, bit for bit. Attention includes excluded cells and long cache rows; matvec cases include dense, routed, and GLU graphs, odd output-row counts, and K lengths 256/512/4096. The same test fails on the pre-change binary at F16 attention width two, confirming that the test detects the original problem.
- With the mode enabled, 151 routed, 501 dense, and 120 attention cases in `test-backend-ops` passed CPU-reference comparisons. Existing routed test loops now also cover width eight.
- Qwen3.5 9B Q6_K model logits were checked after restoring the same prefix state for each width. At a 112-token F16 prefix and a fully populated 8,192-token prefix with both F16 and Q8 K/V, all 248,320 logits for the first verification position matched width one exactly at widths 1-8. Disabled-mode controls showed differences at wider batches; the 8K F16 control reached a maximum absolute logit difference of about 0.172.

Model probe settings: full GPU offload, one sequence, Flash Attention on, batch/ubatch 512, eight CPU threads, no sampling or draft model, and identical fixed input tokens. The 8K test used an 8,704-token context allocation so all eight verification positions fit. It validates restored-state execution and logits, not draft acceptance or coding quality. Qwen3.8-Flash-Next model-level performance and other architectures were not measured by this probe.

## Generation and throughput observations

One warm coding run per configuration used the same Qwen3.5 9B Q6_K model, a 512-token generation limit, F16 K/V, context 8192, batch/ubatch 512, one slot, full GPU offload, Flash Attention, greedy sampling, seed 123, and the default CPU thread setting. MTP used the preserved native draft tensors in the same GGUF and at most seven draft tokens. The runs were sequential with no competing test workload.

| Consistent decode | Plain generation (tokens/s) | Native MTP (tokens/s) |
| --- | ---: | ---: |
| Disabled | 26.4 | 25.3 |
| Enabled | 26.1 | 26.6 |

Plain and MTP greedy text matched within each mode on this prompt. Enabling the mode changed the generated text relative to disabled mode because the selected arithmetic paths changed. This is width consistency within the mode, not bit compatibility with the default dispatch.

These single runs have different generated text across modes and do not establish an end-to-end speed improvement. They also do not replace representative coding benchmarks or measure Qwen3.8-Flash-Next throughput. Keep the mode opt-in until repeated delivered-throughput measurements on the intended workload justify a launcher default.
