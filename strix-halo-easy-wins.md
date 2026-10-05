# Strix Halo (gfx1151) + Qwen4exp/QSA easy wins from halo-box/strix-llama.cpp

Goal of this fork (`fork-goals.md`): optimize for Strix Halo (GFX1151), Qwen4exp and QSA
(Qwen3.8 Flash Next, Qwen3.8 27B). Focus: ROCm/HIP + Windows.

Upstream base here: `thomas9120/llama.cpp-opt` + `ggml-org/llama.cpp` sync, no fork kernels yet.
`halo-box/strix-llama.cpp` is a child of `halo-box/llama.cpp`.

Local gap (verified 2026-10-05): `ggml/src/ggml-cuda/` has no `mmb.*`, `qsa-*`,
`idx-relu-sum.*`, `hyperconn.*`. The big Flash-Next gains need those files first.

## Tier 1: portable cherry-picks

### 1. Q6_K MMQ threshold 256 -> 1024 on RDNA3.5
- PR: https://github.com/halo-box/strix-llama.cpp/pull/38 (MERGED, +5/-0)
- File: `ggml/src/ggml-cuda/mmq.cu`
- Local: `ggml/src/ggml-cuda/mmq.cu:422` has `ne11 <= (... ? 128 : 256)` for Q6_K
- Change: `ne11 <= 1024` under `GGML_CUDA_CC_IS_RDNA3_5`, avoids dequant + hipBLAS fallback
- Measured: +8-10% prefill Q4_K_M/Q5_K_M, +30% pure Q6_K, +50% at `-ub 384`
- Windows: HIP path identical, gfx1151-gated, no behavior change elsewhere
- Port: 1-line gate change

### 2. Shape-aware HIP graph cache key
- PR: https://github.com/halo-box/strix-llama.cpp/pull/100 (MERGED, +18/-0)
- File: `ggml/src/ggml-cuda/ggml-cuda.cu`
- Change: key was `nodes[0]` alone, graphs sharing one arena never replayed.
  Mix in node count + last node + 3 node shapes. Collision only resets warmup.
- Measured: pp2048/tg128 within +/-0.5%, win is MTP speculative verify widths
- Windows: HIP graph path identical, unconditional, backend-agnostic
- Port: small hunk, check `ggml/src/ggml-cuda/ggml-cuda.cu:2651` area

### 3. FA head-192 off WMMA kernel
- PR: https://github.com/halo-box/strix-llama.cpp/pull/55 (MERGED, +3/-1)
- File: `ggml/src/ggml-cuda/fattn.cu`
- Change: route only `DKQ <= 128, == 256` to `BEST_FATTN_KERNEL_MMA_F16`, 192 back to tile kernel.
  Upstream widened gate to `<= 256` but WMMA device code only built for 256 above 128,
  192 trapped with `NO_DEVICE_CODE` / `HSA_STATUS_ERROR_EXCEPTION`.
- Measured: `test-backend-ops -o FLASH_ATTN_EXT` abort -> pass on gfx1151
- Windows: correctness prerequisite before FA tuning
- Port: 1-line gate

### 4. Vulkan small-M mul_mat as swapped mat-vec
- PR: https://github.com/halo-box/strix-llama.cpp/pull/129 (MERGED, +65/-3)
- File: `ggml/src/ggml-vulkan/ggml-vulkan.cpp`
- Change: Flash-Next HC inject `m=4,n=2048,k=10240` x95/batch went through full matmul tile.
  Run m<=8 plain 2-D as swapped mat-vec + small transpose copy. Extends #28457 m=1 trick.
- Measured: op 145ms -> 54ms, +4% prefill depth-0, +3.4% at 12k on gfx1151 RADV. Decode unchanged.
- Windows: Vulkan-on-Windows relevant, isolated, has backend-ops cases
- Port: small, check `ggml/src/ggml-vulkan/ggml-vulkan.cpp:10953` area

### 5. Vulkan mat-vec chunk gating fix
- PR: https://github.com/halo-box/strix-llama.cpp/pull/27 (MERGED, +13/-10)
- File: `ggml/src/ggml-vulkan/ggml-vulkan.cpp`
- Change: #1 chunked all types at cols 3,5,6,8. Restrict to measured `q8_0,q6_K`,
  others take upstream dispatch. `GGML_VK_MMV_NO_SPLIT=1` still disables.
- Measured: dense Qwen3.8-27B UD-Q4_K_XL decode 9.8 -> 15.1 t/s in server test
- Windows: AMD-gated, note upstream later added own tuning `2cdae802e`, expect fuzz
- Port: small

## Tier 2: next, needs base infra first

- #82 + #96 QSA 9..512 ubatches, incremental keys to 512: +28-83% small prompts at
  depth 131k, +50-152% at 250k. Touches fork-new `qsa-decode.*`, needs #94/#95 order.
  https://github.com/halo-box/strix-llama.cpp/pull/82
  https://github.com/halo-box/strix-llama.cpp/pull/96
- #131 MMB gate 512->32 rows for qwen4exp: -9-13% TTFT 256/512 prompts, +53/-16,
  but needs MMB base (#63/#18) and is not math-preserving.
  https://github.com/halo-box/strix-llama.cpp/pull/131
- #40 Q4_0_ROCMFP4_FAST MMQ tiles (+469): only if shipping FP4 GGUFs, else HIP uses
  dequant + hipBLAS fallback. Isolated, zero change for other types.
  https://github.com/halo-box/strix-llama.cpp/pull/40
- #106 OPEN decode (+1635): +19.6% tg 24.28->29.05 t/s byte-exact, 10 commits stacked
  on #91. Idea to steal alone: GDN decode-chain fusion (+5.6%).
  https://github.com/halo-box/strix-llama.cpp/pull/106

## Do not cherry-pick alone

- #75 -> #123 -> #141 MMB gate chain: meaningless without MMB base absent locally.
  #141 notes pp4096 1650->1260 t/s regression after #123 on Flash-Next.
- #101 MTP vocab subset: qwen35/qwen35moe only, not qwen4exp/Flash-Next.
  #108 is the qwen4exp equivalent, still OPEN.
- #18 (+6824), #91 (+2780), #17 (+5009): base infra, review for ideas only.
- #128 Vulkan chunked GDN (+537): +0.2-0.7% today, defer.

## Suggested order

1. #38, #55, #100 (<25 lines, `test-backend-ops -o MUL_MAT,FLASH_ATTN_EXT`)
2. #129, #27 if shipping Vulkan (`-b Vulkan0 -o TOP_K,FLASH_ATTN_EXT`)
3. Decide: port MMB base (#63/#18) or QSA-decode base (#82 infra) as a unit.
   Single MMB/QSA tuning commits do not apply without them.

Bench from PRs: `llama-bench -ngl 99 -fa on -ctk f16 -ctv f16 -b 4096 -ub 4096 -p 512,1024,2048,4096 -d 0,32000,64000 -r 3`
