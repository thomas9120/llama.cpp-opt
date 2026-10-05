# strix01 upstream port log

Goal (from `fork-goals.md`): Strix Halo (gfx1151), Qwen4exp/QSA (Qwen3.8 Flash Next).
Upstream sources: `origin` = thomas9120/llama.cpp-opt, `upstream` = ggml-org/llama.cpp.
Fork reference: halo-box/strix-llama.cpp (child of halo-box/llama.cpp).

## How not to lose work on upstream sync

1. Every ported hunk carries an inline tag: `STRIX01-PR<n> (halo-box/strix-llama.cpp#<n>)`.
   Find them after any merge with:
   `git grep -n STRIX01 -- ggml src common`
2. This file is the source of truth. Before `git fetch upstream` / `git merge`,
   note `git rev-parse HEAD`. After merge, check each row below for conflicts
   or silent upstream refactors of the same lines.
3. If upstream touches the same gate (e.g. new `ne11` threshold, new FA kernel
   dispatch, new graph key), keep the STRIX01 behavior and record the
   upstream commit hash + decision in the row.
4. Verify with the `Test` column binary after every sync, not just on port day.

## Ported

| Local change | Upstream PR | Files | Status | Upstream-conflict risk | Test |
|---|---|---|---|---|---|
| Q6_K MMQ threshold 256 -> 1024 on RDNA3.5 | halo-box/strix-llama.cpp#38 MERGED | `ggml/src/ggml-cuda/mmq.cu` (`ggml_cuda_should_use_mmq`) | ported 2026-10-05, tag `STRIX01-PR38` | Medium: upstream tunes same `ne11` gates often (e.g. ggml-org#25940 Q6_K vec_dot). On conflict keep RDNA3.5 1024 branch. | `test-backend-ops -o MUL_MAT,MUL_MAT_ID` PASS 2026-10-05 (1891/1891, 931/931, gfx1151 Windows HIP). Q4_0 bench pending |
| FA head-192 off WMMA kernel | halo-box/strix-llama.cpp#55 MERGED | `ggml/src/ggml-cuda/fattn.cu` (`ggml_cuda_get_best_fattn_kernel`) | ported 2026-10-05, tag `STRIX01-PR55` | Medium: upstream widens/narrows `Q->ne[0] <= 256` gate. Keep `<=128 \|\| ==256` until WMMA device code covers 192. | `test-backend-ops -o FLASH_ATTN_EXT` PASS 2026-10-05 (3991/3991, gfx1151 Windows HIP) |
| Shape-aware HIP graph cache key | halo-box/strix-llama.cpp#100 MERGED | `ggml/src/ggml-cuda/ggml-cuda.cu` (`ggml_cuda_graph_get_key`) | ported 2026-10-05, tag `STRIX01-PR100` | Low-Medium: upstream may rework graph cache entirely. Keep hash mix; collision only resets warmup. | backend-ops runs show repeated `warmup reset/complete` cycles, no hang; model bench blocked (see below) |

## Validation 2026-10-05 (gfx1151, Windows HIP, TheRock 7.15)

- `test-backend-ops -o MUL_MAT`: 1891/1891 PASS (PR38)
- `test-backend-ops -o MUL_MAT_ID`: 931/931 PASS
- `test-backend-ops -o FLASH_ATTN_EXT`: 3991/3991 PASS (PR55)
- Full-model baseline (with all 3 ports), 2026-10-05, `llama-bench -ngl 99 -fa on -p 128 -n 32 -r 1`
  on `Qwen3.8-Flash-Next-UD-IQ4_XS` (87.24 GiB, 176.94 B params, ROCm gfx1151):
  pp128 = 211.72 t/s, tg32 = 20.21 t/s. Model is IQ4_XS (no Q6_K tensors, so PR38
  inactive here); run validates PR55 (FA256 on MMA path, no abort) and PR100 (graphs replay).
- Draft-file crash note: `mainline-mtp-Qwen3.8-Flash-Next-Q4_0.gguf` is an MTP **draft** GGUF
  (34 tensors, all `blk.48.*`), not a standalone model -- running it as `-m` segfaults
  (`0xC0000005`) in `ggml_mul` during graph build. Correct usage is `--model-draft` alongside
  the full model. GDB trace in `%TEMP%\opencode\gdb_bt.log`.

## Windows HIP build notes (gfx1151, TheRock 7.15, 2026-10-05)

Configure + build needs (powershell):

```powershell
$env:PATH = "C:\TheRock\build\bin;C:\TheRock\build\lib\llvm\bin;" + $env:PATH
$env:HIP_PATH = "C:\TheRock\build"
$env:HIP_DEVICE_LIB_PATH = "C:\TheRock\build\lib\llvm\amdgcn\bitcode"  # has oclc_abi_version_400.bc
$env:VSINSTALLDIR = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
$env:VCINSTALLDIR = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC"
$env:VCToolsInstallDir = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207"
cmake -S . -B build-strix01 -G Ninja -DGPU_TARGETS=gfx1151 -DGGML_HIP=ON `
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ `
  -DCMAKE_RC_COMPILER="C:/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64/rc.exe" `
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-strix01 --config Release --target test-backend-ops llama-bench llama-cli -j 16
```

Why: without `HIP_DEVICE_LIB_PATH`, clang HIP fails with
`cannot find ROCm device library`. Without pinning to VS2022 MSVC 14.44,
clang auto-picks VS18 MSVC 14.51 and dies in
`__clang_cuda_math_forward_declares.h` vs `cmath` `_CLANG_BUILTIN2`
(`__device__` vs `__host__ __device__` overloads).

## Queued (Tier 1 rest, from `strix-halo-easy-wins.md`)

| PR | Files | Note |
|---|---|---|
| halo-box/strix-llama.cpp#129 Vulkan small-M swap | `ggml/src/ggml-vulkan/ggml-vulkan.cpp` | only if shipping Vulkan on Windows |
| halo-box/strix-llama.cpp#27 Vulkan mat-vec chunk gating | `ggml/src/ggml-vulkan/ggml-vulkan.cpp` | expect fuzz vs upstream `2cdae802e` |

## Do-not-port-alone (needs base absent locally)

- #75 -> #123 -> #141 MMB gate chain needs MMB base (#63/#18). `ggml/src/ggml-cuda/` has no `mmb.*` locally (checked 2026-10-05).
- #82 + #96 QSA decode needs `qsa-decode.*` base.
- #101 MTP vocab is qwen35-only; #108 is the qwen4exp equivalent, still OPEN.
