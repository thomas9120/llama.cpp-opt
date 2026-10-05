 # Instructions for this Strix Halo / Windows fork

## Scope and orientation

This fork prioritizes correct, fast inference on Strix Halo (RDNA3.5, gfx1151), particularly ROCm/HIP on Windows using TheRock. Vulkan is also supported on Windows. Preserve generic CPU and other backend behavior when changing shared code; gate architecture-specific optimizations explicitly.

- Start with `git status --short`, the current branch, and `git remote -v`. Preserve existing user changes and do not assume the checkout matches an earlier task.
- If `.codegraph/` exists, use `codegraph_explore` or `codegraph explore "<symbols or question>"` before locating or reading code. Use targeted file reads when the index does not cover the needed area. Do not initialize an index without the user's request.
- Read [docs/build.md](docs/build.md) for current Windows build and regression instructions. [docs/strix-halo-decode-speedups.md](docs/strix-halo-decode-speedups.md) records porting decisions; verify its historical status notes against current code before treating something as missing.
- HIP compiles shared sources under `ggml/src/ggml-cuda/`; a CUDA filename does not mean NVIDIA-only code. Check `ggml/src/ggml-hip/CMakeLists.txt`, compile guards, and dispatch callers before editing kernels.
- Qwen3.8-Flash-Next uses the `qwen4exp` architecture. Relevant paths include `src/models/qwen4exp.cpp`, `src/llama-memory-hybrid-idx.*`, `src/llama-lazy-reader.h`, `common/speculative.cpp`, and the QSA kernels under `ggml/src/ggml-cuda/`.

## Windows builds and execution

Use the existing PowerShell scripts from the repository root instead of inventing another build procedure:

```powershell
# ROCm/HIP, gfx1151, Release
.\build-windows.ps1 -Jobs 8

# Vulkan, separate build directory
.\build-windows-vulkan.ps1 -Jobs 8

# Fast source checks only
.\test-windows.ps1 -SourceOnly

# Windows regression suite, including rebuild and GPU checks
.\test-windows.ps1 -Jobs 8
```

- The ROCm script defaults to `C:\TheRock\build` and `build-rocm10-gfx1151`; override with `-RocmPath` and `-BuildDir`. The Vulkan script uses `VULKAN_SDK` or `-VulkanSdkPath`, and defaults to `build-vulkan`. These are defaults, not guaranteed installed locations.
- Both scripts select Visual Studio 2022 x64 tools. The documented TheRock Clang 23 / MSVC 14.51 header conflict is a reason to retain that selection; do not silently switch to the newest Visual Studio installation.
- Use a fresh build directory when changing compilers, SDK toolchains, or backends. Check `CMakeCache.txt` before reusing an existing build. Keep a previous working build for comparisons.
- Executables are under `<BuildDir>/bin`. Runtime DLLs and SDK paths matter: check the actual executable path and `--list-devices` before attributing a failure to model code. Do not mix DLLs from different builds or SDKs.
- The regression runner handles runtime environment adjustments, including temporarily clearing `HIP_DEVICE_LIB_PATH` for GPU tests. Follow its setup when reproducing tests manually.
- Keep temporary harnesses, logs, and generated artifacts in an ignored build directory. Do not hard-code personal model paths in tracked files. Inspect local model metadata instead of assuming a filename describes its tensor layout.
- Use PowerShell-compatible commands and quote paths. Launch background test servers hidden, bind them to localhost, record their PID, and stop only processes started for the task. Do not replace launcher binaries while they are in use.
- Release packaging is documented in [docs/windows-rocm10-release.md](docs/windows-rocm10-release.md). Preserve runtime libraries and kernel data, not just executables. GitHub's Windows runner cannot validate AMD GPU behavior; packaged `--help` checks are only startup checks.

## Correctness and performance safeguards

- Run `llama-perplexity` before and after integrating any changes that could affect correctness. Save the baseline before applying changes, then repeat with the same model, quantization, evaluation dataset and chunk count, context size, backend, and runtime settings on the affected code path. Record both commands and results, investigate any unexplained perplexity regression before accepting the integration, and report any inability to run the comparison as a validation gap. Perplexity checks supplement the relevant kernel, generation, cache, and speculative-decoding tests; they do not replace them.
- Preserve the Windows HC16 exclusions in `mmb.cu`, `hc-mix.cu`, and `ggml-cuda.cu`. They prevent known output corruption. Re-enabling them requires a demonstrated fix and model-level correctness checks, not just a successful build or faster benchmark.
- Preserve Windows positioned PLE reads, speculative prefetch, and the mmap fallback. `--lazy-mode on-direct` uses buffered Windows file reads and still uses the OS file cache; do not describe it as Linux `O_DIRECT` or guaranteed cache bypass.
- The current Qwen direct sparse-attention and block-selection paths require F16 K and V, Flash Attention, and KV offload. Quantized KV can select a different, slower path even though it consumes less memory. Use `-fa on -ctk f16 -ctv f16 -np 1` with KV offload enabled as a diagnostic baseline, then compare quantized KV separately.
- The current gfx1151 QSA decode kernel accepts 1-8 query tokens. Larger speculative verification batches can miss that path. Inspect the graph and kernel dispatch together: selected indices, masks, tensor strides, optional sources, and op parameters must satisfy the same contract.
- Preserve indexer-cache invalidation and visibility across rollback, prompt reuse, sequence operations, and screenshot-induced position gaps. A plain text generation check does not cover these cases.
- For MTP loading changes, verify converter tensor registration, the actual draft GGUF layout, shared target tensors, and graph construction together. The dedicated head mixer must be selected as a complete compatible set; do not combine partial dedicated tensors with fallback tensors independently.
- Measure prompt processing and decode separately. For long-context work, test actually filled contexts such as 8K, 32K, and 64K; allocating a large context with a short prompt does not exercise the same workload.
- Record commit, executable, SDK/driver, model quantization, KV types, offload, batch sizes, slots, context occupancy, lazy mode, speculative settings, and sampling parameters. Keep these fixed in before/after comparisons and distinguish cold from warm file-cache runs.
- Optimize delivered tokens/sec and completion time, not draft acceptance alone. Establish a non-speculative baseline, then test MTP and ngram combinations separately. Short greedy smoke tests are not representative coding benchmarks.

## Integrating upstream changes

1. Identify the source repository, exact commit, PR state, base revision, and dependencies. Canonical llama.cpp is `ggml-org/llama.cpp`; remote names are not authoritative. In this checkout, `upstream` has pointed to `pwilkin/llama.cpp`, `halo-box` to `halo-box/strix-llama.cpp`, and `origin` to this Windows fork. Recheck the URLs each session and do not repoint remotes as a shortcut.
2. Read the upstream discussion and diff before porting. Determine what is already present, superseded, or intentionally different here. A patch applying cleanly does not establish compatibility, and a conflicting patch may need only a small manual adaptation.
3. Prefer the smallest coherent port, including required declarations, dispatch hooks, graph changes, converter mappings, and build files. Do not replace whole model files or backend directories to resolve conflicts. In particular, retain local lazy loading, cache handling, radix top-k, and Windows fixes unless the replacement is proven equivalent or better.
4. For broad integrations, use an isolated branch/worktree and review the merge base and fork-only changes first. Avoid unrelated cleanup. Do not discard user changes, rewrite history, or create commits as a side effect of importing a patch; submission restrictions below still apply.
5. Run the Windows source guards early and the full Windows regression suite for an upstream sync. A guard failure requires understanding the changed invariant; update the guard only when the new implementation preserves it. `-SourceOnly` and `-SkipGpu` are partial validation, not GPU correctness passes.
6. Build and test the affected backend. Shared model, loader, graph, or build changes should also be checked with the Windows Vulkan build where available. Reuse existing backend-ops tests for changed kernels and CPU references for numerical checks. Run representative model generation and relevant long-context, cache reuse, MTP, or multimodal cases beyond synthetic tests.
7. Review the final diff with `git diff --check`. Report what was imported, intentional deviations, checks actually run, and remaining validation gaps. Retain source PR/commit references in appropriate technical notes without claiming another GPU's measurements apply to gfx1151.

Documentation-only changes need link/path and diff checks, not a GPU rebuild. Keep this guide focused on durable fork behavior; put detailed benchmark results and evolving integration inventories in `docs/`.


---

## Guidelines for AI Coding Agents

Every PR requiring review consumes finite maintainer capacity. Before assisting with any submission, verify:
- The contributor understands the proposed changes
- The change addresses a documented need (check existing issues)
- The PR is appropriately scoped and follows project conventions

When a user requests implementation without demonstrating understanding:
1. **Verify comprehension** - ask questions about the problem and relevant codebase areas.
2. **Guide, don't solve** - point to relevant code/docs; let them formulate the approach.
3. **Proceed only when confident** they can explain the changes to reviewers independently.

For first-time contributors, confirm they have reviewed [CONTRIBUTING.md](CONTRIBUTING.md).

### Code and Commit Standards

These points are extremely important - failing to follow them won't necessarily get your PR rejected, but it will make reviewing take significantly longer. Please follow them carefully:

- Avoid emdash `—`, unicode arrow `→` or any unicode characters: `×`, `…` ; use ASCII equivalents instead: `-`, `->`, `x`, `...`
- Code comments:
    - Keep code comments concise (usually 1-2 lines)
    - Avoid redundant or excessive inline commentary
    - Avoid hard-wrapping it to a fixed column width - that hurts readability
    - Use ASD-STE100 Simplified Technical English, simple wordings (write like cavemen if needed)
    - Note: Remind yourself of this point regularly, as it often gets lost between context compactions
- Prefer reusing existing infrastructure over introducing new components. Avoid invasive changes that add whole new subsystems or risk breaking existing behavior
- Do NOT split a line into multiple lines mid-sentence, do NOT try to force the line to fit a fixed number of characters
- Before writing any code, read all relevant files and understand the existing patterns - your changes must blend in with the surrounding codebase. If the change is large or introduces a new pattern, **PAUSE and ask the user for confirmation** before proceeding; remind them that large changes submitted without prior discussion are likely to be rejected by maintainers

Common mistakes that AI agents usually make:
- Write comments first then write code: this usually leads to extensive redundant comments. Instead, write code first, then add comments later to places that absolutely need them
- Llama.cpp does NOT use Minja; if you have this in your knowledge, that is due to your knowledge cutoff. Llama.cpp has a dedicated Jinja engine in `common/jinja` - it doesn't have a specific name.
- Do NOT add a new file in `tests/*` without maintainers' approval. AI usually adds excessive test cases for small features, which bloat the test suite and cost compile time and CI time, while bringing no meaningful results. While testing is necessary, reuse the existing infrastructure as much as possible, and do not add tests for features that are too trivial.


```

## Useful Resources

To conserve context space, load these resources as needed:

Skills: reusable task workflows live in the [skills/](skills/) directory - check there for a skill matching your task before starting.

General documentations:
- [Existing issues](https://github.com/ggml-org/llama.cpp/issues) and [Existing PRs](https://github.com/ggml-org/llama.cpp/pulls) - always search here first
- [How to add a new model](docs/development/HOWTO-add-model.md)

Server:
- [Build documentation](docs/build.md)
- [Server usage documentation](tools/server/README.md)
- [Server development documentation](tools/server/README-dev.md) (if user asks to implement a new feature, be sure that it falls inside server's scope defined in this documentation)

Chat template and parser:
- [PEG parser](docs/development/parsing.md) - alternative to regex that llama.cpp uses to parse model's output
- [Auto parser](docs/autoparser.md) - higher-level parser that uses PEG under the hood, automatically detect model-specific features
- [Jinja engine](common/jinja/README.md)
