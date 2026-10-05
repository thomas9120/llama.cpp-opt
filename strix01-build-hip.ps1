# Build strix01 for Strix Halo (gfx1151) with HIP/ROCm.
# Run from the repo root:  powershell -ExecutionPolicy Bypass -File .\build-hip-strix01.ps1
# Binaries land in build-strix01\bin\ (llama-server.exe, llama-cli.exe, llama-bench.exe, test-backend-ops.exe).
param(
    [string]$BuildDir = "build-strix01",
    [string]$GpuTargets = "gfx1151",
    [int]$Jobs = 16,
    [string]$Config = "Release"
)

$ErrorActionPreference = "Stop"

$HipRoot = "C:\TheRock\build"
$LlvmBin = Join-Path $HipRoot "lib\llvm\bin"

$env:PATH = "$HipRoot\bin;$LlvmBin;" + $env:PATH
$env:HIP_PATH = $HipRoot
# clang HIP needs the device lib dir (oclc_abi_version_400.bc lives here)
$env:HIP_DEVICE_LIB_PATH = Join-Path $HipRoot "lib\llvm\amdgcn\bitcode"
# Pin clang to VS2022 MSVC 14.44: VS18 14.51 headers clash with TheRock clang 23 HIP wrappers
$env:VSINSTALLDIR     = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
$env:VCINSTALLDIR     = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC"
$env:VCToolsInstallDir = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207"
$env:RC = "C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\rc.exe"

cmake -S . -B $BuildDir -G Ninja `
    -DGPU_TARGETS=$GpuTargets -DGGML_HIP=ON `
    -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ `
    -DCMAKE_RC_COMPILER="$env:RC" `
    -DCMAKE_BUILD_TYPE=$Config

cmake --build $BuildDir --config $Config `
    --target test-backend-ops llama-bench llama-cli llama-server -j $Jobs
