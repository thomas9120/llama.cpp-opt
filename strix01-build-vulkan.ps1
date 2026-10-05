# Build strix01 with the Vulkan backend (validates the STRIX01-PR129 small-M port).
# Run from the repo root:  powershell -ExecutionPolicy Bypass -File .\strix01-build-vulkan.ps1
# Binaries land in build-strix01-vk\bin\ (llama-server.exe, llama-cli.exe, llama-fit-params.exe, llama-bench.exe, test-backend-ops.exe).
param(
    [string]$BuildDir = "build-strix01-vk",
    [int]$Jobs = 16,
    [string]$Config = "Release",
    [string]$VulkanSdk = "C:\VulkanSDK\1.4.350.0"
)

$ErrorActionPreference = "Stop"

$env:PATH = "C:\TheRock\build\lib\llvm\bin;" + $env:PATH
$env:VULKAN_SDK = $VulkanSdk
# The nested vulkan-shaders-gen configure does not inherit CMAKE_RC_COMPILER, it needs RC
$env:RC = "C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\rc.exe"

cmake -S . -B $BuildDir -G Ninja `
    -DGGML_VULKAN=ON -DGGML_HIP=OFF `
    -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ `
    -DCMAKE_RC_COMPILER="$env:RC" `
    -DCMAKE_BUILD_TYPE=$Config

# Fail fast if CMAKE_BUILD_TYPE did not expand (e.g. lines pasted into cmd.exe where $Config stays literal)
$CachedType = Select-String -Path (Join-Path $BuildDir 'CMakeCache.txt') -Pattern '^CMAKE_BUILD_TYPE:STRING=(.+)$' |
    ForEach-Object { $_.Matches.Groups[1].Value }
if ($CachedType -ne $Config) { throw "CMAKE_BUILD_TYPE mismatch: cache has '$CachedType', expected '$Config'. Run this script from PowerShell, not cmd.exe." }

cmake --build $BuildDir --config $Config `
    --target test-backend-ops llama-bench llama-cli llama-fit-params llama-server -j $Jobs
