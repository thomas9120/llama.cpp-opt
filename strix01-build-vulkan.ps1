# Build strix01 with the Vulkan backend (validates the STRIX01-PR129 small-M port).
# Run from the repo root:  powershell -ExecutionPolicy Bypass -File .\build-vulkan-strix01.ps1
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

cmake --build $BuildDir --config $Config `
    --target test-backend-ops llama-bench llama-cli llama-fit-params llama-server -j $Jobs
