# Build the PTX for ggml's flash_attn_ext_f16 MMA kernel.
#
# Run from the repo root:
#   .\backends\cuda\kernels\attention\ggml_mma\build_ptx.ps1
#
# Output: attention_mma_d256_ncols2_4.ptx (commit alongside the .cu).
#
# Why offline nvcc (vs NVRTC):
#   ggml's fattn-mma-f16.cuh transitively includes ggml.h / ggml-impl.h /
#   ggml-cuda.h via common.cuh. Those are host-side ggml infrastructure that
#   NVRTC can't find. Offline nvcc with -Iggml/{include,src,src/ggml-cuda}
#   resolves them just fine.

$ErrorActionPreference = "Stop"

# nvcc on Windows needs cl.exe in PATH for the host compiler portion.
# Adjust this if your VS version differs.
$ClPath = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.38.33130\bin\Hostx64\x64"
if (-not (Test-Path "$ClPath\cl.exe")) {
    Write-Error "cl.exe not found at $ClPath. Update `$ClPath in this script."
}
$env:Path = "$ClPath;$env:Path"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = Resolve-Path "$ScriptDir\..\..\..\..\..\"

Push-Location $RepoRoot
try {
    nvcc `
        -arch=sm_86 `
        -ptx `
        -std=c++17 `
        -O3 `
        --extended-lambda `
        -diag-suppress=177 `
        -diag-suppress=39 `
        -Iggml/include `
        -Iggml/src `
        -Iggml/src/ggml-cuda `
        backends/cuda/kernels/attention/ggml_mma/wrapper.cu `
        -o backends/cuda/kernels/attention/ggml_mma/attention_mma_d256_ncols2_4.ptx
    if ($LASTEXITCODE -ne 0) { Write-Error "nvcc failed (exit $LASTEXITCODE)" }
    Write-Host "OK: backends/cuda/kernels/attention/ggml_mma/attention_mma_d256_ncols2_4.ptx"
} finally {
    Pop-Location
}
