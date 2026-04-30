@echo off
REM Compile every .comp under this directory tree to .spv next to its source.
REM Run after editing shaders. The .spv files are checked in so the project
REM builds without glslc on other machines. -I points at the shader root so
REM `#include "bf16.glsl"` resolves from any subfolder.
setlocal
set SCRIPT_DIR=%~dp0
pushd %SCRIPT_DIR%
for /R %%f in (*.comp) do (
    echo glslc %%f
    glslc -O --target-env=vulkan1.3 -I "%SCRIPT_DIR%." "%%f" -o "%%~dpnf.spv" || goto :fail
)
popd
exit /b 0
:fail
popd
echo build.bat: glslc failed for %%f
exit /b 1
