@echo off
REM Compile every .comp in this directory to .spv. Run after editing shaders.
REM The .spv files are checked in so the project builds without glslc on
REM other machines.
setlocal
set SCRIPT_DIR=%~dp0
pushd %SCRIPT_DIR%
for %%f in (*.comp) do (
    echo glslc %%f
    glslc -O "%%f" -o "%%~nf.spv" || goto :fail
)
popd
exit /b 0
:fail
popd
echo build.bat: glslc failed for %%f
exit /b 1
