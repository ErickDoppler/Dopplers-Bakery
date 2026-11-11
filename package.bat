@echo off
rem Assembles a clean, shareable folder: release\DopplersBakery\  (exe + runtime DLLs, no models, no settings, no CUDA).
rem Run build.bat first. The GPU backend shipped is Vulkan (works on NVIDIA, AMD and Intel); CPU is the fallback.
setlocal
cd /d %~dp0
set OUT=release\DopplersBakery
if exist "%OUT%" rmdir /s /q "%OUT%"
mkdir "%OUT%" || exit /b 1
copy /y build\DopplersBakery.exe "%OUT%\" >nul || exit /b 1
for %%f in (runtime\*.dll) do (
  echo %%~nxf | findstr /i /b "cublas cudart ggml-cuda" >nul || copy /y "%%f" "%OUT%\" >nul
)
copy /y README-DIST.txt "%OUT%\README.txt" >nul
copy /y third_party\llama.cpp\LICENSE "%OUT%\LICENSE-llama.cpp.txt" >nul
echo packaged %OUT%
