@echo off
rem Dopplers Bakery build: uses the toolchain in c:\workenv
set PATH=c:\workenv\w64devkit\bin;c:\workenv\ninja;c:\workenv\cmake-4.4.3-windows-x86_64\bin;%PATH%
cd /d %~dp0
if not exist build cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ || exit /b 1
cmake --build build %* || exit /b 1
if not exist dist mkdir dist
copy /y build\DopplersBakery.exe dist\ >nul
xcopy /y /d /q runtime\*.dll dist\ >nul
echo built dist\DopplersBakery.exe
