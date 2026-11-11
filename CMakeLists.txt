cmake_minimum_required(VERSION 3.20)
project(DopplersBakery CXX C)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# The inference runtime is the official llama.cpp release build of the same commit as third_party/llama.cpp
# (headers only are used from there). Its DLLs live in runtime/ and are copied next to the exe:
#   llama.dll ggml.dll ggml-base.dll ggml-cpu-*.dll  (CPU, picked per machine at run time)
#   ggml-cuda.dll + cudart/cublas  or  ggml-vulkan.dll  (GPU, optional)
set(RT ${CMAKE_SOURCE_DIR}/runtime)

file(GLOB_RECURSE BAKERY_SRC CONFIGURE_DEPENDS src/*.cpp)
add_executable(DopplersBakery WIN32 ${BAKERY_SRC} src/app.rc)
target_include_directories(DopplersBakery PRIVATE src third_party/llama.cpp/include third_party/llama.cpp/ggml/include)
# Single-exe build: -DBAKERY_STATIC_DIR=<llama.cpp static build dir> (CPU + Vulkan linked in, see docs/STATIC-BUILD.md)
if(BAKERY_STATIC_DIR)
  set(RUNTIME_LIBS ${BAKERY_STATIC_DIR}/src/libllama.a ${BAKERY_STATIC_DIR}/ggml/src/ggml.a ${BAKERY_STATIC_DIR}/ggml/src/ggml-cpu.a
      ${BAKERY_STATIC_DIR}/ggml/src/ggml-vulkan/ggml-vulkan.a ${BAKERY_STATIC_DIR}/ggml/src/ggml-base.a ${BAKERY_STATIC_VULKAN_LIB})
else()
  set(RUNTIME_LIBS ${RT}/llama.dll ${RT}/ggml.dll ${RT}/ggml-base.dll)
endif()
target_link_libraries(DopplersBakery PRIVATE ${RUNTIME_LIBS}
  comctl32 msimg32 mfplat mfreadwrite mfuuid winhttp ws2_32 shlwapi windowscodecs ole32 shell32 comdlg32 winmm uuid gdi32 user32 -static-libgcc -static-libstdc++)
target_compile_definitions(DopplersBakery PRIVATE UNICODE _UNICODE WIN32_LEAN_AND_MEAN NOMINMAX)
