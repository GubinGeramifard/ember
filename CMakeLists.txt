cmake_minimum_required(VERSION 3.20)
project(ember LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Portable SIMD target. EMBER_NATIVE tunes for the build machine (fastest);
# turn it off for a portable binary that still requires AVX2 + FMA.
option(EMBER_NATIVE "Optimize for the local CPU (-march=native)" ON)

add_library(ember_flags INTERFACE)
target_compile_options(ember_flags INTERFACE
  $<$<CONFIG:Release>:-O3>
  -Wall -Wextra -Wpedantic)
if(EMBER_NATIVE)
  target_compile_options(ember_flags INTERFACE -march=native)
else()
  target_compile_options(ember_flags INTERFACE -mavx2 -mfma)
endif()

# Core engine library.
add_library(ember STATIC
  src/model.cpp
  src/tokenizer.cpp)
target_include_directories(ember PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(ember PUBLIC ember_flags)

# CLI entry point.
add_executable(ember_cli src/main.cpp)
target_link_libraries(ember_cli PRIVATE ember)
set_target_properties(ember_cli PROPERTIES OUTPUT_NAME ember)

# Statically link the GCC/libstdc++ runtime so the binary is self-contained and
# does not pick up a mismatched libstdc++-6.dll from elsewhere on the system PATH.
if(MINGW)
  target_link_options(ember_cli PRIVATE -static -static-libgcc -static-libstdc++)
endif()
