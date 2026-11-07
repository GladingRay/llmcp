cmake_minimum_required(VERSION 3.20)
project(llmcp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(LLMCP_BUILD_TESTS "Build llmcp tests" ON)
option(LLMCP_ENABLE_BFDOT
       "Enable Arm BF16 dot-product intrinsics when the host supports them" ON)

# bfdot is not available on every Apple Silicon generation (for example some
# M1 runners). Detect it at configure time instead of emitting an intrinsic
# that the target backend cannot select.
if(APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
  execute_process(
    COMMAND sysctl -n hw.optional.arm.FEAT_BF16
    OUTPUT_VARIABLE LLMCP_BF16_FEATURE
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE LLMCP_BF16_RESULT)
  if(NOT LLMCP_BF16_RESULT EQUAL 0 OR NOT LLMCP_BF16_FEATURE STREQUAL "1")
    set(LLMCP_ENABLE_BFDOT OFF CACHE BOOL
        "Enable Arm BF16 dot-product intrinsics when the host supports them"
        FORCE)
  endif()
endif()
if(LLMCP_ENABLE_BFDOT)
  add_compile_definitions(LLMCP_ENABLE_BFDOT=1)
endif()

# Directory of the real Qwen3.5 model used by integration tests. It can be
# overridden at configure time (-DLLMCP_MODEL_DIR=...) or through the
# LLMCP_MODEL_DIR environment variable.
set(LLMCP_MODEL_DIR "$ENV{LLMCP_MODEL_DIR}" CACHE PATH
    "Path to the Qwen3.5 model directory used by integration tests")
if(NOT LLMCP_MODEL_DIR)
  set(LLMCP_MODEL_DIR "/Users/guorui/Work/models/qwen3.5-4b" CACHE PATH
      "Path to the Qwen3.5 model directory used by integration tests" FORCE)
endif()

# The SDK's libz.1.tbd is not understood by the Homebrew clang 20 linker on
# this machine. Use the Homebrew zlib instead; it is already a dependency of
# llvm@20.
if(APPLE AND EXISTS "/opt/homebrew/opt/zlib/lib/libz.dylib")
  set(ZLIB_ROOT "/opt/homebrew/opt/zlib")
  set(ZLIB_INCLUDE_DIR "/opt/homebrew/opt/zlib/include" CACHE PATH "" FORCE)
  set(ZLIB_LIBRARY_RELEASE "/opt/homebrew/opt/zlib/lib/libz.dylib"
      CACHE FILEPATH "" FORCE)
  set(ZLIB_LIBRARY_DEBUG "/opt/homebrew/opt/zlib/lib/libz.dylib"
      CACHE FILEPATH "" FORCE)
endif()

find_package(MLIR REQUIRED CONFIG)

# Homebrew LLVM's imported targets expose the macOS SDK include directory
# through ZLIB/zstd. With Homebrew clang and a newer SDK this puts libc's
# /usr/include ahead of libc++'s <math.h>, breaking <cmath>. The prebuilt LLVM
# libraries already contain their zlib/zstd code, so we do not need these
# include paths in llmcp sources.
if(APPLE)
  foreach(imported_target ZLIB::ZLIB zstd::libzstd_shared)
    if(TARGET ${imported_target})
      set_target_properties(${imported_target} PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "")
    endif()
  endforeach()
endif()

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)
include(TableGen)
include(AddMLIR)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
add_definitions(${LLVM_DEFINITIONS})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(third_party)

# TableGen must run from the include side; the dialect library lives in lib/.
add_subdirectory(include/llmcp/Dialect/LLMCP/IR)

add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(microbench)

if(LLMCP_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
