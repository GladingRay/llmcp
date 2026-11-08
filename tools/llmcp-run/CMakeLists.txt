find_program(CARGO_EXECUTABLE cargo REQUIRED)
option(LLMCP_CARGO_OFFLINE "Build tokenizers-cpp Rust deps from the local cargo cache" ON)

set(TOKENIZERS_CPP_DIR "${PROJECT_SOURCE_DIR}/third_party/tokenizers-cpp")
set(TOKENIZERS_RUST_DIR "${CMAKE_CURRENT_BINARY_DIR}/tokenizers-rust")
set(TOKENIZERS_RUST_LIB "${TOKENIZERS_RUST_DIR}/target/release/libtokenizers_c.a")

# tokenizers-cpp's Rust wrapper currently pins tokenizers 0.21.2. The local
# crates.io cache has 0.22.2; use that compatible version for the HF-only build.
file(COPY "${TOKENIZERS_CPP_DIR}/rust/" DESTINATION "${TOKENIZERS_RUST_DIR}")
file(READ "${TOKENIZERS_RUST_DIR}/Cargo.toml" TOKENIZERS_CARGO)
string(REPLACE "0.21.2" "0.22.2" TOKENIZERS_CARGO "${TOKENIZERS_CARGO}")
file(WRITE "${TOKENIZERS_RUST_DIR}/Cargo.toml" "${TOKENIZERS_CARGO}")

if(APPLE)
  # Let xcrun select the SDK installed on the runner instead of pinning a
  # specific macOS SDK version.
  execute_process(
    COMMAND xcrun --sdk macosx --show-sdk-path
    OUTPUT_VARIABLE TOKENIZERS_SDKROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT TOKENIZERS_SDKROOT)
    execute_process(
      COMMAND xcrun --show-sdk-path
      OUTPUT_VARIABLE TOKENIZERS_SDKROOT
      OUTPUT_STRIP_TRAILING_WHITESPACE)
  endif()
  if(NOT TOKENIZERS_SDKROOT)
    message(FATAL_ERROR "xcrun could not locate a macOS SDK")
  endif()
  set(TOKENIZERS_CARGO_ENV
    ${CMAKE_COMMAND} -E env
    SDKROOT=${TOKENIZERS_SDKROOT})
  if(CMAKE_OSX_DEPLOYMENT_TARGET)
    list(APPEND TOKENIZERS_CARGO_ENV
      MACOSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET})
  endif()
else()
  set(TOKENIZERS_CARGO_ENV ${CMAKE_COMMAND} -E env)
endif()

add_custom_command(
  OUTPUT "${TOKENIZERS_RUST_LIB}"
  COMMAND ${TOKENIZERS_CARGO_ENV}
          ${CARGO_EXECUTABLE} build --release $<$<BOOL:${LLMCP_CARGO_OFFLINE}>:--offline>
  WORKING_DIRECTORY "${TOKENIZERS_RUST_DIR}"
  DEPENDS "${TOKENIZERS_RUST_DIR}/Cargo.toml" "${TOKENIZERS_RUST_DIR}/src/lib.rs"
  COMMENT "Building tokenizers-cpp Rust static library")
add_custom_target(tokenizers_rust DEPENDS "${TOKENIZERS_RUST_LIB}")

add_library(tokenizers_hf STATIC
  "${TOKENIZERS_CPP_DIR}/src/huggingface_tokenizer.cc")
target_include_directories(tokenizers_hf PUBLIC
  "${TOKENIZERS_CPP_DIR}/include")
add_dependencies(tokenizers_hf tokenizers_rust)
target_link_libraries(tokenizers_hf PUBLIC "${TOKENIZERS_RUST_LIB}")
if(APPLE)
  target_link_libraries(tokenizers_hf PUBLIC "-framework Security")
endif()

add_library(llmcp-streaming-decoder STATIC StreamingDecoder.cpp)
target_include_directories(llmcp-streaming-decoder PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(llmcp-streaming-decoder PUBLIC tokenizers_hf)

add_executable(llmcp-run llmcp-run.cpp)
target_link_libraries(llmcp-run PRIVATE
  llmcp-streaming-decoder
  MLIRLLMCPRuntime
  MLIRSupport
  LLVMSupport
)
if(APPLE)
  target_link_libraries(llmcp-run PRIVATE "-framework Accelerate")
endif()
