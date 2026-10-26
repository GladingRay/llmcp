add_executable(llmcp-microbench cpu_microbench.cpp)
if(APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
  target_compile_options(llmcp-microbench PRIVATE
    -O3 -mcpu=apple-m2 -fno-vectorize -fno-slp-vectorize)
endif()
target_link_libraries(llmcp-microbench PRIVATE LLVMSupport)
