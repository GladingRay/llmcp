add_executable(llmcp-benchmark llmcp-benchmark.cpp)
target_link_libraries(llmcp-benchmark PRIVATE
  MLIRSupport
  LLVMSupport
)
