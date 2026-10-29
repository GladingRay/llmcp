add_executable(llmcp-benchmark llmcp-benchmark.cpp)
target_link_libraries(llmcp-benchmark PRIVATE
  MLIRLLMCPRuntime
  MLIRSupport
  LLVMSupport
)

add_executable(llmcp-matmul-benchmark llmcp-matmul-benchmark.cpp)
target_link_libraries(llmcp-matmul-benchmark PRIVATE
  MLIRSupport
  LLVMSupport
)
