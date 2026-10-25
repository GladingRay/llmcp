add_executable(llmcp-dump llmcp-dump.cpp)
target_link_libraries(llmcp-dump PRIVATE
  MLIRLLMCPModel
  MLIRLLMCPDialect
  MLIRIR
  MLIRSupport
  LLVMSupport
)
