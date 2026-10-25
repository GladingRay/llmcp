add_executable(llmcp-compile llmcp-compile.cpp)
target_link_libraries(llmcp-compile PRIVATE
  MLIRLLMCPModel
  MLIRLLMCPDialect
  MLIRIR
  MLIRSupport
  LLVMSupport
)
