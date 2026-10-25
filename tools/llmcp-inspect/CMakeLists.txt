add_executable(llmcp-inspect llmcp-inspect.cpp)
target_link_libraries(llmcp-inspect PRIVATE
  MLIRLLMCPModel
  MLIRSupport
  LLVMSupport
)
