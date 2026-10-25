add_executable(llmcp-pack llmcp-pack.cpp)
target_link_libraries(llmcp-pack PRIVATE
  MLIRLLMCPModel
  MLIRSupport
  LLVMSupport
)
