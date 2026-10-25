add_executable(llmcp-opt llmcp-opt.cpp)
target_link_libraries(llmcp-opt PRIVATE
  MLIRLLMCPDialect
  MLIROptLib
  MLIRParser
  MLIRSupport
  LLVMSupport
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTensorDialect
)
