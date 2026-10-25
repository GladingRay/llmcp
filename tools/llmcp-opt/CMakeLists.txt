add_executable(llmcp-opt llmcp-opt.cpp)
target_link_libraries(llmcp-opt PRIVATE
  MLIRLLMCPDialect
  MLIRLLMCPTransforms
  MLIROptLib
  MLIRParser
  MLIRSupport
  LLVMSupport
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRTensorDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTensorDialect
)
