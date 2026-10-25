add_mlir_dialect_library(MLIRLLMCPDialect
  LLMCPDialect.cpp
  LLMCPOps.cpp
  LLMCPTypes.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/llmcp/Dialect/LLMCP/IR

  DEPENDS
    MLIRLLMCPOpsIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRSupport
    MLIRSideEffectInterfaces
    MLIRArithDialect
    MLIRMathDialect
    MLIRTensorDialect
    MLIRLinalgDialect
    MLIRSCFDialect
)
