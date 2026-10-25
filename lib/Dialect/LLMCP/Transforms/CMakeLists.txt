add_mlir_library(MLIRLLMCPTransforms
  LowerToLinalg.cpp
  LowerCausalConv1d.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/llmcp/Dialect/LLMCP/Transforms

  LINK_LIBS PUBLIC
    MLIRLLMCPDialect
    MLIRArithDialect
    MLIRBufferizationDialect
    MLIRIR
    MLIRLinalgDialect
    MLIRMathDialect
    MLIRPass
    MLIRSupport
    MLIRTensorDialect
    MLIRTransforms
    LLVMSupport
)
