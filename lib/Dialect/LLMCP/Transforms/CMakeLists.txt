add_mlir_library(MLIRLLMCPTransforms
  LowerToLinalg.cpp
  LowerCausalConv1d.cpp
  LowerGatedDeltaNet.cpp
  LowerAttention.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/llmcp/Dialect/LLMCP/Transforms

  LINK_LIBS PUBLIC
    MLIRLLMCPDialect
    MLIRArithDialect
    MLIRBufferizationDialect
    MLIRIR
    MLIRLinalgDialect
    MLIRLLVMDialect
    MLIRMathDialect
    MLIRPass
    MLIRSupport
    MLIRTensorDialect
    MLIRVectorDialect
    MLIRTransforms
    LLVMSupport
)
