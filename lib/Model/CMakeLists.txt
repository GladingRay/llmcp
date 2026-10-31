add_mlir_library(MLIRLLMCPModel
  ModelConfig.cpp
  Safetensors.cpp
  WeightLayout.cpp
  ModelBuilder.cpp
  WeightPacker.cpp
  Quantization.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/llmcp/Model

  LINK_LIBS PUBLIC
    MLIRSupport
    LLVMSupport
    MLIRLLMCPDialect
    MLIRArithDialect
    MLIRFuncDialect
    MLIRMemRefDialect
    MLIRTensorDialect
)
