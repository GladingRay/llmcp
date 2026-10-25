add_mlir_library(MLIRLLMCPModel
  ModelConfig.cpp
  Safetensors.cpp
  WeightLayout.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/llmcp/Model

  LINK_LIBS PUBLIC
    MLIRSupport
    LLVMSupport
)
