add_mlir_library(MLIRLLMCPRuntime
  AotRuntime.cpp
  WeightArena.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/llmcp/Runtime

  LINK_LIBS PUBLIC
    MLIRSupport
    LLVMSupport
)
