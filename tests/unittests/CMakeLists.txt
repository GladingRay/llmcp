add_executable(llmcp-unittests
  ModelConfigTest.cpp
  SafetensorsTest.cpp
  WeightLayoutTest.cpp
  ModelBuilderTest.cpp
  CacheLineAlignedVectorTest.cpp
  QuantizationTest.cpp)
target_sources(llmcp-unittests PRIVATE StreamingDecoderTest.cpp)

target_link_libraries(llmcp-unittests PRIVATE
  MLIRLLMCPModel
  llmcp-streaming-decoder
  GTest::gtest_main)

add_test(NAME llmcp-unittests COMMAND llmcp-unittests)
