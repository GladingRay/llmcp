add_executable(llmcp-unittests
  ModelConfigTest.cpp
  SafetensorsTest.cpp
  WeightLayoutTest.cpp
  ModelBuilderTest.cpp
  CacheLineAlignedVectorTest.cpp)

target_link_libraries(llmcp-unittests PRIVATE
  MLIRLLMCPModel
  GTest::gtest_main)

add_test(NAME llmcp-unittests COMMAND llmcp-unittests)
