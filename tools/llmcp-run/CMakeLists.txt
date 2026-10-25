add_executable(llmcp-run llmcp-run.cpp)
target_link_libraries(llmcp-run PRIVATE
  MLIRSupport
  LLVMSupport
  utf8proc
)
if(APPLE)
  target_link_libraries(llmcp-run PRIVATE "-framework Accelerate")
endif()
