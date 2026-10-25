find_program(LLMCP_FILECHECK
  NAMES FileCheck
  HINTS /opt/homebrew/opt/llvm@20/bin)

find_program(LLMCP_LIT
  NAMES lit
  HINTS /Users/guorui/Work/models/venv314/bin)

set(LLMCP_OPT "${CMAKE_BINARY_DIR}/bin/llmcp-opt")
set(LLMCP_LIT_CFG "${CMAKE_CURRENT_SOURCE_DIR}/lit/lit.cfg.py")
set(LLMCP_TEST_EXEC_ROOT "${CMAKE_CURRENT_BINARY_DIR}/lit")
file(MAKE_DIRECTORY "${LLMCP_TEST_EXEC_ROOT}")

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/lit/lit.site.cfg.py.in"
  "${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py"
  @ONLY)

if(LLMCP_LIT AND LLMCP_FILECHECK)
  add_test(NAME llmcp-lit
    COMMAND ${LLMCP_LIT} -v "${CMAKE_CURRENT_BINARY_DIR}")
else()
  message(WARNING
    "lit or FileCheck not found; the llmcp-lit test will not be registered")
endif()

add_subdirectory(unittests)
