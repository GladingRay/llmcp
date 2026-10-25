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

if(EXISTS "/Users/guorui/Work/models/qwen3.5-4b/config.json")
  add_test(NAME llmcp-dump-real
    COMMAND $<TARGET_FILE:llmcp-dump>
      --model /Users/guorui/Work/models/qwen3.5-4b
      --verify-module --print-stats
      -o "${CMAKE_BINARY_DIR}/qwen35_graph.mlir")
  set_tests_properties(llmcp-dump-real PROPERTIES TIMEOUT 120)

  set(LLMCP_DUMP "${CMAKE_BINARY_DIR}/bin/llmcp-dump")
  set(LLMCP_OPT "${CMAKE_BINARY_DIR}/bin/llmcp-opt")
  set(LLMCP_GRAPH "${CMAKE_BINARY_DIR}/qwen35_graph.mlir")
  set(LLMCP_LINALG "${CMAKE_BINARY_DIR}/qwen35_linalg.mlir")
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/lower-real.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/lower-real.cmake"
    @ONLY)
  add_test(NAME llmcp-lower-real
    COMMAND ${CMAKE_COMMAND} -P
            "${CMAKE_CURRENT_BINARY_DIR}/lower-real.cmake")
  set_tests_properties(llmcp-lower-real PROPERTIES TIMEOUT 120)

  add_test(NAME llmcp-pack-real
    COMMAND $<TARGET_FILE:llmcp-pack>
      --model /Users/guorui/Work/models/qwen3.5-4b --dry-run)
  set_tests_properties(llmcp-pack-real PROPERTIES TIMEOUT 120)

  add_test(NAME llmcp-tokenizer
    COMMAND $<TARGET_FILE:llmcp-run> --tokenize-only --text Hello
      --tokenizer /Users/guorui/Work/models/qwen3.5-4b/tokenizer.json
      --expect-token 9419)
  set_tests_properties(llmcp-tokenizer PROPERTIES TIMEOUT 120)

  add_test(NAME llmcp-compile-real
    COMMAND $<TARGET_FILE:llmcp-compile>
      --model /Users/guorui/Work/models/qwen3.5-4b
      --emit=llvm-ir --out "${CMAKE_BINARY_DIR}/qwen35_aot_test.ll")
  set_tests_properties(llmcp-compile-real PROPERTIES TIMEOUT 300)
endif()

find_program(LLMCP_PYTHON
  NAMES python3
  HINTS /Users/guorui/Work/models/venv314/bin)
find_program(LLMCP_MLIR_OPT NAMES mlir-opt HINTS /opt/homebrew/opt/llvm@20/bin)
find_program(LLMCP_MLIR_TRANSLATE NAMES mlir-translate
  HINTS /opt/homebrew/opt/llvm@20/bin)
find_program(LLMCP_CLANG NAMES clang++ HINTS /opt/homebrew/opt/llvm@20/bin)
set(LLMCP_RUNNER_LIB_DIR /opt/homebrew/opt/llvm@20/lib)

if(LLMCP_PYTHON AND LLMCP_MLIR_OPT AND LLMCP_MLIR_TRANSLATE AND LLMCP_CLANG)
  add_test(NAME llmcp-numeric
    COMMAND ${LLMCP_PYTHON}
      "${CMAKE_CURRENT_SOURCE_DIR}/numeric/run_numeric.py"
      --llmcp-opt "${LLMCP_OPT}"
      --mlir-opt "${LLMCP_MLIR_OPT}"
      --mlir-translate "${LLMCP_MLIR_TRANSLATE}"
      --clang "${LLMCP_CLANG}"
      --runner-lib-dir "${LLMCP_RUNNER_LIB_DIR}"
      --workdir "${CMAKE_CURRENT_BINARY_DIR}/numeric")
  set_tests_properties(llmcp-numeric PROPERTIES TIMEOUT 120)
else()
  message(WARNING "numeric test dependencies not found; llmcp-numeric skipped")
endif()

add_subdirectory(unittests)
