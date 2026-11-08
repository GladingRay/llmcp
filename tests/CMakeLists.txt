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

if(EXISTS "${LLMCP_MODEL_DIR}/config.json")
  add_test(NAME llmcp-dump-real
    COMMAND $<TARGET_FILE:llmcp-dump>
      --model "${LLMCP_MODEL_DIR}"
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
      --model "${LLMCP_MODEL_DIR}" --dry-run)
  set_tests_properties(llmcp-pack-real PROPERTIES TIMEOUT 120)

  add_test(NAME llmcp-pack-real-q8
    COMMAND $<TARGET_FILE:llmcp-pack>
      --model "${LLMCP_MODEL_DIR}" --quant q8_0 --dry-run)
  set_tests_properties(llmcp-pack-real-q8 PROPERTIES TIMEOUT 120)

  add_test(NAME llmcp-tokenizer
    COMMAND $<TARGET_FILE:llmcp-run> --tokenize-only --text Hello
      --tokenizer "${LLMCP_MODEL_DIR}/tokenizer.json"
      --expect-token 9419)
  set_tests_properties(llmcp-tokenizer PROPERTIES TIMEOUT 120)

  add_test(NAME llmcp-chat-template
    COMMAND $<TARGET_FILE:llmcp-run> --tokenize-only --chat --text Hello
      --tokenizer "${LLMCP_MODEL_DIR}/tokenizer.json")
  set_tests_properties(llmcp-chat-template PROPERTIES
    PASS_REGULAR_EXPRESSION
    "248045,846,198,9419,248046,198,248045,74455,198,248068,198"
    TIMEOUT 120)

  add_test(NAME llmcp-compile-real
    COMMAND $<TARGET_FILE:llmcp-compile>
      --model "${LLMCP_MODEL_DIR}"
      --emit=llvm-ir --out "${CMAKE_BINARY_DIR}/qwen35_aot_test.ll")
  set_tests_properties(llmcp-compile-real PROPERTIES TIMEOUT 300)

  add_test(NAME llmcp-compile-real-q8
    COMMAND $<TARGET_FILE:llmcp-compile>
      --model "${LLMCP_MODEL_DIR}" --quant q8_0
      --emit=llvm-ir --out "${CMAKE_BINARY_DIR}/qwen35_aot_q8_test.ll")
  set_tests_properties(llmcp-compile-real-q8 PROPERTIES TIMEOUT 300)
else()
  message(STATUS "llmcp real-model tests disabled: ${LLMCP_MODEL_DIR}/config.json not found")
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

# Full-model golden token tests. These use already-built arenas and AOT
# libraries; they are enabled only when all artifacts exist.
set(LLMCP_GOLDEN_BF16_WEIGHTS "${CMAKE_BINARY_DIR}/weights.bin" CACHE FILEPATH
    "Arena used by the bf16 golden token test")
set(LLMCP_GOLDEN_BF16_AOT "${CMAKE_BINARY_DIR}/libqwen35_aot.dylib" CACHE FILEPATH
    "AOT library used by the bf16 golden token test")
set(LLMCP_GOLDEN_Q8_WEIGHTS "${CMAKE_BINARY_DIR}/weights-q8.bin" CACHE FILEPATH
    "Arena used by the Q8 golden token test")
set(LLMCP_GOLDEN_Q8_AOT "${CMAKE_BINARY_DIR}/libqwen35_aot_q8.dylib" CACHE FILEPATH
    "AOT library used by the Q8 golden token test")
set(LLMCP_GOLDEN_TOKENS "${CMAKE_CURRENT_SOURCE_DIR}/golden/hello_world.tokens")

if(LLMCP_PYTHON AND EXISTS "${LLMCP_MODEL_DIR}/tokenizer.json")
  if(EXISTS "${LLMCP_GOLDEN_BF16_WEIGHTS}"
     AND EXISTS "${LLMCP_GOLDEN_BF16_AOT}")
    add_test(NAME llmcp-golden-tokens-bf16
      COMMAND ${LLMCP_PYTHON}
        "${CMAKE_CURRENT_SOURCE_DIR}/golden/run_golden.py"
        --llmcp-run $<TARGET_FILE:llmcp-run>
        --weights "${LLMCP_GOLDEN_BF16_WEIGHTS}"
        --aot "${LLMCP_GOLDEN_BF16_AOT}"
        --tokenizer "${LLMCP_MODEL_DIR}/tokenizer.json"
        --expected "${LLMCP_GOLDEN_TOKENS}")
    set_tests_properties(llmcp-golden-tokens-bf16 PROPERTIES TIMEOUT 300)
  endif()
  if(EXISTS "${LLMCP_GOLDEN_Q8_WEIGHTS}"
     AND EXISTS "${LLMCP_GOLDEN_Q8_AOT}")
    add_test(NAME llmcp-golden-tokens-q8
      COMMAND ${LLMCP_PYTHON}
        "${CMAKE_CURRENT_SOURCE_DIR}/golden/run_golden.py"
        --llmcp-run $<TARGET_FILE:llmcp-run>
        --weights "${LLMCP_GOLDEN_Q8_WEIGHTS}"
        --aot "${LLMCP_GOLDEN_Q8_AOT}"
        --tokenizer "${LLMCP_MODEL_DIR}/tokenizer.json"
        --expected "${LLMCP_GOLDEN_TOKENS}")
    set_tests_properties(llmcp-golden-tokens-q8 PROPERTIES TIMEOUT 300)
  endif()
else()
  message(STATUS "llmcp golden token tests disabled: missing Python or tokenizer")
endif()

add_test(NAME llmcp-microbench
  COMMAND $<TARGET_FILE:llmcp-microbench>
    --seconds 0.05 --buffer-mb 16 --threads 2)
set_tests_properties(llmcp-microbench PROPERTIES TIMEOUT 120)

add_test(NAME llmcp-matmul-benchmark
  COMMAND $<TARGET_FILE:llmcp-matmul-benchmark>
    --m 64 --n 64 --k 64 --iters 1)
set_tests_properties(llmcp-matmul-benchmark PROPERTIES TIMEOUT 120)

add_test(NAME llmcp-matmul-benchmark-q8
  COMMAND $<TARGET_FILE:llmcp-matmul-benchmark>
    --m 64 --n 64 --k 64 --iters 1 --q8 --threads 4)
set_tests_properties(llmcp-matmul-benchmark-q8 PROPERTIES TIMEOUT 120)

add_subdirectory(unittests)
