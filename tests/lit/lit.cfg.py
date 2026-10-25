import os

import lit.formats

config.name = "llmcp"
config.test_format = lit.formats.ShTest(execute_external=False)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)

config.substitutions.append(("%llmcp-opt", config.llmcp_opt))
config.substitutions.append(("%FileCheck", config.filecheck))
