//===- llmcp-opt.cpp - MLIR optimizer driver for LLMCP --------------------===//
//
// MLIR optimizer driver with LLMCP passes and dialects.
//
//===----------------------------------------------------------------------===//

#include "llmcp/Dialect/LLMCP/IR/LLMCPDialect.h"
#include "llmcp/Dialect/LLMCP/IR/LLMCPOps.h"
#include "llmcp/Dialect/LLMCP/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::llmcp::registerLowerToLinalgPass();
  mlir::llmcp::registerParallelizePass();

  mlir::DialectRegistry registry;
  registry.insert<mlir::llmcp::LLMCPDialect, mlir::arith::ArithDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::LLVM::LLVMDialect, mlir::math::MathDialect,
                  mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                  mlir::tensor::TensorDialect, mlir::vector::VectorDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "llmcp optimizer\n", registry));
}
