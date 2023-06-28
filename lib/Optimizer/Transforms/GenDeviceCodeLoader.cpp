/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Builder/Runtime.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "device-code-loader"

using namespace mlir;

namespace {
class GenerateDeviceCodeLoader
    : public cudaq::opt::GenerateDeviceCodeLoaderBase<
          GenerateDeviceCodeLoader> {
public:
  GenerateDeviceCodeLoader() = default;
  GenerateDeviceCodeLoader(bool genAsQuake) { generateAsQuake = genAsQuake; }

  void runOnOperation() override {
    auto module = getOperation();
    auto *ctx = module.getContext();
    assert(ctx && "must have a context");
    auto loc = module->getLoc();
    std::error_code ec;
    llvm::ToolOutputFile out(outputFilename, ec, llvm::sys::fs::OF_None);
    if (ec) {
      llvm::errs() << "Failed to open output file '" << outputFilename << "'\n";
      std::exit(ec.value());
    }
    auto builder = OpBuilder::atBlockEnd(module.getBody());
    if (generateAsQuake) {
      // Add declaration of deviceCodeHolderAdd
      builder.create<LLVM::LLVMFuncOp>(
          loc, "deviceCodeHolderAdd",
          LLVM::LLVMFunctionType::get(
              cudaq::opt::factory::getVoidType(ctx),
              {cudaq::opt::factory::getPointerType(ctx),
               cudaq::opt::factory::getPointerType(ctx)}));
    }

    // Collect all function declarations to forward as part of each Module.
    // These are thrown in so the Module's CallOps are complete. Unused
    // declarations are just thrown away when the code is JIT compiled.
    SmallVector<Operation *> declarations;
    for (auto &op : *module.getBody()) {
      if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
        if (funcOp.empty()) {
          LLVM_DEBUG(llvm::dbgs() << "adding declaration: " << op);
          declarations.push_back(&op);
        }
      } else if (auto funcOp = dyn_cast<LLVM::LLVMFuncOp>(op)) {
        if (funcOp.empty()) {
          LLVM_DEBUG(llvm::dbgs() << "adding declaration: " << op);
          declarations.push_back(&op);
        }
      }
    }

    for (auto &op : *module.getBody()) {
      // FIXME: May not be a FuncOp in the future.
      if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
        if (!funcOp.getName().startswith(cudaq::runtime::cudaqGenPrefixName))
          continue;
        auto className =
            funcOp.getName().drop_front(cudaq::runtime::cudaqGenPrefixLength);
        LLVM_DEBUG(llvm::dbgs() << "processing function" << className << '\n');
        // Generate LLVM-IR dialect to register the device code loading.
        std::string thunkName = className.str() + ".thunk";
        std::string funcCode;
        llvm::raw_string_ostream strOut(funcCode);
        OpPrintingFlags opf;
        opf.enableDebugInfo(/*enable=*/true,
                            /*pretty=*/false);
        strOut << "module { ";
        funcOp.print(strOut, opf);
        strOut << '\n';

        // We'll also need any non-inlined functions that are
        // called by our cudaq kernel
        funcOp.walk([&](func::CallOp callOp) {
          if (auto neededFunc =
                  module.lookupSymbol<func::FuncOp>(callOp.getCallee())) {
            if (neededFunc.empty())
              return WalkResult::skip();
            neededFunc.print(strOut, opf);
            strOut << '\n';
            return WalkResult::advance();
          }
          return WalkResult::advance();
        });

        // Include the generated kernel thunk if present since it is on the
        // callee side of the launchKernel() callback.
        if (auto *thunkFunc = module.lookupSymbol(thunkName)) {
          LLVM_DEBUG(llvm::dbgs() << "found thunk function\n");
          strOut << *thunkFunc << '\n';
        }
        if (auto *zeroDynRes =
                module.lookupSymbol("__nvqpp_zeroDynamicResult")) {
          LLVM_DEBUG(llvm::dbgs() << "found zero dyn result function\n");
          strOut << *zeroDynRes << '\n';
        }
        if (auto *createDynRes =
                module.lookupSymbol("__nvqpp_createDynamicResult")) {
          LLVM_DEBUG(llvm::dbgs() << "found create dyn result function\n");
          strOut << *createDynRes << '\n';
        }
        for (auto *op : declarations)
          strOut << *op << '\n';
        strOut << "\n}\n" << '\0';
        auto devCode = builder.create<LLVM::GlobalOp>(
            loc, cudaq::opt::factory::getStringType(ctx, funcCode.size()),
            /*isConstant=*/true, LLVM::Linkage::Private,
            className.str() + "CodeHolder.extract_device_code",
            builder.getStringAttr(funcCode), /*alignment=*/0);
        auto devName = builder.create<LLVM::GlobalOp>(
            loc, cudaq::opt::factory::getStringType(ctx, className.size() + 1),
            /*isConstant=*/true, LLVM::Linkage::Private,
            className.str() + "CodeHolder.extract_device_name",
            builder.getStringAttr(className.str() + '\0'), /*alignment=*/0);
        auto initFun = builder.create<LLVM::LLVMFuncOp>(
            loc, className.str() + ".init_func",
            LLVM::LLVMFunctionType::get(cudaq::opt::factory::getVoidType(ctx),
                                        {}),
            LLVM::Linkage::External);
        auto insPt = builder.saveInsertionPoint();
        auto *initFunEntry = initFun.addEntryBlock();
        builder.setInsertionPointToStart(initFunEntry);
        auto devRef = builder.create<LLVM::AddressOfOp>(
            loc, cudaq::opt::factory::getPointerType(devName.getType()),
            devName.getSymName());
        auto codeRef = builder.create<LLVM::AddressOfOp>(
            loc, cudaq::opt::factory::getPointerType(devCode.getType()),
            devCode.getSymName());
        auto castDevRef = builder.create<LLVM::BitcastOp>(
            loc, cudaq::opt::factory::getPointerType(ctx), devRef);
        auto castCodeRef = builder.create<LLVM::BitcastOp>(
            loc, cudaq::opt::factory::getPointerType(ctx), codeRef);
        builder.create<LLVM::CallOp>(loc, std::nullopt, "deviceCodeHolderAdd",
                                     ValueRange{castDevRef, castCodeRef});
        builder.create<LLVM::ReturnOp>(loc, ValueRange{});
        builder.restoreInsertionPoint(insPt);
        cudaq::opt::factory::createGlobalCtorCall(
            module, mlir::FlatSymbolRefAttr::get(ctx, initFun.getName()));
      }
    }
    out.keep();
  }
};
} // namespace

std::unique_ptr<Pass>
cudaq::opt::createGenerateDeviceCodeLoader(bool genAsQuake) {
  return std::make_unique<GenerateDeviceCodeLoader>(genAsQuake);
}
