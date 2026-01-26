/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "py_translate.h"
#include "common/Logger.h"
#include "common/Timing.h"
#include "cudaq/Optimizer/CodeGen/OpenQASMEmitter.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/platform/default/python/QPU.h"
#include "runtime/cudaq/platform/py_alt_launch_kernel.h"
#include "utils/OpaqueArguments.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

using namespace mlir;

/// @brief Run `cudaq::translate` on the provided kernel.
static std::string translate_impl(const std::string &shortName,
                                  MlirModule module, MlirType returnTy,
                                  const std::string &format,
                                  py::args runtimeArguments) {
  StringRef format_ = format;
  auto formatPair = format_.split(':');
  auto mod = unwrap(module);
  if (!mod->hasAttr(cudaq::runtime::pythonUniqueAttrName))
    throw std::runtime_error("Module is malformed for python. Requires unique "
                             "entry point attribute.");
  auto shortNameAttr = cast<mlir::StringAttr>(
      mod->getAttr(cudaq::runtime::pythonUniqueAttrName));
  std::string shortName_ = shortNameAttr.getValue().str();
  assert(shortName == shortName_ && "kernel names must match");
  std::string longName = cudaq::runtime::cudaqGenPrefixName + shortName;
  auto fn = mod.lookupSymbol<func::FuncOp>(longName);
  if (!fn)
    throw std::runtime_error(
        "Module is malformed for python. Unique entry point cannot be found.");
  auto opaques =
      cudaq::marshal_arguments_for_module_launch(mod, runtimeArguments, fn);

  return StringSwitch<std::function<std::string()>>(formatPair.first)
      .Cases("qir", "qir-full", "qir-adaptive", "qir-base",
             [&]() {
               return cudaq::detail::lower_to_qir_llvm(shortName, mod, opaques,
                                                       format);
             })
      .Case("openqasm2",
            [&]() {
              // For translate to openqasm2, the user is required to (1)
              // synthesize the arguments \e before calling translate and (2)
              // provide no arguments in the translate call itself. Check the
              // latter condition now.
              if (!opaques.empty())
                throw std::runtime_error("Translation to OpenQASM 2.0 requires "
                                         "kernel to have 0 arguments.");
              return cudaq::detail::lower_to_openqasm(shortName, mod, opaques);
            })
      .Default([&]() {
        throw std::runtime_error("Invalid format to translate to: " + format);
        return "Failed to translate to " + format;
      })();
}

/// @brief Bind the translate cudaq function
void cudaq::bindPyTranslate(py::module &mod) {
  mod.def("translate_impl", translate_impl,
          "See python documentation for translate.");
}
