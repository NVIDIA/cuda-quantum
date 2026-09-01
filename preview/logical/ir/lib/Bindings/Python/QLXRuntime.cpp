//===- QLXRuntime.cpp - QLX text/translation Python helpers ------------===//
//
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// All rights reserved.
//
// This source code and the accompanying materials are made available under
// the terms of the Apache License 2.0 which accompanies this distribution.
//
//===----------------------------------------------------------------------===//
//
// Textual / module-level helpers exposed to Python:
//
//   - run_pass(module, pipe)  : run a named pass pipeline on an MlirModule
//   - clone_module(module)    : deep-clone an MlirModule
//   - translate(module, name) : named MLIR translation (CAPI)
//
//===----------------------------------------------------------------------===//

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"

#include "qlx-c/Dialect/Fabric.h"
#include "qlx-c/Dialect/QLX.h"
#include "qlx-c/Passes.h"
#include "qlx-c/Target/Translations.h"

// Upstream MLIR transform passes (provides --symbol-dce).
#include "mlir-c/Transforms.h"

// Module-text helpers that don't have a CAPI entry yet still talk to the
// C++ surface directly.  The dialect bindings (DialectQLX/DialectFabric)
// stay CAPI-only; this file is the "sin bin" for the textual surface.
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/Plugins/DialectPlugin.h"
#include "mlir/Tools/Plugins/PassPlugin.h"

#include "qlx/Dialect/QLX/IR/QLXAttrs.h"
#include "qlx/Dialect/QLX/IR/QLXDialect.h"
#include "qlx/Dialect/QLX/IR/QLXTypes.h"
#include "qlx/Dialect/QLX/Transforms/QLXSynthesize.h"
#include "qlx/Dialect/QLX/Transforms/QLXToPBC.h"
#include "qlx/Dialect/QLX/Transforms/QLXVerifyPBC.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "qlx/Dialect/Fabric/IR/FabricDialect.h"

#include "PassDebug.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/raw_ostream.h"

#if defined(__linux__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

namespace nb = nanobind;

// Process-global list of plugin .so paths registered via load_plugin().
// Paths are stored here so that dialect registration can be applied to new
// MLIRContexts if needed (pass registration is immediate via PassPlugin).
static std::vector<std::string> &getPluginPaths() {
  static auto *paths = new std::vector<std::string>();
  return *paths;
}

namespace {

// Trampoline used by qlxTranslate*: append to a std::string captured
// through the userData pointer.
void appendToString(MlirStringRef ref, void *userData) {
  auto *out = reinterpret_cast<std::string *>(userData);
  out->append(ref.data, ref.length);
}

inline MlirStringRef toRef(const std::string &s) {
  return MlirStringRef{s.data(), s.size()};
}

void runPassPipeline(MlirModule pyModule, const std::string &pipeline) {
  mlir::ModuleOp mod = unwrap(pyModule);
  mlir::MLIRContext *ctx = mod.getContext();
  mlir::PassManager pm(ctx);
  qlx::python::configureIRPrinting(pm);
  if (mlir::failed(mlir::parsePassPipeline(pipeline, pm)))
    throw std::runtime_error("Invalid pass pipeline: " + pipeline);

  std::string diagnostics;
  mlir::ScopedDiagnosticHandler handler(ctx, [&](mlir::Diagnostic &diag) {
    if (diag.getSeverity() != mlir::DiagnosticSeverity::Error)
      return mlir::failure();
    std::string entry;
    llvm::raw_string_ostream os(entry);
    os << diag.getLocation() << ": " << diag.str();
    for (mlir::Diagnostic &note : diag.getNotes())
      os << "\n  note: " << note.getLocation() << ": " << note.str();
    os.flush();
    if (!diagnostics.empty())
      diagnostics += "\n";
    diagnostics += entry;
    return mlir::success();
  });
  if (mlir::failed(pm.run(mod))) {
    std::string message = "Pass pipeline failed: " + pipeline;
    if (!diagnostics.empty())
      message += "\n" + diagnostics;
    throw std::runtime_error(message);
  }
}

MlirModule moduleFromPythonCapsule(nb::handle pyModule) {
  if (!nb::hasattr(pyModule, "_CAPIPtr"))
    throw nb::type_error("module must expose the MLIR _CAPIPtr protocol");
  nb::object capsule = pyModule.attr("_CAPIPtr");
  if (!PyCapsule_CheckExact(capsule.ptr()))
    throw nb::type_error("module._CAPIPtr must be a Python capsule");
  const char *name = PyCapsule_GetName(capsule.ptr());
  if (!name || !llvm::StringRef(name).ends_with("ir.Module._CAPIPtr"))
    throw nb::type_error("module._CAPIPtr is not an MLIR module capsule");
  void *ptr = PyCapsule_GetPointer(capsule.ptr(), name);
  if (!ptr)
    throw nb::python_error();
  return MlirModule{ptr};
}

} // namespace

NB_MODULE(_qlxRuntime, m) {
  m.doc() = "QLX text / module translation helpers";

  mlirRegisterTransformsPasses();
  qlxRegisterAllPasses();
  m.attr("has_quake_import") = true;

  //===-----------------------------------------------------------------===//
  // Plugin loading
  //===-----------------------------------------------------------------===//

  m.def(
      "load_plugin",
      [](const std::string &path) {
  // Promote the QLX Python CAPI shared library to RTLD_GLOBAL so that
  // plugins built with --unresolved-symbols=ignore-all can find MLIR
  // symbols (pass registry, dialect registry, etc.) at dlopen time.
  // We locate the library via dladdr on a known CAPI symbol.
#if defined(__linux__) || defined(__APPLE__)
        {
          ::Dl_info capi_info;
          // qlxRegisterAllPasses is defined in libQLXPythonCAPI.so.
          if (::dladdr((void *)qlxRegisterAllPasses, &capi_info) &&
              capi_info.dli_fname) {
            ::dlopen(capi_info.dli_fname,
                     RTLD_LAZY | RTLD_GLOBAL | RTLD_NOLOAD);
          }
        }
#endif

        // Register the plugin's passes immediately into the global pass
        // registry so that parsePassPipeline can resolve them in any
        // PassManager from this point on.
        auto passPlugin = mlir::PassPlugin::load(path);
        if (!passPlugin) {
          throw std::runtime_error("Failed to load pass plugin: " + path +
                                   " -- " +
                                   llvm::toString(passPlugin.takeError()));
        }
        passPlugin->registerPassRegistryCallbacks();

        // Store the path for dialect injection into new contexts.
        getPluginPaths().push_back(path);

        // Also open the library permanently so JIT'd code that references
        // plugin runtime symbols can resolve them via the process namespace.
        std::string errMsg;
        if (llvm::sys::DynamicLibrary::LoadLibraryPermanently(path.c_str(),
                                                              &errMsg)) {
          // Non-fatal: the plugin may be pass-only with no runtime symbols.
          // Emit a warning but do not throw.
          llvm::errs() << "[qlx] load_plugin: LoadLibraryPermanently(" << path
                       << "): " << errMsg << "\n";
        }
      },
      nb::arg("path"),
      "Load an MLIR dialect+pass plugin .so into the QLX runtime.\n\n"
      "Registers the plugin's passes in the global MLIR pass registry "
      "immediately, and stores the path so its dialect can be loaded into "
      "subsequently-created MLIRContexts.");

  m.def(
      "get_plugin_paths",
      []() -> std::vector<std::string> { return getPluginPaths(); },
      "Return the list of plugin .so paths registered via load_plugin().");

  m.def(
      "apply_dialect_plugin",
      [](const std::string &path, MlirDialectRegistry registry) {
        // Register a single dialect plugin's dialect callbacks into the
        // supplied registry. Called from Python's load_plugin() wrapper to
        // inject the plugin's dialects into the global dialect registry that
        // MLIR's Context.__init__ snapshots at import time.
        auto plugin = mlir::DialectPlugin::load(path);
        if (!plugin) {
          throw std::runtime_error("Failed to load dialect plugin: " + path +
                                   " -- " + llvm::toString(plugin.takeError()));
        }
        plugin->registerDialectRegistryCallbacks(*unwrap(registry));
      },
      nb::arg("path"), nb::arg("registry"),
      "Register a dialect plugin's callbacks into the given "
      "DialectRegistry.\n\n"
      "Call this after load_plugin() with the global MLIR dialect registry so "
      "that subsequently-created Contexts include the plugin's dialects.");

  //===-----------------------------------------------------------------===//
  // Text-form helpers (string in, string out).
  //===-----------------------------------------------------------------===//

  m.def(
      "synthesize_qlx",
      [](const std::string &mlirText, double precision) -> std::string {
        mlir::MLIRContext context;
        context.getOrLoadDialect<qlx::QLXDialect>();
        context.getOrLoadDialect<mlir::arith::ArithDialect>();

        auto module =
            mlir::parseSourceString<mlir::ModuleOp>(mlirText, &context);
        if (!module)
          throw std::runtime_error("Failed to parse MLIR for synthesis");

        if (mlir::failed(qlx::synthesizeRotations(*module, precision)))
          throw std::runtime_error("QLX rotation synthesis failed");

        std::string result;
        llvm::raw_string_ostream os(result);
        module->print(os);
        return result;
      },
      nb::arg("mlir_text"), nb::arg("precision") = 1e-10,
      "Native device-free legalization of a P0 module to positive H/S/T/CX. "
      "Returns the transformed MLIR text.");

  m.def(
      "verify_clifford_t",
      [](const std::string &mlirText) -> bool {
        mlir::MLIRContext context;
        context.getOrLoadDialect<qlx::QLXDialect>();
        context.getOrLoadDialect<mlir::arith::ArithDialect>();

        auto module =
            mlir::parseSourceString<mlir::ModuleOp>(mlirText, &context);
        if (!module)
          throw std::runtime_error(
              "Failed to parse MLIR for Clifford+T verification");
        return mlir::succeeded(qlx::verifyCliffordT(*module));
      },
      nb::arg("mlir_text"),
      "Return true when every logical apply is legal in positive H/S/T/CX.");

  m.def(
      "to_pbc",
      [](const std::string &mlirText) -> std::string {
        mlir::MLIRContext context;
        context.getOrLoadDialect<qlx::QLXDialect>();
        context.getOrLoadDialect<mlir::arith::ArithDialect>();

        auto module =
            mlir::parseSourceString<mlir::ModuleOp>(mlirText, &context);
        if (!module)
          throw std::runtime_error("Failed to parse MLIR for PBC lowering");

        if (mlir::failed(qlx::lowerToPBC(*module)))
          throw std::runtime_error("QLX PBC lowering failed");

        std::string result;
        llvm::raw_string_ostream os(result);
        module->print(os);
        return result;
      },
      nb::arg("mlir_text"),
      "Lower a synthesized Clifford+T program to Pauli-based-computation form: "
      "pi/4 Pauli-product rotations + Pauli-product measurements (device-free "
      "P0). Returns the transformed MLIR text.");

  m.def(
      "verify_pbc",
      [](const std::string &mlirText) -> bool {
        mlir::MLIRContext context;
        context.getOrLoadDialect<qlx::QLXDialect>();
        context.getOrLoadDialect<mlir::arith::ArithDialect>();

        auto module =
            mlir::parseSourceString<mlir::ModuleOp>(mlirText, &context);
        if (!module)
          throw std::runtime_error("Failed to parse MLIR for PBC verification");

        std::string error;
        if (mlir::failed(qlx::verifyPBCForm(*module, error)))
          throw std::runtime_error(error);
        return true;
      },
      nb::arg("mlir_text"),
      "Certify a module is in Pauli-based-computation normal form (only "
      "prepare/pi-4 pauli_rotation/mpp/discard/return; rotations before "
      "measurements; measured Paulis pairwise commute). Returns true, or "
      "raises "
      "with the first violation.");

  //===-----------------------------------------------------------------===//
  // MlirModule-based bindings (no string round-trip)
  //===-----------------------------------------------------------------===//

  m.def(
      "clone_module",
      [](MlirModule pyModule) -> MlirModule {
        MlirOperation clonedOp =
            mlirOperationClone(mlirModuleGetOperation(pyModule));
        MlirModule clonedModule = mlirModuleFromOperation(clonedOp);
        if (mlirModuleIsNull(clonedModule)) {
          mlirOperationDestroy(clonedOp);
          throw std::runtime_error("Cloned operation is not a module");
        }
        return clonedModule;
      },
      nb::arg("module"),
      "Return a deep clone of an MLIR module in the same MLIRContext.");

  m.def(
      "verify_clifford_t_module",
      [](MlirModule pyModule) -> bool {
        return mlirLogicalResultIsSuccess(qlxVerifyCliffordTModule(pyModule));
      },
      nb::arg("module"),
      "Return true when a typed module is legal positive H/S/T/CX.");

  m.def(
      "lower_to_pbc_module",
      [](MlirModule pyModule) {
        if (!qlxLowerToPBC(pyModule))
          throw std::runtime_error("QLX PBC lowering failed");
      },
      nb::arg("module"),
      "Lower a typed Clifford+T module to PBC form in place.");

  m.def(
      "verify_pbc_module",
      [](MlirModule pyModule) -> bool {
        return mlirLogicalResultIsSuccess(qlxVerifyPBCModule(pyModule));
      },
      nb::arg("module"), "Certify that a typed module is in PBC normal form.");

  m.def(
      "clone_module_capsule",
      [](nb::object pyModule) -> nb::capsule {
        MlirModule source = moduleFromPythonCapsule(pyModule);
        MlirOperation clonedOp =
            mlirOperationClone(mlirModuleGetOperation(source));
        MlirModule clonedModule = mlirModuleFromOperation(clonedOp);
        if (mlirModuleIsNull(clonedModule)) {
          mlirOperationDestroy(clonedOp);
          throw std::runtime_error("Cloned operation is not a module");
        }
        return nb::capsule(clonedModule.ptr, MLIR_PYTHON_CAPSULE_MODULE);
      },
      nb::arg("module"),
      "Clone any shared-C-API MLIR module into a qlx.ir module capsule.");

  m.def(
      "replace_module_contents_capsule",
      [](nb::object destination, nb::object source) {
        mlir::ModuleOp destinationModule =
            unwrap(moduleFromPythonCapsule(destination));
        mlir::ModuleOp sourceModule = unwrap(moduleFromPythonCapsule(source));
        if (destinationModule.getContext() != sourceModule.getContext())
          throw nb::value_error(
              "source and destination modules must share one MLIRContext");
        destinationModule->setAttrs(sourceModule->getAttrDictionary());
        destinationModule.getBodyRegion().takeBody(
            sourceModule.getBodyRegion());
      },
      nb::arg("destination"), nb::arg("source"),
      "Atomically replace a live module's attributes and body from a verified "
      "same-context module.");

  m.def(
      "run_pass",
      [](MlirModule pyModule, const std::string &pipeline) {
        runPassPipeline(pyModule, pipeline);
      },
      nb::arg("module"), nb::arg("pipeline"),
      "Run a named MLIR pass pipeline on a module (no re-parse).");

  m.def(
      "run_pass_capsule",
      [](nb::object pyModule, const std::string &pipeline) {
        runPassPipeline(moduleFromPythonCapsule(pyModule), pipeline);
      },
      nb::arg("module"), nb::arg("pipeline"),
      "Run a pass on any live MLIR Python module sharing this C API runtime.");

  m.def(
      "translate",
      [](MlirModule pyModule, const std::string &name) -> nb::object {
        std::string out;
        if (name == "fabric-to-stim") {
          if (mlirLogicalResultIsFailure(
                  qlxTranslateFabricToStim(pyModule, appendToString, &out)))
            throw std::runtime_error("Fabric Stim emission failed");
          return nb::cast(out);
        }
        throw std::runtime_error("Unknown translation: " + name);
      },
      nb::arg("module"), nb::arg("name"),
      "Run a named fail-closed MLIR translation on a verified module.");
}
