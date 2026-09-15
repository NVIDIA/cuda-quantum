/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/
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
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Bytecode/BytecodeWriter.h"
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
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/raw_ostream.h"
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

#include "PassDebug.h"

#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <mutex>
#include <unordered_map>

namespace nb = nanobind;

// Process-global list of plugin .so paths registered via load_plugin().
// Paths are stored here so that dialect registration can be applied to new
// MLIRContexts if needed (pass registration is immediate via PassPlugin).
static std::vector<std::string> &getPluginPaths() {
  static auto *paths = new std::vector<std::string>();
  return *paths;
}
namespace {

struct ContextDiagnosticLockRegistry {
  std::mutex mutex;
  std::unordered_map<mlir::MLIRContext *, std::weak_ptr<std::mutex>> locks;
};

std::shared_ptr<std::mutex> contextDiagnosticMutex(mlir::MLIRContext *context) {
  // The registry stores weak references so transient Python contexts do not
  // retain one mutex for the lifetime of the process.  Reuse of an address
  // after context destruction is harmless: an expired entry mints a new lock.
  static auto *registry = new ContextDiagnosticLockRegistry();
  std::lock_guard<std::mutex> registryLock(registry->mutex);
  std::weak_ptr<std::mutex> &slot = registry->locks[context];
  std::shared_ptr<std::mutex> result = slot.lock();
  if (!result) {
    result = std::make_shared<std::mutex>();
    slot = result;
  }
  return result;
}

// Trampoline used by qlxTranslate*: append to a std::string captured
// through the userData pointer.
void appendToString(MlirStringRef ref, void *userData) {
  auto *out = reinterpret_cast<std::string *>(userData);
  out->append(ref.data, ref.length);
}

inline MlirStringRef toRef(const std::string &s) {
  return MlirStringRef{s.data(), s.size()};
}

void runPassPipeline(MlirModule pyModule, const std::string &pipeline,
                     bool verify) {
  mlir::ModuleOp mod = unwrap(pyModule);
  mlir::MLIRContext *ctx = mod.getContext();
  std::shared_ptr<std::mutex> diagnosticMutex = contextDiagnosticMutex(ctx);
  std::unique_lock<std::mutex> diagnosticLock(*diagnosticMutex);
  mlir::PassManager pm(ctx);
  pm.enableVerifier(verify);
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

void materializeVerifiedAnalyticalLowerTier(
    MlirModule pyModule, const std::string &root, const std::string &device,
    const std::string &staticResult, const std::string &analyticalResult,
    double physicalError, double failureBudget, double cycleTime,
    double scalingPrefactor, double scalingThreshold,
    bool requireEstablishedDistance) {
  mlir::ModuleOp module = unwrap(pyModule);
  mlir::MLIRContext *context = module.getContext();
  std::shared_ptr<std::mutex> diagnosticMutex = contextDiagnosticMutex(context);
  std::unique_lock<std::mutex> diagnosticLock(*diagnosticMutex);
  std::string diagnostics;
  mlir::ScopedDiagnosticHandler handler(
      context, [&](mlir::Diagnostic &diagnostic) {
        if (diagnostic.getSeverity() != mlir::DiagnosticSeverity::Error)
          return mlir::failure();
        std::string entry;
        llvm::raw_string_ostream stream(entry);
        stream << diagnostic.getLocation() << ": " << diagnostic.str();
        for (mlir::Diagnostic &note : diagnostic.getNotes())
          stream << "\n  note: " << note.getLocation() << ": " << note.str();
        if (!diagnostics.empty())
          diagnostics += "\n";
        diagnostics += entry;
        return mlir::success();
      });

  if (mlirLogicalResultIsFailure(qlxMaterializeVerifiedAnalyticalLowerTier(
          pyModule, toRef(root), toRef(device), toRef(staticResult),
          toRef(analyticalResult), physicalError, failureBudget, cycleTime,
          scalingPrefactor, scalingThreshold, requireEstablishedDistance))) {
    std::string message = "Native analytical lower-tier materialization failed";
    if (!diagnostics.empty())
      message += "\n" + diagnostics;
    throw std::runtime_error(message);
  }
}

std::string estimateScheduleImpl(MlirModule pyModule,
                                 const std::string &schedule,
                                 const std::string &lowerTier,
                                 bool fullWorkload, bool verifiedInput) {
  mlir::ModuleOp module = unwrap(pyModule);
  std::shared_ptr<std::mutex> diagnosticMutex =
      contextDiagnosticMutex(module.getContext());
  std::unique_lock<std::mutex> diagnosticLock(*diagnosticMutex);
  std::string diagnostics;
  mlir::ScopedDiagnosticHandler handler(
      module.getContext(), [&](mlir::Diagnostic &diagnostic) {
        if (diagnostic.getSeverity() != mlir::DiagnosticSeverity::Error)
          return mlir::failure();
        std::string entry;
        llvm::raw_string_ostream stream(entry);
        stream << diagnostic.getLocation() << ": " << diagnostic.str();
        for (mlir::Diagnostic &note : diagnostic.getNotes())
          stream << "\n  note: " << note.getLocation() << ": " << note.str();
        if (!diagnostics.empty())
          diagnostics += "\n";
        diagnostics += entry;
        return mlir::success();
      });
  std::string result;
  MlirLogicalResult status =
      verifiedInput ? qlxEstimateVerifiedScheduleJSONWithTermination(
                          pyModule, toRef(schedule), toRef(lowerTier),
                          fullWorkload, appendToString, &result)
                    : qlxEstimateScheduleJSONWithTermination(
                          pyModule, toRef(schedule), toRef(lowerTier),
                          fullWorkload, appendToString, &result);
  if (mlirLogicalResultIsFailure(status)) {
    std::string message = "Native Tier-3 schedule estimation failed";
    if (!diagnostics.empty())
      message += "\n" + diagnostics;
    throw std::runtime_error(message);
  }
  return result;
}

std::string estimateSchedule(MlirModule pyModule, const std::string &schedule,
                             const std::string &lowerTier, bool fullWorkload) {
  return estimateScheduleImpl(pyModule, schedule, lowerTier, fullWorkload,
                              /*verifiedInput=*/false);
}

std::string estimateVerifiedSchedule(MlirModule pyModule,
                                     const std::string &schedule,
                                     const std::string &lowerTier,
                                     bool fullWorkload) {
  return estimateScheduleImpl(pyModule, schedule, lowerTier, fullWorkload,
                              /*verifiedInput=*/true);
}

std::string scheduleAndEstimateImpl(MlirModule pyModule,
                                    const std::string &graph,
                                    const std::string &schedule,
                                    const std::string &lowerTier,
                                    bool fullWorkload, bool verifiedInput) {
  mlir::ModuleOp module = unwrap(pyModule);
  std::shared_ptr<std::mutex> diagnosticMutex =
      contextDiagnosticMutex(module.getContext());
  std::unique_lock<std::mutex> diagnosticLock(*diagnosticMutex);
  std::string diagnostics;
  mlir::ScopedDiagnosticHandler handler(
      module.getContext(), [&](mlir::Diagnostic &diagnostic) {
        if (diagnostic.getSeverity() != mlir::DiagnosticSeverity::Error)
          return mlir::failure();
        std::string entry;
        llvm::raw_string_ostream stream(entry);
        stream << diagnostic.getLocation() << ": " << diagnostic.str();
        for (mlir::Diagnostic &note : diagnostic.getNotes())
          stream << "\n  note: " << note.getLocation() << ": " << note.str();
        if (!diagnostics.empty())
          diagnostics += "\n";
        diagnostics += entry;
        return mlir::success();
      });
  std::string result;
  MlirLogicalResult status =
      verifiedInput
          ? qlxScheduleVerifiedAndEstimateJSONWithTermination(
                pyModule, toRef(graph), toRef(schedule), toRef(lowerTier),
                fullWorkload, appendToString, &result)
          : qlxScheduleAndEstimateJSONWithTermination(
                pyModule, toRef(graph), toRef(schedule), toRef(lowerTier),
                fullWorkload, appendToString, &result);
  if (mlirLogicalResultIsFailure(status)) {
    std::string message = "Native P3 scheduling and estimation failed";
    if (!diagnostics.empty())
      message += "\n" + diagnostics;
    throw std::runtime_error(message);
  }
  return result;
}

std::string scheduleAndEstimate(MlirModule pyModule, const std::string &graph,
                                const std::string &schedule,
                                const std::string &lowerTier,
                                bool fullWorkload) {
  return scheduleAndEstimateImpl(pyModule, graph, schedule, lowerTier,
                                 fullWorkload,
                                 /*verifiedInput=*/false);
}

std::string scheduleVerifiedAndEstimate(MlirModule pyModule,
                                        const std::string &graph,
                                        const std::string &schedule,
                                        const std::string &lowerTier,
                                        bool fullWorkload) {
  return scheduleAndEstimateImpl(pyModule, graph, schedule, lowerTier,
                                 fullWorkload,
                                 /*verifiedInput=*/true);
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

MlirContext contextFromPythonCapsule(nb::handle pyContext) {
  if (!nb::hasattr(pyContext, "_CAPIPtr"))
    throw nb::type_error("context must expose the MLIR _CAPIPtr protocol");
  nb::object capsule = pyContext.attr("_CAPIPtr");
  if (!PyCapsule_CheckExact(capsule.ptr()))
    throw nb::type_error("context._CAPIPtr must be a Python capsule");
  const char *name = PyCapsule_GetName(capsule.ptr());
  if (!name || !llvm::StringRef(name).ends_with("ir.Context._CAPIPtr"))
    throw nb::type_error("context._CAPIPtr is not an MLIR context capsule");
  void *ptr = PyCapsule_GetPointer(capsule.ptr(), name);
  if (!ptr)
    throw nb::python_error();
  return MlirContext{ptr};
}

} // namespace

NB_MODULE(_qlxRuntime, m) {
  m.doc() = "QLX text / module translation helpers";

  mlirRegisterTransformsPasses();
  qlxRegisterAllPasses();
#ifdef QLX_HAS_CUDAQ_QUAKE
  m.attr("has_quake_import") = true;
#else
  m.attr("has_quake_import") = false;
#endif

  //===-----------------------------------------------------------------===//
  // Plugin loading
  //===-----------------------------------------------------------------===//

  m.def(
      "load_plugin",
      [](const std::string &path) {
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
      "absorb_clifford_frame_module",
      [](MlirModule pyModule) {
        if (!mlirLogicalResultIsSuccess(qlxAbsorbCliffordFrameModule(pyModule)))
          throw std::runtime_error("QLX Clifford-frame lowering failed");
      },
      nb::arg("module"),
      "Absorb exact Clifford actions in a typed P0 module while preserving "
      "non-Clifford Pauli rotations for downstream selection.");

  m.def(
      "verify_clifford_frame_module",
      [](MlirModule pyModule) -> bool {
        return mlirLogicalResultIsSuccess(
            qlxVerifyCliffordFrameModule(pyModule));
      },
      nb::arg("module"),
      "Certify that a typed P0 module is in hybrid Clifford-frame form.");

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
      "clone_module_into_context_capsule",
      [](nb::object pyModule, nb::object pyDestination) -> nb::capsule {
        mlir::ModuleOp source = unwrap(moduleFromPythonCapsule(pyModule));
        MlirContext destination = contextFromPythonCapsule(pyDestination);
        std::string bytecode;
        llvm::raw_string_ostream output(bytecode);
        if (failed(mlir::writeBytecodeToFile(source, output)))
          throw std::runtime_error("Failed to serialize MLIR module bytecode");
        output.flush();

        mlir::Block parsed;
        mlir::ParserConfig config(unwrap(destination));
        if (failed(mlir::readBytecodeFile(
                llvm::MemoryBufferRef(bytecode, "qlx-context-transfer"),
                &parsed, config)) ||
            !llvm::hasSingleElement(parsed))
          throw std::runtime_error(
              "Failed to transfer MLIR module into the destination context");
        auto module = dyn_cast<mlir::ModuleOp>(parsed.front());
        if (!module)
          throw std::runtime_error(
              "Transferred bytecode did not contain exactly one module");
        module->remove();
        MlirModule result = wrap(module);
        return nb::capsule(result.ptr, MLIR_PYTHON_CAPSULE_MODULE);
      },
      nb::arg("module"), nb::arg("destination"),
      "Clone a shared-C-API MLIR module by typed bytecode into a supplied "
      "QLX MLIRContext.");

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
      [](MlirModule pyModule, const std::string &pipeline, bool verify) {
        runPassPipeline(pyModule, pipeline, verify);
      },
      nb::arg("module"), nb::arg("pipeline"), nb::arg("verify") = true,
      "Run a named MLIR pass pipeline on a module (no re-parse).");

  m.def(
      "run_pass_capsule",
      [](nb::object pyModule, const std::string &pipeline, bool verify) {
        runPassPipeline(moduleFromPythonCapsule(pyModule), pipeline, verify);
      },
      nb::arg("module"), nb::arg("pipeline"), nb::arg("verify") = true,
      "Run a pass on any live MLIR Python module sharing this C API runtime.");

  m.def("_materialize_verified_analytical_lower_tier",
        materializeVerifiedAnalyticalLowerTier, nb::arg("module"),
        nb::arg("root"), nb::arg("device"), nb::arg("static_result"),
        nb::arg("analytical_result"), nb::arg("physical_error"),
        nb::arg("failure_budget"), nb::arg("cycle_time"),
        nb::arg("scaling_prefactor"), nb::arg("scaling_threshold"),
        nb::arg("require_established_distance"),
        nb::call_guard<nb::gil_scoped_release>(),
        "Internal count-once Tier-2 closure for a compiler-authenticated "
        "in-process module.");

  m.def("estimate_schedule_json", estimateSchedule, nb::arg("module"),
        nb::arg("schedule"), nb::arg("lower_tier") = "",
        nb::arg("full_workload") = false,
        nb::call_guard<nb::gil_scoped_release>(),
        "Derive a Tier-3 schedule estimate without cloning or mutating the "
        "retained MLIR module.");

  m.def("_estimate_verified_schedule_json", estimateVerifiedSchedule,
        nb::arg("module"), nb::arg("schedule"), nb::arg("lower_tier") = "",
        nb::arg("full_workload") = false,
        nb::call_guard<nb::gil_scoped_release>(),
        "Internal estimate path for a compiler-authenticated in-process "
        "schedule ModuleOp.");

  m.def("schedule_and_estimate_json", scheduleAndEstimate, nb::arg("module"),
        nb::arg("graph"), nb::arg("schedule"), nb::arg("lower_tier") = "",
        nb::arg("full_workload") = false,
        nb::call_guard<nb::gil_scoped_release>(),
        "Schedule a live P3 graph and estimate its typed rows before "
        "serializing/parsing the retained schedule representation.");

  m.def("_schedule_verified_and_estimate_json", scheduleVerifiedAndEstimate,
        nb::arg("module"), nb::arg("graph"), nb::arg("schedule"),
        nb::arg("lower_tier") = "", nb::arg("full_workload") = false,
        nb::call_guard<nb::gil_scoped_release>(),
        "Internal fused path for a compiler-authenticated in-process P3 "
        "ModuleOp.");

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
      "Run a named MLIR translation on a module (no re-parse).");
}
