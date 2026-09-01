//===- FabricLowerProtocols.cpp - P2N backend legalization ---------------===//
//
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// All rights reserved.
//
// This source code and the accompanying materials are made available under
// the terms of the Apache License 2.0 which accompanies this distribution.
//
//===----------------------------------------------------------------------===//

#include "qlx/Dialect/Fabric/IR/FabricOps.h"
#include "qlx/Dialect/Fabric/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/SmallVector.h"

namespace qlx {
namespace fabric {
#define GEN_PASS_DEF_FABRICLOWERPROTOCOLS
#include "qlx/Dialect/Fabric/Transforms/Passes.h.inc"
} // namespace fabric
} // namespace qlx

using namespace mlir;
using namespace qlx::fabric;

namespace {

class FabricLowerProtocolsPass
    : public qlx::fabric::impl::FabricLowerProtocolsBase<
          FabricLowerProtocolsPass> {
public:
  using FabricLowerProtocolsBase::FabricLowerProtocolsBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (rootSymbol.empty()) {
      module.emitError("fabric-lower-protocols requires root-symbol");
      return signalPassFailure();
    }

    Operation *root = SymbolTable::lookupSymbolIn(module, rootSymbol);
    if (!root || (!isa<ProtocolOp>(root) && !isa<GadgetOp>(root))) {
      module.emitError("fabric-lower-protocols root @")
          << rootSymbol << " is not a fabric.protocol or fabric.gadget";
      return signalPassFailure();
    }

    SmallVector<ProtocolOp> protocols;
    module.walk([&](ProtocolOp protocol) { protocols.push_back(protocol); });

    for (ProtocolOp protocol : protocols) {
      StringRef name = protocol.getSymName();
      bool isRoot = name == rootSymbol;
      OpBuilder builder(protocol);

      SmallVector<NamedAttribute> attrs;
      if (isRoot) {
        attrs.emplace_back(builder.getStringAttr("entry"),
                           builder.getUnitAttr());
        attrs.emplace_back(
            builder.getStringAttr("device"),
            FlatSymbolRefAttr::get(builder.getContext(), "__qlx_target"));
      }
      attrs.emplace_back(builder.getStringAttr("realization_kind"),
                         builder.getStringAttr("protocol"));
      if (auto generatedBy = protocol.getGeneratedByAttr())
        attrs.emplace_back(builder.getStringAttr("generated_by"), generatedBy);
      if (auto actionSite = protocol.getActionSiteAttr())
        attrs.emplace_back(builder.getStringAttr("action_site"), actionSite);

      // Keep the public symbol stable. The old protocol is renamed only for
      // the short interval in which both symbol operations coexist.
      std::string retired = (name + ".__p2n_source").str();
      protocol.setSymName(retired);
      auto gadget = GadgetOp::create(builder, protocol.getLoc(), name,
                                     protocol.getFunctionType(), attrs);

      Block &block = protocol.getBody().front();
      auto protocolReturn = cast<ProtocolReturnOp>(block.getTerminator());
      builder.setInsertionPoint(protocolReturn);
      ReturnOp::create(builder, protocolReturn.getLoc(),
                       protocolReturn.getOperands());
      protocolReturn.erase();

      gadget.getBody().takeBody(protocol.getBody());
      protocol.erase();
    }

    if (auto gadget = dyn_cast_or_null<GadgetOp>(
            SymbolTable::lookupSymbolIn(module, rootSymbol))) {
      gadget->setAttr("entry", UnitAttr::get(module.getContext()));
      if (!gadget.getDeviceAttr())
        gadget->setAttr("device", FlatSymbolRefAttr::get(module.getContext(),
                                                         "__qlx_target"));
    }

    // A QLX Build intentionally retains its P0/P1 definitions and P2
    // evidence. Native execution engines, however, consume one closed P2
    // representation. Keep only the declarations used by the existing
    // gadget backends and the legalized call graph. This happens on the
    // target's replayed clone, never on the caller's Build.
    SmallVector<Operation *> erase;
    for (Operation &operation : module.getBody()->getOperations()) {
      if (isa<DeviceOp, RegionOp, InterconnectOp, CodeOp, GadgetOp>(&operation))
        continue;
      erase.push_back(&operation);
    }
    for (Operation *operation : erase)
      operation->erase();

    if (failed(module.verify()))
      signalPassFailure();
  }
};

} // namespace
