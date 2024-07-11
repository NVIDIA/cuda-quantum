
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/ADT/GraphCSR.h"
#include "cudaq/Support/Graph.h"
#include "cudaq/Support/Device.h"
#include "cudaq/Support/Placement.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ScopedPrinter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/TopologicalSortUtils.h"

#define DEBUG_TYPE "quantum-scoper"

using namespace mlir;

// Use specific cudaq elements without bringing in the full namespace
using cudaq::Device;
using cudaq::Placement;
using cudaq::QuantumMeasure;

//===----------------------------------------------------------------------===//
// Generated logic
//===----------------------------------------------------------------------===//

namespace cudaq::opt {
#define GEN_PASS_DEF_ASYNCSCOPEPASS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
}


namespace {
  
struct AsyncScopePass : public cudaq::opt::impl::AsyncScopePassBase<AsyncScopePass> {
  using AsyncScopePassBase::AsyncScopePassBase;  
  std::size_t deviceDim[2];

  enum DeviceTopologyEnum { Unknown, Path, Ring, Star, Grid, File, Bypass };
  DeviceTopologyEnum deviceTopoType;

  /// If the deviceTopoType is File, this is the path to the file.
  StringRef deviceFilename;
    virtual LogicalResult initialize(MLIRContext *context) override {
    // Initialize prior to parsing
    deviceDim[0] = deviceDim[1] = 0;

    // Get device
    StringRef deviceDef = device;
    StringRef deviceTopoStr =
        deviceDef.take_front(deviceDef.find_first_of('('));

    // Trim the dimensions off of `deviceDef` if dimensions were provided in the
    // string
    if (deviceTopoStr.size() < deviceDef.size())
      deviceDef = deviceDef.drop_front(deviceTopoStr.size());

    if (deviceTopoStr.equals_insensitive("file")) {
      if (deviceDef.consume_front("(")) {
        deviceDef = deviceDef.ltrim();
        if (deviceDef.consume_back(")")) {
          deviceFilename = deviceDef;
          // Remove any leading and trailing single quotes that may have been
          // added in order to pass files with spaces into the pass (required
          // for parsePassPipeline).
          if (deviceFilename.size() >= 2 && deviceFilename.front() == '\'' &&
              deviceFilename.back() == '\'')
            deviceFilename = deviceFilename.drop_front(1).drop_back(1);
          // Make sure the file exists before continuing
          if (!llvm::sys::fs::exists(deviceFilename)) {
            llvm::errs() << "Path " << deviceFilename << " does not exist\n";
            return failure();
          }
        } else {
          llvm::errs() << "Missing closing ')' in device option\n";
          return failure();
        }
      } else {
        llvm::errs() << "Filename must be provided in device option like "
                        "file(/full/path/to/device_file.txt): "
                     << device.getValue() << '\n';
        return failure();
      }
    } else {
      if (deviceDef.consume_front("(")) {
        deviceDef = deviceDef.ltrim();
        deviceDef.consumeInteger(/*Radix=*/10, deviceDim[0]);
        deviceDef = deviceDef.ltrim();
        if (deviceDef.consume_front(","))
          deviceDef.consumeInteger(/*Radix=*/10, deviceDim[1]);
        deviceDef = deviceDef.ltrim();
        if (!deviceDef.consume_front(")")) {
          llvm::errs() << "Missing closing ')' in device option\n";
          return failure();
        }
      }
    }

    deviceTopoType = llvm::StringSwitch<DeviceTopologyEnum>(deviceTopoStr)
                         .Case("path", Path)
                         .Case("ring", Ring)
                         .Case("star", Star)
                         .Case("grid", Grid)
                         .Case("file", File)
                         .Case("bypass", Bypass)
                         .Default(Unknown);
    if (deviceTopoType == Unknown) {
      llvm::errs() << "Unknown device option: " << deviceTopoStr << '\n';
      return failure();
    }

    return success();
  }
  void runOnOperation() override {
    LLVM_DEBUG(llvm::dbgs() << "Ranjani checking: enterd run op\n");
    auto func = getOperation();
    OpBuilder builder(func);

    // Assuming mapping_v2p is available globally or passed as an attribute
    auto mapping_v2p= dyn_cast_if_present<ArrayAttr>(func->getAttr("mapping_v2p")).getValue(); // Example mapping

    // Group operations by their QPU mapping

    DenseMap<Value, Placement::VirtualQ> wireToVirtualQ;
    SmallVector<quake::NullWireOp> sources;
      std::size_t x = deviceDim[0];
    std::size_t y = deviceDim[1];
    std::size_t deviceNumQubits = deviceTopoType == Grid ? x * y : x;

    if (deviceNumQubits && sources.size() > deviceNumQubits) {
      signalPassFailure();
      return;
    }

    if (!deviceNumQubits) {
      x = deviceTopoType == Grid ? std::sqrt(sources.size()) : sources.size();
      y = x;
    }

    // These are captured in the user help (device options in Passes.td), so if
    // you update this, be sure to update that as well.
    Device d;
    if (deviceTopoType == Path)
      d = Device::path(x);
    else if (deviceTopoType == Ring)
      d = Device::ring(x);
    else if (deviceTopoType == Star)
      d = Device::star(/*numQubits=*/x, /*centerQubit=*/y);
    else if (deviceTopoType == Grid)
      d = Device::grid(/*width=*/x, /*height=*/y);
    else if (deviceTopoType == File)
      d = Device::file(deviceFilename);

    if (d.getNumQubits() == 0) {
      func.emitError("Trying to target an empty device.");
      signalPassFailure();
      return;
    }

    LLVM_DEBUG({ d.dump(); });

    if (sources.size() > d.getNumQubits()) {
      func.emitError("Your device [" + device + "] has fewer qubits [" +
                     std::to_string(d.getNumQubits()) +
                     "] than your program is " + "attempting to use [" +
                     std::to_string(sources.size()) + "]");
      signalPassFailure();
      return;
    }
    LLVM_DEBUG(llvm::dbgs() << "Ranjani checking: finished device stuff\n");
    DenseMap<int, SmallVector<Operation *>> qpuGroups;
    int currentComponent=-1;
    

    func.walk([&](Operation *op) {
      LLVM_DEBUG(llvm::dbgs() << "Ranjani checking: walking operations"<<op<<"\n");
      int i=0;
      Value prev;
      if (auto qop = dyn_cast<quake::NullWireOp>(op)) {
      // Assign a new virtual qubit to the resulting wire.
      wireToVirtualQ[qop.getResult()] = Placement::VirtualQ(sources.size());
      sources.push_back(qop);
      } else if (quake::isSupportedMappingOperation(op)) {
        auto wireOperands = quake::getQuantumOperands(op);
        for (auto &&[wire, newWire] :
             llvm::zip_equal(wireOperands, quake::getQuantumResults(op))) {
          // Don't use wireToVirtualQ[a] = wireToVirtualQ[b]. It will work
          // *most* of the time but cause memory corruption other times because
          // DenseMap references can be invalidated upon insertion of new pairs.
          wireToVirtualQ.insert({newWire, wireToVirtualQ[wire]}); // Ranjani: Is this what Eric was talking about- infinite wires?
        }

      }
      auto loc = builder.getUnknownLoc();
      builder.setInsertionPointToStart(&func.getBody().front());

      auto asyncScopeOp = builder.create<quake::AsyncScopeOp>(loc, builder.getI64IntegerAttr(-1));

      // Create a new block for the async_scope body
      auto *asyncScopeBlock = new Block();
      asyncScopeOp.getRegion().push_back(asyncScopeBlock);

      // Set insertion point to the new block
      builder.setInsertionPointToStart(asyncScopeBlock);
      LLVM_DEBUG(llvm::dbgs() << "Ranjani checking: finished preprocessing\n");
      //llvm::errs()
      for (auto wireOp : quake::getQuantumOperands(op)) {
        if (i==0){
          prev=wireOp;
          i++;
          continue;
        }
        auto q1=static_cast<unsigned int>( mapping_v2p[wireToVirtualQ[prev].index].dyn_cast<mlir::IntegerAttr>().getInt());
        auto q2= static_cast<unsigned int>( mapping_v2p[wireToVirtualQ[wireOp].index].dyn_cast<mlir::IntegerAttr>().getInt());
        LLVM_DEBUG(llvm::dbgs() << "Ranjani checking: In loop "<<q1<<" "<<q2<<"\n");
        if (!(d.getComponent(q1)==currentComponent)) {
          if (d.getComponent(q1)==d.getComponent(q2)){
            int qpu = d.getComponent(q1);
            builder.setInsertionPointToEnd(asyncScopeBlock);
            builder.create<quake::AsyncContinueOp>(loc);
            currentComponent=qpu;
            auto loc = builder.getUnknownLoc();

            builder.setInsertionPointToStart(&func.getBody().front());

            auto asyncScopeOp = builder.create<quake::AsyncScopeOp>(loc, builder.getI64IntegerAttr(currentComponent));

            // Create a new block for the async_scope body
            auto *asyncScopeBlock = new Block();
            asyncScopeOp.getRegion().push_back(asyncScopeBlock);

            // Set insertion point to the new block
            builder.setInsertionPointToStart(asyncScopeBlock);
            op->moveBefore(asyncScopeBlock, asyncScopeBlock->end());

          }
          else{
            builder.setInsertionPointToEnd(asyncScopeBlock);
            builder.create<quake::AsyncContinueOp>(loc);
            auto loc = builder.getUnknownLoc();
            currentComponent=-1;

            builder.setInsertionPointToStart(&func.getBody().front());

            auto asyncScopeOp = builder.create<quake::AsyncScopeOp>(loc, builder.getI64IntegerAttr(currentComponent));

            // Create a new block for the async_scope body
            auto *asyncScopeBlock = new Block();
            asyncScopeOp.getRegion().push_back(asyncScopeBlock);

            // Set insertion point to the new block
            builder.setInsertionPointToStart(asyncScopeBlock);
            op->moveBefore(asyncScopeBlock, asyncScopeBlock->end());
          }
        }
        else{
          if (d.getComponent(q1)==d.getComponent(q2)){
            op->moveBefore(asyncScopeBlock, asyncScopeBlock->end());
          }
          else{
            builder.setInsertionPointToEnd(asyncScopeBlock);
            builder.create<quake::AsyncContinueOp>(loc);
            auto loc = builder.getUnknownLoc();
            currentComponent=-1;

            builder.setInsertionPointToStart(&func.getBody().front());

            auto asyncScopeOp = builder.create<quake::AsyncScopeOp>(loc, builder.getI64IntegerAttr(currentComponent));

            // Create a new block for the async_scope body
            auto *asyncScopeBlock = new Block();
            asyncScopeOp.getRegion().push_back(asyncScopeBlock);

            // Set insertion point to the new block
            builder.setInsertionPointToStart(asyncScopeBlock);
            op->moveBefore(asyncScopeBlock, asyncScopeBlock->end());
          }
        }
      }
    });
    //builder.setInsertionPointToEnd(asyncScopeBlock);
    //builder.create<quake::AsyncContinueOp>(loc);

    LLVM_DEBUG(llvm::dbgs() << "Ranjani checking:\n"
                              << func << '\n');
  }

};

} // end anonymous namespace
/*
std::unique_ptr<mlir::Pass> cudaq::opt::createAsyncScopePass() {
  return std::make_unique<AsyncScopePass>();
}*/

//module.push_back()
//funcOp.erase()