
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
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/IRMapping.h"

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
    LLVM_DEBUG(llvm::dbgs() << "Ranjani checking: entered run op\n");
    auto func = getOperation();


    // Assuming mapping_v2p is available globally or passed as an attribute
    auto mapping_v2p= dyn_cast_if_present<ArrayAttr>(func->getAttr("mapping_v2p")).getValue(); // Example mapping

    // Group operations by their QPU mapping

    DenseMap<Value, Placement::VirtualQ> wireToVirtualQ;
    SmallVector<quake::AllocaOp> sources;
    SmallVector<Operation *> sinksToRemove;
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
    SmallVector< SmallVector<Operation *>> qpuGroups;
    SmallVector<Operation *> CurrentVector;
    mlir::SmallVector<int> AsyncComponentIds;
    int numBlocks=0;
    int currentComponent=-1;
    

    func.walk([&](Operation *op) {
      LLVM_DEBUG(llvm::dbgs() << "Ranjani checking: walking operations"<<op<<"\n");
      int i=0;
      Value prev;
      if (auto qop = dyn_cast<quake::AllocaOp>(op)) {
        // Assign a new virtual qubit to the resulting wire.
        wireToVirtualQ[qop.getResult()] = Placement::VirtualQ(sources.size());
        LLVM_DEBUG(llvm::dbgs() << "Ranjani checking: walking operations"<<op<<"\n");
        sources.push_back(qop);
      } else if (quake::isSupportedMappingOperation(op)) {
          if (isa<quake::SinkOp>(op)) {
            sinksToRemove.push_back(op);
          }
          // else {
          //   auto wireOperands = quake::getQuantumOperands(op);
          //   LLVM_DEBUG(llvm::dbgs() << "Ranjani checking: zip" <<quake::getQuantumResults(op).size()<<" "<<wireOperands.size()<<" "<<"\n");
          //   for (auto &&[wire, newWire] :
          //       llvm::zip_equal(wireOperands, quake::getQuantumResults(op))) {
          //     // Don't use wireToVirtualQ[a] = wireToVirtualQ[b]. It will work
          //     // *most* of the time but cause memory corruption other times because
          //     // DenseMap references can be invalidated upon insertion of new pairs.
          //     wireToVirtualQ.insert({newWire, wireToVirtualQ[wire]}); // Ranjani: Is this what Eric was talking about- infinite wires?
          //   }
          // }
        

          LLVM_DEBUG(llvm::dbgs() << "Ranjani checking: finished preprocessing\n");
        //llvm::errs()
          for (auto wireOp : quake::getQuantumOperands(op)) {
            LLVM_DEBUG(llvm::dbgs() << "Ranjani checking: zip" <<wireOp<<"\n");
            if (i==0){
              prev=wireOp;
              i++;
              LLVM_DEBUG(llvm::dbgs() << "Ranjani checking: In loop IF "<<"\n");

              // if(quake::getQuantumOperands(op).size()==1){
              //   auto q1=static_cast<unsigned int>( mapping_v2p[wireToVirtualQ[prev].index].dyn_cast<mlir::IntegerAttr>().getInt());
              //   LLVM_DEBUG(llvm::dbgs() << "Ranjani checking: In loop "<<q1<<"\n");
              //   if (!(d.getComponent(q1)==currentComponent)) {
              //     numBlocks++;
              //     currentComponent=d.getComponent(q1);
              //     qpuGroups.push_back(CurrentVector);
              //     CurrentVector.clear();
              //     CurrentVector.push_back(op);
              //     AsyncComponentIds.push_back(currentComponent);
              //   }
              // }
              continue;
            }
            auto q1=static_cast<unsigned int>( mapping_v2p[wireToVirtualQ[prev].index].dyn_cast<mlir::IntegerAttr>().getInt());
            auto q2= static_cast<unsigned int>( mapping_v2p[wireToVirtualQ[wireOp].index].dyn_cast<mlir::IntegerAttr>().getInt());
            LLVM_DEBUG(llvm::dbgs() << "Ranjani checking: In loop "<<q1<<" "<<q2<<"\n");
            //auto q1=static_cast<unsigned int>( mapping_v2p[wireToVirtualQ[prev].index].dyn_cast<mlir::IntegerAttr>().getInt());
            //auto q2= static_cast<unsigned int>( mapping_v2p[wireToVirtualQ[wireOp].index].dyn_cast<mlir::IntegerAttr>().getInt());
            //LLVM_DEBUG(llvm::dbgs() << "Ranjani checking: In loop "<<q1<<" "<<q2<<"\n");
            if (!(d.getComponent(q1)==currentComponent)) {        
              if (d.getComponent(q1)==d.getComponent(q2)){
                int qpu = d.getComponent(q1);
                currentComponent=qpu;
                numBlocks++;
                qpuGroups.push_back(CurrentVector);
                CurrentVector.clear();
                CurrentVector.push_back(op);
                AsyncComponentIds.push_back(currentComponent);
              }
              else{

                currentComponent=-1;
                numBlocks++;
                qpuGroups.push_back(CurrentVector);
                CurrentVector.clear();
                CurrentVector.push_back(op);
                AsyncComponentIds.push_back(currentComponent);
              }
            }
            else{
              if (d.getComponent(q1)==d.getComponent(q2)){
                CurrentVector.push_back(op);
              }
              else{
                currentComponent=-1;
                numBlocks++;
                qpuGroups.push_back(CurrentVector);
                CurrentVector.clear();
                CurrentVector.push_back(op);
                AsyncComponentIds.push_back(currentComponent);
              }
            }
          }
      }
      
    });
    //builder.setInsertionPointToEnd(asyncScopeBlock);
    //builder.create<quake::AsyncContinueOp>(loc);
    ModuleOp module = func->getParentOfType<ModuleOp>();
    OpBuilder builder(module.getBodyRegion());
    IRMapping mapping;
    auto funcType = builder.getFunctionType(/*inputs=*/TypeRange{}, /*results=*/TypeRange{});

    // Create the new function
    auto newFunc = builder.create<func::FuncOp>(func.getLoc(), "new_function", funcType);

    for (auto argPair : llvm::zip(func.getArguments(), newFunc.getArguments())) {
      mapping.map(std::get<0>(argPair), std::get<1>(argPair));
    }
    // Add a new block to the function
    Block *entryBlock = newFunc.addEntryBlock();

    // Set the insertion point to the new block
    builder.setInsertionPointToStart(entryBlock);
    

    //LLVM_DEBUG(llvm::dbgs() << "Ranjani checking:\n"<< func << '\n');

    //auto loc = newFunc.getLoc();


    // auto asyncScopeOp = builder.create<quake::AsyncScopeOp>(loc, builder.getI64IntegerAttr(-1));
    // auto qpuIdAttr = builder.getI64IntegerAttr(-1);
    // asyncScopeOp->setAttr("qpuId", qpuIdAttr);
    // // Create a new block for the async_scope body
    // Block *asyncScopeBlock = builder.createBlock(&asyncScopeOp.getRegion());
    // //asyncScopeOp.getRegion().push_back(asyncScopeBlock);
    // // Set insertion point to the new block
    // builder.setInsertionPointToStart(asyncScopeBlock);
    // Move operations into the new block
    for (Operation *op : sources) {
      builder.insert(op->clone());
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Ranjani checking: sources added"<< '\n');
    //builder.create<func::ReturnOp>(newFunc.getLoc());
    // Create async_scope_end and set insertion point after async_scope
    //builder.setInsertionPointToEnd(entryBlock);
    //builder.create<quake::AsyncContinueOp>(newFunc.getLoc());
    for (size_t i = 0; i < qpuGroups.size(); ++i) {
      int blockNum = i;
      int qpu=AsyncComponentIds[blockNum];
      // Create async_scope and async_scope_end
      auto loc = newFunc.getLoc();

      auto asyncScopeOp = builder.create<quake::AsyncScopeOp>(loc, builder.getI64IntegerAttr(qpu));
      auto qpuIdAttr = builder.getI64IntegerAttr(qpu);
      asyncScopeOp->setAttr("qpuId", qpuIdAttr);
      LLVM_DEBUG(llvm::dbgs() << "Ranjani checking: Created async scope op"<<blockNum<< '\n');
      // Create a new block for the async_scope body
      Block *asyncScopeBlock = builder.createBlock(&asyncScopeOp.getRegion());
      LLVM_DEBUG(llvm::dbgs() << "Ranjani checking: created new block"<< '\n');
      //asyncScopeOp.getRegion().push_back(asyncScopeBlock);
      LLVM_DEBUG(llvm::dbgs() << "Ranjani checking: block added"<< '\n');
      // Set insertion point to the new block
      builder.setInsertionPointToStart(asyncScopeBlock);
      LLVM_DEBUG(llvm::dbgs() << "Ranjani checking: insertion point"<< '\n');
      // Move operations into the new block
      for (Operation *op : qpuGroups[i]) {
        //builder.insert(op->clone());
        auto clonedOp = builder.clone(*op, mapping);
        // Update the mapping for the results of the cloned operation
        for (auto [oldResult, newResult] : llvm::zip(op->getResults(), clonedOp->getResults())) {
          mapping.map(oldResult, newResult);
        }
        op->print(llvm::dbgs());
        LLVM_DEBUG(llvm::dbgs() << "Ranjani checking: adding ops"<<op<<'\n');
      }

      // Create async_scope_end and set insertion point after async_scope
      //builder.setInsertionPointToEnd(asyncScopeBlock);
      
      //builder.create<func::ReturnOp>(newFunc.getLoc());
      builder.setInsertionPointToEnd(entryBlock);
      LLVM_DEBUG(llvm::dbgs() << "Ranjani checking: insertion point end"<< '\n');
      builder.create<quake::AsyncContinueOp>(newFunc.getLoc());
      LLVM_DEBUG(llvm::dbgs() << "Ranjani checking: async continue added"<< '\n');
      
    }
    auto returnOp = func.getBody().front().getTerminator();
    SmallVector<Value, 4> mappedOperands;
    for (Value operand : returnOp->getOperands()) {
      mappedOperands.push_back(mapping.lookupOrNull(operand));
    }
    builder.create<func::ReturnOp>(returnOp->getLoc(), mappedOperands);
    SymbolTable::setSymbolVisibility(newFunc, SymbolTable::getSymbolVisibility(func));
    //builder.create<quake::AsyncContinueOp>(newFunc.getLoc());
    

    //LLVM_DEBUG(llvm::dbgs() << "Ranjani checking:\n"<< newFunc << '\n');
    
  }

};

} // end anonymous namespace
/*
std::unique_ptr<mlir::Pass> cudaq::opt::createAsyncScopePass() {
  return std::make_unique<AsyncScopePass>();
}
,
           
    OpBuilder<(ins "mlir::IntegerAttr":$qpu_id), [{
      return build($_builder, $_state, qpu_id);
    }]>
*/

//module.push_back()
//funcOp.erase()

// If value semantic model: Need to change async scope op to have a return type 'wire'--all the modified wirtes. 
// remove the first async scope op.

//try memtoreg

//extract Op
// Adam Geller --- appending IDs to Qrefs.