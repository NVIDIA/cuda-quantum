Create your Own MLIR Pass 
*************************

The CUDA Quantum IR can be transformed, analyzed, or optimized 
using standard MLIR patterns and tools. CUDA Quantum provides a registration 
mechanism for the :code:`cudaq-opt` tool that allows one to create, load, and 
use custom MLIR passes on Quake and QTX code. 

Creating a CUDA Quantum IR pass starts with the implementation of an 
:code:`mlir::OperationPass`. A full discussion of the MLIR Pass infrastructure 
is beyond the scope of this document, please see `MLIR Passes <https://mlir.llvm.org/docs/PassManagement>`_. To create such 
a pass, start with the following template 

.. code:: cpp 
    
    #include "cudaq/Optimizer/Dialect/QTX/QTXDialect.h"
    #include "cudaq/Optimizer/Dialect/QTX/QTXOps.h"
    #include "cudaq/Support/Plugin.h"
    #include "llvm/Analysis/CallGraph.h"
    #include "mlir/IR/BuiltinOps.h"

    using namespace mlir;

    namespace {

      class HelloWorldQTXPass
        : public PassWrapper<HelloWorldQTXPass, OperationPass<qtx::CircuitOp>> {
      public:
        MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HelloWorldQTXPass)

        llvm::StringRef getArgument() const override {
          return "cudaq-hello-world-qtx";
        }

        void runOnOperation() override {
          auto circuit = getOperation();
          llvm::errs() << "-- dump the module\n";
          circuit.dump();
        }
      };

    } // namespace

    CUDAQ_REGISTER_MLIR_PASS(HelloWorldQTXPass)

The CMake to configure and build this is as follows 

.. code:: cmake 

    add_llvm_pass_plugin(HelloWorldQTXPass HelloWorldQTXPass.cpp)
    get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
    get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
    target_link_libraries(HelloWorldQTXPass PRIVATE ${dialect_libs} ${conversion_libs})

Configure and building this :code:`HelloWorldQTXPass` will produce a 
library that can be loaded and used with :code:`cudaq-opt`. 

.. code:: bash 

    cudaq-opt --load-pass-plugin HelloWorldQTXPass.so file.qtx -cudaq-hello-world-qtx

