Create your Own CUDA Quantum Compiler Pass 
******************************************

The CUDA Quantum IR can be transformed, analyzed, or optimized 
using standard MLIR patterns and tools. CUDA Quantum provides a registration 
mechanism for the :code:`cudaq-opt` tool that allows one to create, load, and 
use custom MLIR passes on Quake code. 

CUDA Quantum MLIR Passes can only be created within an existing CUDA Quantum 
development environment. Therefore, you must clone the repository and add your 
Pass code as part of the existing CUDA Quantum CMake system. 

As an example, clone the repository and add the following directory structure 
under :code:`lib`, :code:`lib/Plugins/MyCustomPlugin/`. Within this directory create a 
:code:`CMakeLists.txt` file and a :code:`MyCustomPlugin.cpp` file. In the CMake file, 
add the following 

.. code:: cmake 

    add_llvm_pass_plugin(MyCustomPlugin MyCustomPlugin.cpp)

Creating a CUDA Quantum IR pass starts with the implementation of an 
:code:`mlir::OperationPass`. A full discussion of the MLIR Pass infrastructure 
is beyond the scope of this document, please see 
`MLIR Passes <https://mlir.llvm.org/docs/PassManagement>`_. To create such 
a pass, start with the following template in the :code:`MyCustomPlugin.cpp` file

.. code:: cpp 
    
    #include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
    #include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
    #include "cudaq/Support/Plugin.h"
    #include "mlir/Rewrite/FrozenRewritePatternSet.h"
    #include "mlir/Transforms/DialectConversion.h"

    // Here is an example MLIR Pass that one can write externally and 
    // use via the cudaq-opt tool, with the --load-cudaq-plugin flag. 
    // The pass here is simple, replace Hadamard operations with S operations. 

    using namespace mlir;

    namespace {

    struct ReplaceH : public OpRewritePattern<quake::HOp> {
      using OpRewritePattern::OpRewritePattern;
      LogicalResult matchAndRewrite(quake::HOp hOp,
                                    PatternRewriter &rewriter) const override {
        rewriter.replaceOpWithNewOp<quake::SOp>(
            hOp, hOp.isAdj(), hOp.getParameters(), hOp.getControls(),
            hOp.getTargets());
        return success();
      }
    };

    class CustomPassPlugin
        : public PassWrapper<CustomPassPlugin, OperationPass<func::FuncOp>> {
    public:
      MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CustomPassPlugin)
  
      llvm::StringRef getArgument() const override { return "cudaq-custom-pass"; }

      void runOnOperation() override {
        auto circuit = getOperation();
        auto ctx = circuit.getContext();

        RewritePatternSet patterns(ctx);
        patterns.insert<ReplaceH>(ctx);
        ConversionTarget target(*ctx);
        target.addLegalDialect<quake::QuakeDialect>();
        target.addIllegalOp<quake::HOp>();
        if (failed(applyPartialConversion(circuit, target, std::move(patterns)))) {
          circuit.emitOpError("simple pass failed");
          signalPassFailure();
        }
      }
    };

    } // namespace

    CUDAQ_REGISTER_MLIR_PASS(CustomPassPlugin)

This example serves as a very simple template for creating custom MLIR 
Passes that analyze the CUDA Quantum Quake representation and perform 
some general transformation. In this example, we create a rewrite pattern 
that replaces :code:`Hadamard` operations with :code:`S` operations. 

Ensure that :code:`add_subdirectory(Plugins)` is in the :code:`lib/CMakeLists.txt` file, 
and also that there is a :code:`lib/Plugins/CMakeLists.txt` file that adds your 
plugin directory with :code:`add_subdirectory`.

Then build CUDA Quantum and you will have a :code:`MyCustomPlugin.so` plugin library 
in the install. You can load the plugin and use it with :code:`cudaq-opt` as follows 

.. code:: bash 

    cudaq-opt --load-cudaq-plugin MyCustomPlugin.so file.qke -cudaq-custom-pass

