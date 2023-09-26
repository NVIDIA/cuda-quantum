Working with the CUDA Quantum IR
********************************
Let's see the output of :code:`nvq++` in verbose mode. Given a simple code like 

.. code-block:: cpp 

    #include <cudaq.h>

    struct ghz {
      void operator()(int N) __qpu__ {
        cudaq::qreg q(N);
        h(q[0]);
        for (int i = 0; i < N - 1; i++) {
          x<cudaq::ctrl>(q[i], q[i + 1]);
        }
        mz(q);
      }
    };

    int main() { ... }

saved to file :code:`simple.cpp`, we see the following output from :code:`nvq++` verbose mode (up to some absolute paths)

.. code-block:: console 

    $ nvq++ simple.cpp -v --save-temps
    
    cudaq-quake --emit-llvm-file simple.cpp -o simple.qke
    cudaq-opt --pass-pipeline=builtin.module(canonicalize,lambda-lifting,canonicalize,apply-op-specialization,kernel-execution,inline{default-pipeline=func.func(indirect-to-direct-calls)},func.func(quake-add-metadata),device-code-loader{use-quake=1},expand-measurements,func.func(lower-to-cfg),canonicalize,cse) simple.qke -o simple.qke.LpsXpu
    cudaq-translate --convert-to=qir simple.qke.LpsXpu -o simple.ll.p3De4L
    fixup-linkage.pl simple.qke simple.ll
    llc --relocation-model=pic --filetype=obj -O2 simple.ll.p3De4L -o simple.qke.o
    llc --relocation-model=pic --filetype=obj -O2 simple.ll -o simple.classic.o
    clang++ -L/usr/lib/gcc/x86_64-linux-gnu/12 -L/usr/lib64 -L/lib/x86_64-linux-gnu -L/lib64 -L/usr/lib/x86_64-linux-gnu -L/lib -L/usr/lib -L/usr/local/cuda/lib64/stubs -r simple.qke.o simple.classic.o -o simple.o
    clang++ -Wl,-rpath,lib -Llib -L/usr/lib/gcc/x86_64-linux-gnu/12 -L/usr/lib64 -L/lib/x86_64-linux-gnu -L/lib64 -L/usr/lib/x86_64-linux-gnu -L/lib -L/usr/lib -L/usr/local/cuda/lib64/stubs simple.o -lcudaq -lcudaq-common -lcudaq-mlir-runtime -lcudaq-builder -lcudaq-ensmallen -lcudaq-nlopt -lcudaq-spin -lcudaq-em-default -lcudaq-platform-default -lnvqir -lnvqir-qpp

The workflow orchestrated above is best visualized in the following figure. 

.. image:: ../../_static/nvqpp_workflow.png

We start by mapping CUDA Quantum C++ kernel representations (structs, lambdas, and free functions) 
to the Quake dialect. Since we added :code:`--save-temps`, 
we can look at the IR code that was produced. :code:`simple.qke` contains 

.. code-block:: console 

    module attributes {qtx.mangled_name_map = {__nvqpp__mlirgen__ghz = "_ZN3ghzclEi"}} {
        func.func @__nvqpp__mlirgen__ghz(%arg0: i32) attributes {"cudaq-entrypoint", "cudaq-kernel"} {
           %alloca = memref.alloca() : memref<i32>
           memref.store %arg0, %alloca[] : memref<i32>
           %0 = memref.load %alloca[] : memref<i32>
           %1 = arith.extsi %0 : i32 to i64
           %2 = quake.alloca(%1 : i64) !quake.veq<?>
           %c0_i32 = arith.constant 0 : i32
           %3 = arith.extsi %c0_i32 : i32 to i64
           %4 = quake.extract_ref %2[%3] : (!quake.veq<?>, i64) -> !quake.ref
           quake.h %4 : (!quake.ref) -> ()
           cc.scope {
            %c0_i32_0 = arith.constant 0 : i32
            %alloca_1 = memref.alloca() : memref<i32>
            memref.store %c0_i32_0, %alloca_1[] : memref<i32>
            cc.loop while {
                %6 = memref.load %alloca_1[] : memref<i32>
                %7 = memref.load %alloca[] : memref<i32>
                %c1_i32 = arith.constant 1 : i32
                %8 = arith.subi %7, %c1_i32 : i32
                %9 = arith.cmpi slt, %6, %8 : i32
                cc.condition %9
            } do {
              cc.scope {
                %6 = memref.load %alloca_1[] : memref<i32>
                %7 = arith.extsi %6 : i32 to i64
                %8 = quake.extract_ref %2[%7] : (!quake.veq<?>, i64) -> !quake.ref
                %9 = memref.load %alloca_1[] : memref<i32>
                %c1_i32 = arith.constant 1 : i32
                %10 = arith.addi %9, %c1_i32 : i32
                %11 = arith.extsi %10 : i32 to i64
                %12 = quake.extract_ref %2[%11] : (!quake.veq<?>, i64) -> !quake.ref
                quake.x [%8] %12 : (!quake.ref, !quake.ref) -> ()
              }
              cc.continue
            } step {
                %6 = memref.load %alloca_1[] : memref<i32>
                %c1_i32 = arith.constant 1 : i32
                %7 = arith.addi %6, %c1_i32 : i32
                memref.store %7, %alloca_1[] : memref<i32>
            }
            }
            %5 = quake.mz %2 : (!quake.veq<?>) -> !cc.stdvec<i1>
            return
        }
    }

This is the base Quake file, unoptimized and unchanged. It is produced by the 
:code:`cudaq-quake` tool, which also allows us to output the full LLVM IR representation 
for the code. This LLVM IR is classical-only, and is directly produced by :code:`clang++` 
code-generation. The LLVM IR file :code:`simple.ll` contains the CUDA Quantum kernel 
:code:`operator()(Args...)` LLVM function, with a mangled name. Ultimately, we 
want to replace this function with our own MLIR-generated function. 

Next, the :code:`cudaq-opt` tool is invoked on the :code:`simple.qke` file. This runs an
MLIR pass-pipeline that canonicalizes and optimizes the code. It will also process quantum 
lambdas, lift those lambdas to functions, and synthesis adjoint and controlled versions of 
CUDA Quantum kernel functions if necessary. The most important pass this this step applies is the 
:code:`kernel-execution` pass, which synthesizes a new entry point LLVM function with the 
same name and signature as the original :code:`operator()(Args...)` call function in the 
classical :code:`simple.ll` file. We also extract all Quake code representations as strings
and register them with the CUDA Quantum runtime for runtime IR introspection. 

After :code:`cudaq-opt`, the :code:`cudaq-translate` tool is used to lower the transformed 
Quake representation to an LLVM IR representation, specifically the QIR. We finish by lowering 
this representation to object code via standard LLVM tools (e.g. :code:`llc`), and merge all 
object files into a single object file, ensuring that our new mangled :code:`operator()(Args...)` 
call is injected first, thereby over-writing the original. Finally, based on user compile flags, 
we configure the link line with specific libraries that implement the :code:`quantum_platform` 
(here the :code:`libcudaq-platform-default.so`) and NVQIR circuit simulator backend (the 
:code:`libnvqir-qpp.so` Q++ CPU-only simulation backend). These latter libraries are controlled 
via the :code:`--platform` and :code:`--target` compiler flags. 

.. image:: ../../_static/dialects.png

The above figure demonstrate the MLIR dialects involved and the overall workflow mapping 
high-level language constructs to lower-level MLIR dialect code, and ultimately LLVM IR. 

CUDA Quantum also provides value-semantics form of Quake for static circuit
representation. This dialect directly enables robust circuit 
optimizations via data-flow analysis of the representative circuit. This dialect 
is typically produced just-in-time when the structure of the circuit is fully known. 

You will notice that there are a number of CUDA Quantum executable tools installed as part 
of this open beta release. These tools are directly related to the generation, 
processing, optimization, and lowering of the core :code:`nvq++` compiler representations.
The tools available are 

1. :code:`cudaq-quake` - Lower C++ to Quake, can also output classical LLVM IR file
2. :code:`cudaq-opt` - Process Quake with various MLIR Passes
3. :code:`cudaq-translate` - Lower Quake to external representations like QIR (or Base Profile QIR)

CUDA Quantum and :code:`nvq++` rely on Quake for the core quantum intermediate representation.
Quake represents an IR closer to the CUDA Quantum source language and models qubits and
quantum instructions via memory semantics. Quake can be fully dynamic and in
that sense represents a quantum circuit template or generator. With runtime 
arguments fully specified, Quake code can be used to generate or synthesize
a fully-known quantum circuit. The value semantics form of Quake can thus be
used as a representation for fully-known
or synthesized quantum circuits. Its utility, therefore, lies in its ability to 
optimize quantum code. It departs from the memory semantics model of Quake and 
expresses the flow of quantum information explicitly as MLIR values.
This approach makes it easier for finding circuit patterns and leveraging it for common 
optimization tasks. 

To demonstrate how these tools work together, let's take the simple GHZ CUDA Quantum 
program and lower the kernel from C++ to Quake, synthesize that Quake code, 
and produce QIR. Recall the code snippet for the kernel

.. code-block:: cpp 

    // Define a quantum kernel
    struct ghz {
      auto operator()() __qpu__ {
        cudaq::qreg q(5);
        h(q[0]);
        for (int i = 0; i < 4; i++) 
          x<cudaq::ctrl>(q[i], q[i + 1]);
        mz(q);
      }
    };

Using the toolchain, we can lower this to directly to QIR

.. code-block:: console

    cudaq-quake simple.cpp | cudaq-opt --canonicalize | cudaq-translate --convert-to=qir 

which prints 

.. code-block:: console 

    ; ModuleID = 'LLVMDialectModule'
    source_filename = "LLVMDialectModule"
    target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
    target triple = "x86_64-unknown-linux-gnu"

    %Array = type opaque
    %Qubit = type opaque
    %Result = type opaque

    declare void @invokeWithControlQubits(i64, void (%Array*, %Qubit*)*, ...) local_unnamed_addr

    declare void @__quantum__qis__x__ctl(%Array*, %Qubit*)

    declare %Result* @__quantum__qis__mz(%Qubit*) local_unnamed_addr

    declare void @__quantum__rt__qubit_release_array(%Array*) local_unnamed_addr

    declare i64 @__quantum__rt__array_get_size_1d(%Array*) local_unnamed_addr

    declare void @__quantum__qis__h(%Qubit*) local_unnamed_addr

    declare i8* @__quantum__rt__array_get_element_ptr_1d(%Array*, i64) local_unnamed_addr

    declare %Array* @__quantum__rt__qubit_allocate_array(i64) local_unnamed_addr

    define void @__nvqpp__mlirgen__ghz(i32 %0) local_unnamed_addr {
      %2 = sext i32 %0 to i64
      %3 = tail call %Array* @__quantum__rt__qubit_allocate_array(i64 %2)
      %4 = tail call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %3, i64 0)
      %5 = bitcast i8* %4 to %Qubit**
      %6 = load %Qubit*, %Qubit** %5, align 8
      tail call void @__quantum__qis__h(%Qubit* %6)
      %7 = add i32 %0, -1
      %8 = icmp sgt i32 %7, 0
      br i1 %8, label %.lr.ph.preheader, label %._crit_edge

    .lr.ph.preheader:                                 ; preds = %1
      %wide.trip.count = zext i32 %7 to i64
      br label %.lr.ph

    .lr.ph:                                           ; preds = %.lr.ph.preheader, %.lr.ph
      %indvars.iv = phi i64 [ 0, %.lr.ph.preheader ], [ %indvars.iv.next, %.lr.ph ]
      %9 = tail call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %3, i64 %indvars.iv)
      %10 = bitcast i8* %9 to %Qubit**
      %11 = load %Qubit*, %Qubit** %10, align 8
      %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
      %12 = tail call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %3, i64 %indvars.iv.next)
      %13 = bitcast i8* %12 to %Qubit**
      %14 = load %Qubit*, %Qubit** %13, align 8
      tail call void (i64, void (%Array*, %Qubit*)*, ...) @invokeWithControlQubits(i64 1, void (%Array*, %Qubit*)* nonnull @__quantum__qis__x__ctl, %Qubit* %11, %Qubit* %14)
      %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
      br i1 %exitcond.not, label %._crit_edge, label %.lr.ph

    ._crit_edge:                                      ; preds = %.lr.ph, %1
      %15 = tail call i64 @__quantum__rt__array_get_size_1d(%Array* %3)
      %16 = icmp sgt i64 %15, 0
      br i1 %16, label %.lr.ph3, label %._crit_edge4

    .lr.ph3:                                          ; preds = %._crit_edge, %.lr.ph3
      %17 = phi i64 [ %22, %.lr.ph3 ], [ 0, %._crit_edge ]
      %18 = tail call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %3, i64 %17)
      %19 = bitcast i8* %18 to %Qubit**
      %20 = load %Qubit*, %Qubit** %19, align 8
      %21 = tail call %Result* @__quantum__qis__mz(%Qubit* %20)
      %22 = add nuw nsw i64 %17, 1
      %exitcond5.not = icmp eq i64 %22, %15
      br i1 %exitcond5.not, label %._crit_edge4, label %.lr.ph3

    ._crit_edge4:                                     ; preds = %.lr.ph3, %._crit_edge
      tail call void @__quantum__rt__qubit_release_array(%Array* %3)
      ret void
    }

    !llvm.module.flags = !{!0}

    !0 = !{i32 2, !"Debug Info Version", i32 3}


Note that the results of each tool can be piped to further tools, creating a
composable pipeline of compiler lowering tools. 


