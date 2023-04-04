/********************************************************** -*- tablegen -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#ifndef CUDAQ_OPTIMIZER_DIALECT_CC_OPS
#define CUDAQ_OPTIMIZER_DIALECT_CC_OPS

include "mlir/Interfaces/InferTypeOpInterface.td"
include "cudaq/Optimizer/Dialect/Common/Traits.td"
include "cudaq/Optimizer/Dialect/CC/CCDialect.td"
include "cudaq/Optimizer/Dialect/CC/CCTypes.td"

def AnyPointerType : Type<CPred<"$_self.isa<mlir::LLVM::LLVMPointerType,"
                          "cudaq::cc::PointerType>()">, "any pointer type">;

def AnyCallableType : Type<CPred<"$_self.isa<cudaq::cc::LambdaType, "
                          "mlir::FunctionType>()">, "any callable type">;

//===----------------------------------------------------------------------===//
// Base operation definition.
//===----------------------------------------------------------------------===//

class CCOp<string mnemonic, list<Trait> traits = []> :
    Op<CCDialect, mnemonic, traits>;

def cc_ScopeOp : CCOp<"scope",
        [AutomaticAllocationScope, RecursiveMemoryEffects, NoRegionArguments,
         DeclareOpInterfaceMethods<RegionBranchOpInterface,
                                   ["getNumRegionInvocations",
                                    "getRegionInvocationBounds"]>]> {
  let summary = "A compound statement in which allocations are scoped.";
  let description = [{
    The cc.scope Op is used to delineate the scope of allocations, such as for
    variables declared in a compound statement. All such allocations that appear
    in the context of the ScopeOp are dead when the ScopeOp exits.

    This allows us to lower to LLVM IR with the appropriate calls to intrinsics
    `llvm.stacksave` and `llvm.stackrestore`. It also gives a point to `free`
    (deconstruct) any compiler-generated heap allocations.

    A ScopeOp that contains no allocations has no semantics and can be inlined
    into the parent Region.

    ```mlir
      cc.scope {
        %alloc = cc.alloca i32
        ...
      }
      // %alloc is deallocated at this point
    ```
  }];

  let results = (outs Variadic<AnyType>:$results);
  let regions = (region AnyRegion:$initRegion);
  
  let hasCustomAssemblyFormat = 1;
  let hasCanonicalizer = 1;

  let builders = [
    OpBuilder<(ins 
      CArg<"llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)>",
           "nullptr">)>
  ];

  let extraClassDeclaration = [{
    using BodyBuilderFn =
        llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)>;
  }];
}

def cc_LoopOp : CCOp<"loop",
        [AutomaticAllocationScope, RecursiveMemoryEffects,
         DeclareOpInterfaceMethods<LoopLikeOpInterface>,
         DeclareOpInterfaceMethods<RegionBranchOpInterface>]> {
  let summary = "generalized loop construct";
  let description = [{
    The `cc.loop` operation is a very generalized loop-like structure that
    can capture generalized loop structures such as C's `for`, `while`, and
    `do while`. It is in essence a gated backedge with four distinct phases.
    This loop construct takes no arguments and generates no results. All
    arguments and results must be conveyed via memory operations.

    ```mlir
    cc.loop while { ... } do { ... } step { ... }
    ```

    A C++ `for` loop
    ```c++
    for (@init; @while; @step) { @do }
    ```
    is lowered to
    ```mlir
    cc.scope {
      @init
      cc.loop while { @while} do { @do } step { @step }
    }
    ```
    A `cc.loop` op terminates the `cc.scope`.

    A C++ `while` loop
    ```c++
    while (@while) { @do }
    ```
    maps as (<step> region is empty)
    ```mlir
    cc.loop while { @while} do { @do } // step {}
    ```

    A C++ `do while` loop
    ```c++
    do { @do } while (@while);
    ```
    maps as (<step> region is empty)
    ```mlir
    cc.loop do { @do } while { @while} // step {}
    ```
    The semantics are identical to a C `do while` loop in that the body of the
    loop will be executed one time before the control condition is evaluated.
  }];

  let arguments = (ins
    Variadic<AnyType>:$initArgs,
    BoolAttr:$post_condition
  );
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region
    SizedRegion<1>:$whileRegion,
    AnyRegion:$bodyRegion,
    AnyRegion:$stepRegion
  );

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "mlir::ValueRange":$iterArgs, "bool":$postCond,
      "llvm::function_ref<void(mlir::OpBuilder &, mlir::Location, "
                              "mlir::Region &)>":$whileBuilder,
      "llvm::function_ref<void(mlir::OpBuilder &, mlir::Location, "
                              "mlir::Region &)>":$bodyBuilder,
      CArg<"llvm::function_ref<void(mlir::OpBuilder &, mlir::Location, "
                                   "mlir::Region &)>",
                                   "nullptr">:$stepBuilder)>,
    OpBuilder<(ins "mlir::TypeRange":$results, "mlir::ValueRange":$iterArgs,
      "bool":$postCond,
      "llvm::function_ref<void(mlir::OpBuilder &, mlir::Location, "
                              "mlir::Region &)>":$whileBuilder,
      "llvm::function_ref<void(mlir::OpBuilder &, mlir::Location, "
                              "mlir::Region &)>":$bodyBuilder,
      CArg<"llvm::function_ref<void(mlir::OpBuilder &, mlir::Location, "
                                   "mlir::Region &)>",
                                   "nullptr">:$stepBuilder)>
  ];

  let extraClassDeclaration = [{
    using RegionBuilderFn = llvm::function_ref<void(mlir::OpBuilder &,
        mlir::Location, mlir::Region &)>;

    bool hasArguments() { return getOperands().size(); }
    constexpr static llvm::StringRef postCondAttrName() {
      return llvm::StringLiteral("post_condition");
    }
    bool isPostConditional() {
      return getOperation()
          ->getAttrOfType<mlir::IntegerAttr>(postCondAttrName())
          .getInt();
    }

    mlir::Block *getWhileBlock() { return &getWhileRegion().front(); }
    mlir::Block::BlockArgListType getWhileArguments() {
      return getWhileBlock()->getArguments();
    }
    mlir::Block *getDoEntryBlock() { return &getBodyRegion().front(); }
    mlir::Block::BlockArgListType getDoEntryArguments() {
      return getDoEntryBlock()->getArguments();
    }
    bool hasStep() { return !getStepRegion().empty(); }
    mlir::Block *getStepBlock() {
      return hasStep() ? &getStepRegion().front() : nullptr;
    }
    mlir::Block::BlockArgListType getStepArguments() {
      if (hasStep())
        return getStepBlock()->getArguments();
      return {};
    }

    mlir::OperandRange
    getSuccessorEntryOperands(std::optional<unsigned> index);
  }];
}

def cc_IfOp : CCOp<"if",
    [DeclareOpInterfaceMethods<RegionBranchOpInterface,
                               ["getNumRegionInvocations",
                                "getRegionInvocationBounds"]>,
     RecursiveMemoryEffects, NoRegionArguments]> {
  let summary = "if-then-else operation";
  let description = [{
    ```mlir
    cc.if (%cond) {
      ^bb0:
        ...
    } else {
      ^bb9:
        ...
    }
    ```
  }];

  let arguments = (ins I1:$condition);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region AnyRegion:$thenRegion, AnyRegion:$elseRegion);

  let builders = [
    OpBuilder<(ins "mlir::TypeRange":$resultTypes, "mlir::Value":$cond,
      "llvm::function_ref<void(mlir::OpBuilder &, mlir::Location, "
                              "mlir::Region &)>":$thenBuilder,
      CArg<"llvm::function_ref<void(mlir::OpBuilder &, mlir::Location, "
           "mlir::Region &)>", "nullptr">:$elseBuilder)>
  ];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    using RegionBuilderFn = llvm::function_ref<void(mlir::OpBuilder &,
        mlir::Location, mlir::Region &)>;

    bool hasResults() { return getResults().size(); }
  }];
}

def cc_ConditionOp : CCOp<"condition",
        [Pure, Terminator, ParentOneOf<["LoopOp"]>,
         DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface>]> {
  let summary = "Conditional branch in a where region's basic block.";
  let description = [{
    The condition op is used as the terminator of the basic block in the where
    region of a loop op. It takes, at minimum, an `i1` value as the control
    condition. If the value is true, a branch to the body of the loop is taken.
    If the value is false, the loop op exits.

    Any additional arguments to the condition op are forwarded as block
    arguments to the body or the result of the loop op accordingly.
  }];

  let arguments = (ins
    I1:$condition,
    Variadic<AnyType>:$results
  );
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

  let assemblyFormat = [{
    $condition attr-dict ( `(` $results^ `:` qualified(type($results)) `)` )?
  }];
  let hasVerifier = 1;
}

def cc_ContinueOp : CCOp<"continue", [Pure, ReturnLike, Terminator,
        ParentOneOf<["LoopOp", "ScopeOp", "IfOp"]>]> {
  let summary = "Continue branch.";
  let description = [{
    The continue op is overloaded. In the body region of a loop op, the continue
    op has the standard C semantics of a branch to the next iteration of the
    loop. More precisely, a branch to the step region's basic block.

    When used in other ops or regions of a loop op, the continue op is an
    unconditional branch that forwards its arguments to the next logical region
    or to the op's results.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
  let assemblyFormat = [{
    attr-dict ($operands^ `:` qualified(type($operands)))?
  }];
}

def cc_BreakOp : CCOp<"break",
        [Pure, ReturnLike, Terminator, ParentOneOf<["LoopOp"]>]> {
  let summary = "Break branch.";
  let description = [{
    The break op can be used in a loop op's body region. This terminator is
    a branch immediately from the body to exit the loop op. No other operations
    in the loop op (from any region) will be executed.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
  let assemblyFormat = [{
    attr-dict ($operands^ `:` qualified(type($operands)))?
  }];
}

// TODO: Should relax the arguments restriction. Lambdas can have arguments in
// addition to free variables.
def cc_CreateLambdaOp : CCOp<"create_lambda",
        [AutomaticAllocationScope, RecursiveMemoryEffects,
         SingleBlockImplicitTerminator<"cudaq::cc::ReturnOp">]> {
  let summary = "Create a lambda value.";
  let description = [{
    Creates a λ expression. A λ expression is a Callable instance that may have
    associated data (or data references) that it captures. Currently, captured
    data is implied by using values from the enclosing regions in the body of
    the create_lambda.
  }];
  let results = (outs cc_LambdaType:$signature);
  let regions = (region SizedRegion<1>:$initRegion);
  let hasCustomAssemblyFormat = 1;

  let builders = [
      OpBuilder<(ins
        "cudaq::cc::LambdaType":$signature,
        CArg<"llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)>",
           "nullptr">)>];

  let extraClassDeclaration = [{
    using BodyBuilderFn =
        llvm::function_ref<void(mlir::OpBuilder &, mlir::Location)>;

    llvm::ArrayRef<mlir::Type> getResultTypes() {
      return getType().getSignature().getResults();
    }

    unsigned getNumResults() {
      return getType().getSignature().getNumResults();
    }
  }];
}

def cc_AllocaOp : CCOp<"alloca", [
    MemoryEffects<[MemAlloc<AutomaticAllocationScopeResource>]>]> {
  let summary = "Allocate a dynamic block of memory on the stack.";
  let description = [{
    This is similar to the C library `alloca()` function. The number of bytes
    to be allocated is computed by the size of the `elementType` (which may not
    have a non-positive size) times `seqSize` if present or `1` if `seqSize` is
    omitted. This returns a pointer to the first element in the allocated
    sequence of `seqSize` elements of `elementType`.
  }];

  let arguments = (ins
    TypeAttr:$elementType,
    Optional<AnyInteger>:$seqSize
  );
  let results = (outs cc_PointerType:$address);

  let assemblyFormat = [{
    $elementType (`[` $seqSize^ `]`)? attr-dict `:`
      functional-type(operands, results)
  }];
}

def cc_InstantiateCallableOp : CCOp<"instantiate_callable", [Pure]> {
  let summary = "Fine-grained construction of a C++ Callable.";
  let description = [{
    When we lift a λ expression, a `cc.create_lambda`, to a set of functions,
    this operation can be used to construct a pair that is the captured values
    and a pointer to the associated currying trampoline function.
  }];

  let arguments = (ins
    SymbolRefAttr:$callee,
    Variadic<AnyType>:$closure_data
  );
  let results = (outs cc_LambdaType:$signature);

  let assemblyFormat = [{
    $callee `(` $closure_data `)` attr-dict `:`
      functional-type(operands, results)
  }];
}

def cc_CallableFuncOp : CCOp<"callable_func", [Pure]> {
  let summary = "Extract the function from the λ expression.";
  let description = [{
    A λ expression contains both code (a function) and data (a closure). This
    op can be used to retrieve the function from the λ expression.
  }];

  let arguments = (ins cc_LambdaType:$callable);
  let results = (outs FunctionType:$function);

  let assemblyFormat = [{
    $callable attr-dict `:` functional-type(operands, results)
  }];
}

def cc_CallableClosureOp : CCOp<"callable_closure", [Pure]> {
  let summary = "Extract the closure data from the λ expression.";
  let description = [{
    A λ expression contains both code (a function) and data (a closure). This
    op can be used to retrieve the closure (data) from the λ expression. The
    unpacked values are returned as a (transparent) tuple of values. A closure
    has a (compile-time) constant number of values. The tuple cannot be extended
    dynamically.
  }];

  let arguments = (ins cc_LambdaType:$callable);
  let results = (outs Variadic<AnyType>:$closure_data);

  let assemblyFormat = [{
    $callable attr-dict `:` functional-type(operands, results)
  }];
}

def cc_CallCallableOp : CCOp<"call_callable", [CallOpInterface]> {
  let summary = "Abstract application of a callable.";
  let description = [{
    Call a function of C++ callable (a closure). The call target is specified
    as an SSA-value as the first argument.

    For a plain old function, the call target will effectively degenerate to a
    pointer to a function (code) and the argument list must be a type correct
    one-to-one match. The closure of the calling context is empty and can be
    elided.

    A callable defined as a C++ `class` will pass an object instance (`this`
    pointer) as a special (free) argument. C++ semantics are that all calling
    state must be included as members of this instance or passed as arguments.

    For a λ expression callable, the call target will involve a trampoline to
    load any captured free variables as special arguments to the ultimate target
    function.
  }];

  let arguments = (ins
    AnyCallableType:$callee,
    Variadic<AnyType>:$args
  );
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;

  let assemblyFormat = [{
    $callee (`,` $args^)? attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return ++operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    mlir::CallInterfaceCallable getCallableForCallee() { return getCallee(); }

    mlir::FunctionType getFunctionType() {
      return mlir::FunctionType::get(getContext(), getOperands().getType(),
        getResults().getTypes());
    }
  }];
}

def cc_UnwindBreakOp : CCOp<"unwind_break", [ 
        ParentOneOf<["IfOp", "ScopeOp"]>]> {
  let summary = "Non-local break (exiting) branch with unwind semantics.";
  let description = [{
    An UnwindBreakOp may imply unwinding the stack frame for the current
    activation. In the following example the break at `(1)` is _not_ a branch
    to the Op `%exit`. Instead it unwinds the various scopes, `S1, S2, S3, S4`,
    consecutively and in the specified order before exiting the loop as in
    `(2)`. Note that this is _not_ the same semantics as `cc.return` or
    `cc.unwind_return`.

    Because an UnwindBreakOp is terminating the current innermost loop, its
    arguments are the return values for the loop, if present.

    ```mlir
      func.func @example() -> (i32, i32) {
        %0 = ... : f64
        %ival = cc.loop while ((%arg0 = %0) -> f64) {
          ...
        } body {
         ^bb0(%arg0 : f64):
          ... // (S4)
          cc.scope { // (S3)
            cc.if ... {
              cc.scope { // (S2)
                cc.loop ... {
                  cc.scope { // (S1)
                    cc.if ... {
                      ...
                      cc.unwind_break %val : f64 // (1)
                    }
                  }
                }
              }
            }
          } step {
           ^bb0(%arg0 : f64):
            ...
          }
        }
        %exit = ...
      }
    ```

    An UnwindBreakOp is clearly not pure. It is also not a terminator. Note
    that it cannot be a ReturnLikeOp because it's arguments do not correspond
    to the nearest enclosing structured Op in any way. MLIR's builtin
    verification does not support Ops with non-local return semantics.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let assemblyFormat = [{
    ($operands^ `:` qualified(type($operands)))? attr-dict
  }];

  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];  
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def cc_UnwindContinueOp : CCOp<"unwind_continue", [ 
        ParentOneOf<["IfOp", "ScopeOp"]>]> {
  let summary = "Non-local continue branch with unwind semantics.";
  let description = [{
    An UnwindContinueOp may imply unwinding the stack frame for the current
    activation. In the following example the continue at `(1)` is _not_ a branch
    to the Op `%next_iter`. Instead it unwinds the various scopes, `S1, S2, S3,
    S4`, consecutively and in the specified order before exiting the loop as
    in `(2)`. Note that this is _not_ the same semantics as `cc.return` or
    `cc.unwind_return`.

    Because an UnwindContinueOp is jumping to the next iteration of the current
    innermost loop, its arguments are the return values for the loop's backedge,
    if present.

    ```mlir
      func.func @example() -> (i32, i32) {
        %0 = ... : f64
        %ival = cc.loop while ((%arg0 = %0) -> f64) {
          ...
        } body {
         ^bb0(%arg0 : f64):
          ... // (S4)
          cc.scope { // (S3)
            cc.if ... {
              cc.scope { // (S2)
                cc.loop ... {
                  cc.scope { // (S1)
                    %0 = cc.if ... -> i32 {
                      ...
                      cc.unwind_continue %val : f64 // (1)
                    }
                  }
                }
              }
            }
          } step {
           ^bb0(%arg0 : f64):
            %next_iter = ...
            ...
          }
        }
      }
    ```

    An UnwindContinueOp is clearly not pure. It is also not a terminator. Note
    that it cannot be a ReturnLikeOp because it's arguments do not correspond
    to the nearest enclosing structured Op in any way. MLIR's builtin
    verification does not support Ops with non-local return semantics.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let assemblyFormat = [{
    ($operands^ `:` qualified(type($operands)))? attr-dict
  }];

  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];  
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def cc_UnwindReturnOp : CCOp<"unwind_return", [ 
        ParentOneOf<["LoopOp", "IfOp", "ScopeOp"]>]> {
  let summary = "Non-local return (exiting) branch with unwind semantics.";
  let description = [{
    An UnwindReturnOp may imply unwinding the stack frame for the current
    activation. In the following example the return at `(1)` is _not_ a branch
    to the block `^exit`. Instead it unwinds the various scopes, `S1, S2, S3,
    S4`, consecutively and in the specified order before exiting the
    function as in `(2)`. Note that this is _not_ the same semantics as
    `cc.return`.

    Because an UnwindReturnOp is terminating the current function, its
    arguments are the return values for the function, if present. Returning a
    pointer returned by an `alloca` operation has undefined semantics. The
    compiler may raise an error if such a case exists.

    ```mlir
      func.func @example() -> i32 {
        ... // (S4)
        cc.scope { // (S3)
          cc.if ... {
            cc.scope { // (S2)
              cc.loop ... {
                cc.scope { // (S1)
                  cc.if ... {
                    ...
                    cc.unwind_return %val : i32 // (1)
                  }
                }
              }
            }
          }
        }
        ...
       ^exit:
        func.return %result : i32 // (2)
      }
    ```

    An UnwindReturnOp is clearly not pure. It is also not a terminator. Note
    that it cannot be a ReturnLikeOp because it's arguments do not correspond
    to the nearest enclosing structured Op in any way. MLIR's builtin
    verification does not support Ops with non-local return semantics.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let assemblyFormat = [{
    ($operands^ `:` qualified(type($operands)))? attr-dict
  }];

  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];  
  let hasVerifier = 1;
}

def cc_ReturnOp : CCOp<"return", [Pure, ReturnLike, Terminator,
        ParentOneOf<["mlir::func::FuncOp", "CreateLambdaOp"]>]> {
  let summary = "Return (exiting) branch.";
  let description = [{
    A ReturnOp returns control from the current activation to the enclosing
    dynamic scope, the calling function.

    In callable expressions, the ReturnOp is always to end the execution of the
    callable.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
  let assemblyFormat = [{
    ($operands^ `:` qualified(type($operands)))? attr-dict
  }];

  let hasVerifier = 1;
  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    /// Return true if the cc.return is directly owned by a func.func's region.
    /// If it is in a region of some other op, returns false.
    bool ownedByFuncOp() {
      if (auto *region = getOperation()->getParentRegion())
        return isa<mlir::func::FuncOp>(region->getParentOp());
      return false;
    }
  }];
}

def cc_StdvecInitOp : CCOp<"stdvec_init", [Pure]> {
  let summary = "Initialize a stdvec object from a pointer and length.";
  let description = [{
    The stdvec_init op can be used to create a "high-level" std::vector object
    in cc. The construction is similar to an initializer list constructor.
    It takes two arguments: a pointer to a memory buffer and the length of
    the buffer as a number of elements.

    ```mlir
    %buff = ... : !llvm.ptr<f64>
    %len = ... : i64
    %svec = cc.stdvec_init %buff, %len : (!llvm.ptr<f64>, i64) -> !cc.stdvec<f64>
    func.call @kernel(%svec) ...
    ```

    Note this op has value semantics and does not touch memory. It is used to
    create an abstract aggregate value.
  }];

  let arguments = (ins AnyPointerType:$buffer, AnyInteger:$length);
  let results = (outs cc_StdVectorType:$stdvec);

  let assemblyFormat = [{
    $buffer `,` $length attr-dict `:` functional-type(operands, results)
  }];
}

def cc_StdvecDataOp : CCOp<"stdvec_data", [Pure]> {
  let summary = "Retrieve the data pointer from a stdvec object.";
  let description = [{
    The stdvec_data op can be used to retreive the data pointer from a stdvec
    object. Analogous to a call to `vec.data()`.

    Note that this operation has value semantics. It does not touch memory as
    the data pointer is part of the aggregate value and nothing is dereferenced.
  }];

  let arguments = (ins cc_StdVectorType:$stdvec);
  let results = (outs AnyPointerType:$data);

  let assemblyFormat = [{
    $stdvec attr-dict `:` functional-type(operands, results)
  }];
}

def cc_StdvecSizeOp : CCOp<"stdvec_size", [Pure]> {
  let summary = "Retrieve the size from a stdvec object.";
  let description = [{
    The stdvec_size op can be used to retreive the size from a stdvec object.
    Analogous to a call to `vec.size()`.

    Note that this operation has value semantics. It does not touch memory as
    the size is part of the aggregate value and nothing is dereferenced.
  }];

  let arguments = (ins cc_StdVectorType:$stdvec);
  let results = (outs AnyInteger:$size);

  let assemblyFormat = [{
    $stdvec attr-dict `:` functional-type(operands, results)
  }];
}

def cc_FuncToPtrOp : CCOp<"func_ptr", [Pure]> {
  let summary = "Cast a function to a pointer.";
  let description = [{
    Degenerate a function (such as the result of `func.constant`) to a pointer.
    For now, pointers are expressed in the LLVM-IR dialect.
  }];

  let arguments = (ins FunctionType:$func);
  let results = (outs AnyPointerType:$pointer);

  let assemblyFormat = [{
    $func attr-dict `:` functional-type(operands, results)
  }];
}

def cc_UndefOp : CCOp<"undef", [Pure]> {
  let summary = "Explicit undefined value.";
  let description = [{
    An undefined value of a specified type. This op will be translated to LLVM
    IR as an undefined instruction.
  }];
  let results = (outs AnyType:$inputType);
  let assemblyFormat = "type($inputType) attr-dict";
}

#endif // CUDAQ_OPTIMIZER_DIALECT_CC_OPS
