/********************************************************** -*- tablegen -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#ifndef CUDAQ_OPTIMIZER_DIALECT_CC_OPS
#define CUDAQ_OPTIMIZER_DIALECT_CC_OPS

include "mlir/Interfaces/InferTypeOpInterface.td"
include "cudaq/Optimizer/Dialect/Common/Traits.td"
include "cudaq/Optimizer/Dialect/CC/CCDialect.td"
include "cudaq/Optimizer/Dialect/CC/CCTypes.td"

def AnyPointerType : Type<CPred<"$_self.isa<mlir::LLVM::LLVMPointerType,"
                          "cudaq::cc::PointerType>()">, "any pointer type">;

def AnyCallableType : Type<CPred<"$_self.isa<cudaq::cc::CallableType, "
                          "mlir::FunctionType>()">, "any callable type">;

def AnyAggregateType : Type<CPred<"$_self.isa<cudaq::cc::StructType, "
                            "cudaq::cc::ArrayType>()">, "any aggregate type">;

//===----------------------------------------------------------------------===//
// Base operation definition.
//===----------------------------------------------------------------------===//

class CCOp<string mnemonic, list<Trait> traits = []> :
    Op<CCDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Local Scope.
//===----------------------------------------------------------------------===//

def cc_ScopeOp : CCOp<"scope",
        [AutomaticAllocationScope, RecursiveMemoryEffects, NoRegionArguments,
         DeclareOpInterfaceMethods<RegionBranchOpInterface,
                                   ["getNumRegionInvocations",
                                    "getRegionInvocationBounds"]>]> {
  let summary = "A compound statement in which allocations are scoped.";
  let description = [{
    A ScopeOp is used to delineate the scope of local allocations, such as
    for variables declared in a compound statement. All such allocations that
    appear in the context of the ScopeOp are dead when the ScopeOp exits.

    For a C++ variable, this means that the variable may require a call to a
    destructor as well as deallocation. For example, after inserting calls to
    dtors the LLVM IR intrinsics `llvm.stacksave` and `llvm.stackrestore` may
    be added to reclaim stack space, or a call to `free` may be inserted to
    deallocate any compiler-generated heap allocations.

    For a CUDA Quantum variable, cc.scope will, at minimum add calls to
    deallocate any quantum references created inside the ScopeOp's region. It
    may also be desirable to autogenerate adjoint code.

    A ScopeOp that contains no allocations has no semantics and can be inlined
    into the parent Region. This transformation is done in canonicalization.

    A ScopeOp may contain a single-entry, multiple-exit CFG with multiple
    basic blocks in its region. A ContinueOp is the terminator that exits a
    ScopeOp.

    Example:
    
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

//===----------------------------------------------------------------------===//
// Generalized Loop.
//===----------------------------------------------------------------------===//

def cc_LoopOp : CCOp<"loop",
        [AutomaticAllocationScope, RecursiveMemoryEffects,
         DeclareOpInterfaceMethods<LoopLikeOpInterface>,
         DeclareOpInterfaceMethods<RegionBranchOpInterface>]> {
  let summary = "generalized loop construct";
  let description = [{
    A LoopOp is a generalized C-like loop structure. It can be used to capture
    the semantics of C's `for`, `while`, and `do while` statements. A LoopOp is
    at its core a gated backedge with three distinct phases. Each phase is
    a Region in the LoopOp. A LoopOp has the following structure.

    Example:
    
    ```mlir
      cc.loop while {
        <while-code>
      } do {
        <do-code>
      } step {
        <step-code>
      }
    ```

    - `for` statement:
    
    ```c++
      for (<init-code>; <while-code>; <step-code>) {
        <do-code>
      }
    ```
    
    The above C++ `for` loop (C-style) is lowered to the following ScopeOp and
    LoopOp.
    
    ```mlir
      cc.scope {
        <init-code>
        cc.loop while {
          <while-code>
          cc.condition ...
        } do {
          <do-code>
        } step {
          <step-code>
        }
        cc.continue
      }
    ```

    - `while` statement:

    ```c++
      while (<while-code>) {
        <do-code>
      }
    ```
    
    The above C++ `while` loop is lowered to the following LoopOp. Note that
    the LoopOp's step region is left empty.
    
    ```mlir
      cc.loop while {
        <while-code>
        cc.condition ...
      } do {
        <do-code>
      }
    ```

    - `do while` statement:

    ```c++
      do {
        <do-code>
      } while (<while-code>);
    ```
    
    The above C++ `do while` loop is lowered to the following LoopOp. Once
    again, the LoopOp's step region is left empty. The semantics of this form of
    LoopOp are identical to a C `do while` loop. The body of the loop will be
    executed exactly one time before the control condition is evaluated.
    
    ```mlir
      cc.loop do {
        <do-code>
      } while {
        <while-code>
        cc.condition ...
      }
    ```
  }];

  let arguments = (ins
    Variadic<AnyType>:$initialArgs,
    BoolAttr:$post_condition
  );
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region
    SizedRegion<1>:$whileRegion,
    AnyRegion:$bodyRegion,
    AnyRegion:$stepRegion
  );

  let hasCanonicalizer = 1;
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

    bool hasBreakInBody();
  }];
}

//===----------------------------------------------------------------------===//
// If statement.
//===----------------------------------------------------------------------===//

def cc_IfOp : CCOp<"if",
    [DeclareOpInterfaceMethods<RegionBranchOpInterface,
                               ["getNumRegionInvocations",
                                "getRegionInvocationBounds"]>,
     RecursiveMemoryEffects, NoRegionArguments]> {
  let summary = "if-then-else operation";
  let description = [{
    An IfOp is a C-like if statement that supports single-entry, multiple-exit
    code blocks for both the true (then) and false (else) execution cases. The
    else region may be left empty only if the IfOp does not return any results
    to the parent operation.

    Example:
    
    ```mlir
      cc.if (%cond) {
        ^bb0:
          ...
        ^bb2:
          ...
        ^bb3:
          ...
          cc.continue // exits then region
        ^bb4:
          cf.cond_br ..., ^bb0, ^bb2
      } else {
        ^bb5:
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

//===----------------------------------------------------------------------===//
// Terminators for control-flow operations.
//===----------------------------------------------------------------------===//

def cc_ConditionOp : CCOp<"condition",
        [Pure, Terminator, ParentOneOf<["LoopOp"]>,
         DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface>]> {
  let summary = "Conditional branch in a where region's basic block.";
  let description = [{
    A ConditionOp is used as the terminator of the basic block in the where
    region of a LoopOp. It takes, at minimum, an `i1` value as the control
    condition. If the value is true, a branch to the body of the loop is taken.
    If the value is false, the parent LoopOp exits.

    Any additional arguments to the ConditionOp are forwarded as block
    arguments to the body or the result of the parent LoopOp accordingly.

    Example:

    ```mlir
      cc.loop while {
        ...
        cc.condition %1 (%2, %3 : i32, !quake.wire)
      } do {
       ^bb0(%0 : i32, %1 : !quake.wire):
        ...
      }
    ```
  }];

  let arguments = (ins
    I1:$condition,
    Variadic<AnyType>:$results
  );
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

  let assemblyFormat = [{
    $condition ( `(` $results^ `:` qualified(type($results)) `)` )? attr-dict
  }];
  let hasVerifier = 1;
}

def cc_ContinueOp : CCOp<"continue", [Pure, ReturnLike, Terminator,
        ParentOneOf<["LoopOp", "ScopeOp", "IfOp"]>]> {
  let summary = "Continue branch.";
  let description = [{
    A ContinueOp is a generalized exit terminator for the dialect operations
    with regions.

    In the body region of a loop op, a ContinueOp has the standard C semantics.
    If the LoopOp has a step region, control is first transferred to the entry
    block of the step region and from there to the next iteration of the loop,
    which starts at the while region. If the LoopOp does not have a step
    region, control is transferred to the while region's entry block.

    In the step region of a LoopOp, a ContinueOp is an unconditinal branch from
    the step region to the while region's entry block.

    In the parent op is a ScopeOp or IfOp, a ContinueOp is an unconditional
    branch exiting the parent op and returning control to the parent's parent.

    Example:

    ```mlir
    %0 = cc.scope -> (!quake.wire) {
      ...
      cc.continue %4 : !quake.wire
    }
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

  let assemblyFormat = [{
    ($operands^ `:` qualified(type($operands)))? attr-dict
  }];
}

def cc_BreakOp : CCOp<"break",
        [Pure, ReturnLike, Terminator, ParentOneOf<["LoopOp"]>]> {
  let summary = "Break branch.";
  let description = [{
    A BreakOp can be used in a LoopOp's body region (only). A BreakOp is a
    terminator. Semantically, a BreakOp is an unconditional, immediate branch
    from the body of the parent LoopOp to the parent's parent operation. When
    a BreakOp is reached, no other operations in the parent LoopOp (from any
    region) will be executed. Any arguments to the BreakOp will be forwarded as
    the results of the parent LoopOp.

    Example:

    ```mlir
      %w = cc.loop while ((%arg0 = %0) -> !quake.wire) {
        ...
      } do {
        ...
       ^bb6:
        cc.break %4 : !quake.wire
       ...
     }
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
  let assemblyFormat = [{
    ($operands^ `:` qualified(type($operands)))? attr-dict
  }];
}

def cc_ReturnOp : CCOp<"return", [Pure, ReturnLike, Terminator,
        ParentOneOf<["mlir::func::FuncOp", "CreateLambdaOp"]>]> {
  let summary = "Return (exiting) branch.";
  let description = [{
    A ReturnOp returns control from the current activation of a λ expression to
    the enclosing dynamic scope, the calling function.

    In callable expressions, the ReturnOp is always ends the execution of the
    callable.

    ```mlir
      %lambda = cc.create_lambda {
       ^entry(%arg0 : i32):
        ...
        cc.return
      } : !cc.callable<(i32) -> ()>
      ...
      cc.call_callable %lambda, %20 : (!cc.callable<(i32) -> ()>, i32) -> ()
    ```
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

//===----------------------------------------------------------------------===//
// Global transfers of control.
//===----------------------------------------------------------------------===//

def cc_UnwindBreakOp : CCOp<"unwind_break", [
        ParentOneOf<["IfOp", "ScopeOp"]>, JumpWithUnwind]> {
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
        ParentOneOf<["IfOp", "ScopeOp"]>, JumpWithUnwind]> {
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
        ParentOneOf<["LoopOp", "IfOp", "ScopeOp"]>, JumpWithUnwind]> {
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

//===----------------------------------------------------------------------===//
// Memory operations and initializations.
//===----------------------------------------------------------------------===//

def cc_UndefOp : CCOp<"undef", [Pure]> {
  let summary = "Explicit undefined value.";
  let description = [{
    An UndefOp is used to create an undefined value of a specified type. This
    op will be translated to LLVM IR as an undefined instruction.

    Example:

    ```mlir
      %structValue = cc.undef !cc.struct<{i32, i8}>
    ```
  }];
  let results = (outs AnyType:$inputType);
  let assemblyFormat = [{
    type($inputType) attr-dict
  }];
}

def cc_AllocaOp : CCOp<"alloca", [
    MemoryEffects<[MemAlloc<AutomaticAllocationScopeResource>]>]> {
  let summary = "Allocate a dynamic block of memory on the stack.";
  let description = [{
    The AllocaOp is similar to the C library `alloca()` function. The number of
    bytes to be allocated is computed by the size of the `elementType` (which
    may not have a non-positive size) times `seqSize` if present or `1` if
    `seqSize` is omitted. This returns a pointer to the first element in the
    allocated sequence of `seqSize` elements of `elementType`.
  }];

  let arguments = (ins
    TypeAttr:$elementType,
    Optional<AnySignlessInteger>:$seqSize
  );
  let results = (outs cc_PointerType:$address);

  let hasFolder = 1;
  let hasCustomAssemblyFormat = 1;
  let builders = [
    OpBuilder<(ins "mlir::Type":$elementType, "mlir::Value":$seqSize), [{
      auto resTy = cudaq::cc::PointerType::get(seqSize ?
        cudaq::cc::ArrayType::get(elementType) : elementType);
      return build($_builder, $_state, resTy, elementType, seqSize);
    }]>,
    OpBuilder<(ins "mlir::Type":$elementType), [{
      return build($_builder, $_state, elementType, mlir::Value{});
    }]>
  ];
}

def cc_LoadOp : CCOp<"load",
    [TypesMatchWith<"result type matches element type of pointer value",
        "ptrvalue", "result",
        "$_self.cast<cudaq::cc::PointerType>().getElementType()">]> {
  let summary = "Load a value from a pointer into a virtual register.";
  let description = [{
    A LoadOp is used to load a value from a memory location, specified by a
    pointer, into a virtual register.

    Example:

    ```mlir
      %1 = cc.load %0 : !cc.ptr<i32>
    ```
  }];

  let arguments = (ins
    Arg<cc_PointerType,"pointer to memory location",[MemRead]>:$ptrvalue
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $ptrvalue `:` qualified(type($ptrvalue)) attr-dict
  }];
}

def cc_StoreOp : CCOp<"store",
    [TypesMatchWith<"type of value matches element type of pointer",
        "ptrvalue", "value",
        "$_self.cast<cudaq::cc::PointerType>().getElementType()">]> {
  let summary = "Store a virtual register value to a memory location.";
  let description = [{
    A StoreOp is used to store the value from a virtual register to a memory
    location, specified by a pointer.

    Example:

    ```mlir
      cc.store %value, %pointer : !cc.ptr<f64> {cpp_variable="my_double"}
    ```
  }];

  let arguments = (ins
    AnyType:$value,
    Arg<cc_PointerType,"pointer to memory location",[MemWrite]>:$ptrvalue
  );
  let assemblyFormat = [{
    $value `,` $ptrvalue `:` qualified(type($ptrvalue)) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Aggregate operations.
//===----------------------------------------------------------------------===//


def cc_AddressOfOp : CCOp<"address_of", [Pure,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  // Note: should be used to get a pointer to a global object as well, but we
  // haven't added globals to the dialect yet.

  let summary = "Creates a pointer pointing to a function";
  let description = [{
    An AddressOfOp is used to get the address of a function by its symbol name.

    Example:

    ```mlir
      %fp = cc.address_of @my_void_function : !cc.ptr<() -> ()>
    ```
  }];
  
  let arguments = (ins FlatSymbolRefAttr:$global_name);
  let results = (outs cc_PointerType:$res);

  let assemblyFormat = [{
    $global_name `:` qualified(type($res)) attr-dict
  }];
}

def cc_ExtractValueOp : CCOp<"extract_value", [Pure]> {
  let summary = "Extract a value from an aggregate value.";
  let description = [{
    This will translate to LLVM's extract_value instruction. It makes a copy of
    a subvalue that is part of a larger aggregate value.

    Example:

    ```mlir
      %i = cc.extract_value %r [1, 0] : (!cc.struct<{i32, !cc.struct<{i16, f32,
                                        i64}>}>) -> i16
    ```
  }];

  let arguments = (ins
    AnyAggregateType:$container,
    DenseI64ArrayAttr:$position
  );
  let results = (outs AnyType);

  let assemblyFormat = [{
    $container `` $position `:` functional-type(operands, results) attr-dict
  }];
}

def cc_InsertValueOp : CCOp<"insert_value", [Pure]> {
  let summary = "Insert a value into an aggregate value.";
  let description = [{
    This will translate to LLVM's insert_value instruction. It copies an
    aggregate value and replaces one particular subvalue with a new value.

    Example:

    ```mlir
      %s = cc.insert_value %i16, %r[1, 0] :
           (!cc.struct<{i32, !cc.struct<{i16, f32}>}>, i16) ->
           !cc.struct<{i32, !cc.struct<{i16, f32}>}>
    ```
  }];

  let arguments = (ins
    AnyAggregateType:$container,
    AnyType:$value,
    DenseI64ArrayAttr:$position
  );
  let results = (outs AnyAggregateType);

  let assemblyFormat = [{
    $value `,` $container `` $position `:` functional-type(operands, results)
      attr-dict
  }];
}

def cc_ComputePtrOp : CCOp<"compute_ptr", [Pure]> {
  let summary = "Creates a pointer pointing to a subobject in an object.";
  let description = [{
    This translates to LLVM's get_element_ptr operation. The initial thinking
    is that unlike the GEP, the base object must be either an array or a single
    object and a ComputerPtrOp does not accept a pointer as both implicitly.

    Example:

    ```mlir
      %p = cc.compute_ptr %struct, 1 : (!cc.ptr<!cc.struct<{i32, f32}>>) ->
                                       !cc.ptr<f32>
      %q = cc.compute_ptr %array, %i : (!cc.ptr<!cc.array<i32 x 100>>, i64) ->
                                       !cc.ptr<i32>
    ```
  }];

  let arguments = (ins
    cc_PointerType:$base,
    Variadic<AnyInteger>:$dynamicIndices,
    DenseI32ArrayAttr:$rawConstantIndices
  );
  let results = (outs cc_PointerType);

  let assemblyFormat = [{
    $base `[` custom<ComputePtrIndices>($dynamicIndices, $rawConstantIndices)
      `]` `:` functional-type(operands, results) attr-dict
  }];

  let hasFolder = 1;
  let hasCanonicalizer = 1;

  let builders = [
    OpBuilder<(ins "mlir::Type":$resultType, "mlir::Value":$basePtr,
               "mlir::ValueRange":$indices,
               CArg<"mlir::ArrayRef<mlir::NamedAttribute>", "{}">:$attributes)>,
    OpBuilder<(ins "mlir::Type":$resultType, "mlir::Value":$basePtr,
               "mlir::ArrayRef<cudaq::cc::ComputePtrArg>":$indices,
               CArg<"mlir::ArrayRef<mlir::NamedAttribute>", "{}">:$attributes)> 
  ];
  let extraClassDeclaration = [{
    static constexpr std::int32_t kDynamicIndex =
      std::numeric_limits<std::int32_t>::min();
  }];
}

def cc_ConstantArrayOp : CCOp<"const_array", [Pure]> {
  let summary = "A one dimensional array of constant values.";
  let description = [{
    A ConstantArrayOp is used to create a constant value that is an aggregate of
    type !cc.array<T,n> and is composed of constant values. The aggregate has a
    constant length.

    Example:

    ```mlir
      %arrCon = cc.const_array [1.0, 2.0] : !cc.array<f64 x 2>
    ```
  }];
  let arguments = (ins
    ArrayAttr:$constantValues
  );
  let results = (outs cc_ArrayType:$arrayType);
  let assemblyFormat = [{
    $constantValues `:` qualified(type($arrayType)) attr-dict
  }];
}

def cc_GetConstantElementOp : CCOp<"get_const_element", [Pure]> {
  let summary = "Get a constant value from a constant array.";
  let description = [{
    A GetConstantElementOp is used to extract a constant value from an aggregate
    of type !cc.array<T,n>. The aggregate must be built by a `cc.const_array`
    operation. The offset of the element must be between 0 and the length of the
    array minus 1 (inclusive). Any other value has undefined behavior. If the
    offset is a constant, this operation can be folded into a scalar constant
    (or a cc.undef operation).

    Example:

    ```mlir
      %arrCon = cc.get_const_element %4[%5] : (!cc.array<f64 x 2>, i32) -> f64
    ```
  }];

  let arguments = (ins
    cc_ArrayType:$constantArray,
    AnySignlessInteger:$offset
  );
  let results = (outs AnyType);

  let hasFolder = 1;
  //let hasVerifier = 1; // FIXME: output type must match incoming const array

  let assemblyFormat = [{
    $constantArray `,` $offset `:` functional-type(operands, results) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Cast operation.
//===----------------------------------------------------------------------===//

def cc_CastOp : CCOp<"cast", [Pure]> {
  let summary = "Overloads LLVM's casting instructions.";
  let description = [{
    A CastOp can be used to create one of LLVM's cast instructions. The signed
    (unsigned) keyword can/must be used with otherwise sign agnostic integer
    conversions to distinguish between sext (zext).

    Example:

    ```mlir
      %11 = cc.cast %10 : (!cc.ptr<i8>) -> i64           // ptrtoint
      %12 = cc.cast %11 : (i64) -> !cc.ptr<i32>          // inttoptr
      %13 = cc.cast %12 : (!cc.ptr<i32>) -> !cc.ptr<f64> // bitcast
      %14 = cc.cast %11 : (i64) -> f64                   // bitcast
      %15 = cc.cast signed %11 : (i64) -> f64            // sitofp
      %16 = cc.cast unsigned %11 : (i64) -> f64          // uitofp
      %17 = cc.cast signed %15 : (f64) -> i32            // fptosi
      %18 = cc.cast unsigned %16 : (f64) -> i16          // fptoui
      %21 = cc.cast signed %20 : (i8) -> i32             // sext
      %22 = cc.cast unsigned %20 : (i8) -> i32           // zext
      %23 = cc.cast %22 : (i32) -> i16                   // trunc
      %24 = cc.cast %16 : (f64) -> f32                   // fptrunc
      %25 = cc.cast %24 : (f32) -> f64                   // fpext
    ```
  }];

  let arguments = (ins
    AnyType:$value,
    OptionalAttr<UnitAttr>:$sint,
    OptionalAttr<UnitAttr>:$zint
  );
  let results = (outs AnyType:$result);
  let hasFolder = 1;
  let hasVerifier = 1;
  
  let assemblyFormat = [{
    ( `signed` $sint^ ):( ( `unsigned` $zint^ )? )? $value `:`
      functional-type(operands, results) attr-dict
  }];

  let builders = [
    OpBuilder<(ins "mlir::Type":$resTy, "mlir::Value":$value), [{
      return build($_builder, $_state, resTy, value, {}, {});
    }]>,
    OpBuilder<(ins "mlir::Type":$resTy, "mlir::Value":$value,
        "CastOpMode":$mode), [{
      auto unitAttr = mlir::UnitAttr::get($_builder.getContext());
      if (mode == CastOpMode::Signed)
        return build($_builder, $_state, resTy, value, /*signed=*/unitAttr, {});
      return build($_builder, $_state, resTy, value, {}, /*unsigned=*/unitAttr);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// Stdvec operations.
//===----------------------------------------------------------------------===//

def cc_StdvecInitOp : CCOp<"stdvec_init", [Pure]> {
  let summary = "Initialize a stdvec object from a pointer and length.";
  let description = [{
    A StdvecInitOp can be used to create a "high-level" `std::vector` object in
    cc. The construction is similar to the `std::vector` initializer list
    constructor. It takes two arguments: a pointer to a memory buffer and the
    length of the buffer in terms of the number of elements.

    Note this op has value semantics and does not touch memory. It is used to
    create an abstract aggregate value. No data is moved or copied.

    Example:

    ```mlir
    %buff = ... : !cc.ptr<f64>
    %len = ... : i64
    %svec = cc.stdvec_init %buff, %len : (!cc.ptr<f64>, i64) -> !cc.stdvec<f64>
    func.call @kernel(%svec) ...
    ```
  }];

  let arguments = (ins
    AnyPointerType:$buffer,
    AnyInteger:$length
  );
  let results = (outs cc_StdVectorType:$stdvec);

  let assemblyFormat = [{
    $buffer `,` $length `:` functional-type(operands, results) attr-dict
  }];
}

def cc_StdvecDataOp : CCOp<"stdvec_data", [Pure]> {
  let summary = "Retrieve the data pointer from a stdvec object.";
  let description = [{
    A StdvecDataOp can be used to retreive the data pointer from a stdvec
    object. (See StdvecInitOp.) This operation is analogous to a call to
    `std::vector<T>::data()`.

    Note that this operation has value semantics. It does not touch memory as
    the data pointer is part of the aggregate value and nothing is dereferenced.
  }];

  let arguments = (ins cc_StdVectorType:$stdvec);
  let results = (outs AnyPointerType:$data);

  let hasCanonicalizer = 1;

  let assemblyFormat = [{
    $stdvec `:` functional-type(operands, results) attr-dict
  }];
}

def cc_StdvecSizeOp : CCOp<"stdvec_size", [Pure]> {
  let summary = "Retrieve the size from a stdvec object.";
  let description = [{
    A StdvecSizeOp can be used to retreive the size from a stdvec object.
    This is analogous to a call to `std::vector<T>::size()`.

    Note that this operation is pure and has value semantics. It does not
    modify memory as the size is part of a stdvec's aggregate value and nothing
    is dereferenced.
  }];

  let arguments = (ins cc_StdVectorType:$stdvec);
  let results = (outs AnyInteger:$size);

  let assemblyFormat = [{
    $stdvec `:` functional-type(operands, results) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Type conversions.
//===----------------------------------------------------------------------===//

def cc_FuncToPtrOp : CCOp<"func_ptr", [Pure]> {
  let summary = "Cast a function to a pointer.";
  let description = [{
    A FuncToPtrOp is used to degenerate a function value (such as the result of
    a `func.constant` operation) to a pointer.

    Example:

    ```mlir
      %func = func.constant @foo : () -> ()
      %ptr = cc.func_ptr %func : (() -> ()) -> !cc.ptr<i8>
    ```
  }];

  let arguments = (ins FunctionType:$func);
  let results = (outs AnyPointerType:$pointer);

  let assemblyFormat = [{
    $func `:` functional-type(operands, results) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Lambda (λ) expressions.
//===----------------------------------------------------------------------===//

def cc_CreateLambdaOp : CCOp<"create_lambda",
        [AutomaticAllocationScope, RecursiveMemoryEffects,
         SingleBlockImplicitTerminator<"cudaq::cc::ReturnOp">]> {
  let summary = "Create a lambda expression as an abstract callable.";
  let description = [{
    A CreateLambdaOp creates a λ expression. A λ expression is a Callable
    instance that may have associated data (or data references) that it
    captures. Currently, captured data is implied by using values from the
    parent operation's regions in the body of the CreateLambdaOp.

    Example:

    ```mlir
      %lambda = cc.create_lambda {
       ^entry(%arg0 : i32):
        ...
        cc.return
      } : !cc.callable<(i32) -> ()>
      ...
      cc.call_callable %lambda, %20 : (!cc.callable<(i32) -> ()>, i32) -> ()
    ```
  }];

  let results = (outs cc_CallableType:$signature);
  let regions = (region SizedRegion<1>:$initRegion);
  let hasCustomAssemblyFormat = 1;

  let builders = [
      OpBuilder<(ins
        "cudaq::cc::CallableType":$signature,
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

    mlir::Region &getBody() { return getInitRegion(); }
    llvm::StringRef getName() { return "*lambda*"; }
  }];
}

def cc_CallCallableOp : CCOp<"call_callable", [CallOpInterface]> {
  let summary = "Abstract application of a callable object with arguments.";
  let description = [{
    A CallCallableOp results in the transfer of control to a C++ callable (a
    closure). The call target is specified as an SSA-value as the first
    argument position. In the CC dialect, a callable is an object (not a
    function pointer/symbol).

    A plain old function may be used to construct a callable object. In this
    case the callable object will contain a pointer to the function and no
    additional data. The argument list to the CallCallableOp must be a type
    correct one-to-one match with the function's arguments. The compiler may
    recognize that the closure of the calling context is empty and can be
    elided.

    A C++ class can be used to construct a callable object. In this case, the
    closure can be thought of as containing a pointer to the function call
    (`operator()`) and a `this` pointer to the captured data. In this case, all
    values captured in the closure are data members of the instance.

    For a λ expression callable, the call target will involve creating an object
    at the instantiation site (InstantiateCallableOp). This object will capture
    a pointer to a trampoline function and construct a tuple for all captured
    variables. The compiler-generated trampoline will know how to unpack the
    tuple and call the ultimate outlined lambda function with the unpacked
    captured variables passed as individual arguments.

    Example:

    ```mlir
      %lambda = cc.create_lambda {
       ^entry(%arg0 : i32):
        ...
        cc.return
      } : !cc.callable<(i32) -> ()>
      ...
      cc.call_callable %lambda, %20 : (!cc.callable<(i32) -> ()>, i32) -> ()
    ```
  }];

  let arguments = (ins
    AnyCallableType:$callee,
    Variadic<AnyType>:$args
  );
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;

  let assemblyFormat = [{
    $callee (`,` $args^)? `:` functional-type(operands, results) attr-dict
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

def cc_InstantiateCallableOp : CCOp<"instantiate_callable", [Pure]> {
  let summary = "Construction of a callable object.";
  let description = [{
    An InstantiateCallableOp is created by the compiler when lifting a λ
    expression, i.e. a CreateLambdaOp. The CreateLambdaOp is replaced at its
    definition site by an InstantiateCallableOp. It is also replaced by a pair
    of functions, a trampoline and the code from the CreateLambdaOp's body.

    The InstantiateCallableOp constructs a pair. The pair is a pointer to the
    associated currying trampoline function and a tuple of the captured values.

    When there are no captured values, such as when the callable is a plain old
    function, the attribute `nocapture` should be used to build the callable
    such that no unmarshalling trampoline is required. See below.

    Example:

    Take the following λ expression.

    ```mlir
      %88 = ... : f64
      ...
      %lambda = cc.create_lambda {
       ^entry(%arg0 : i32):
        ... %88 ...
        cc.return
      } : !cc.callable<(i32) -> ()>
    ```

    Since this λ expression captures the value %88 from the parent, we have to
    instantiate a closure that includes this %88 value when we lift the λ.
    The lifted λ is given the name `@lifted_lambda.54` in the following. A
    second function, `@stub.54`, unpacks the closure into arguments to the new
    outlined λ function in the following example.

    ```mlir
      %lambda = cc.instantiate_callable @trampoline.54(%88) : (f64) -> !cc.callable<(i32) -> ()>
      ...

      // Trampoline function
      func.func private @trampoline.54(%arg0: !cc.callable<(i32) -> ()>, %arg1: i32) {
        %0 = cc.callable_closure %arg0: !cc.callable<(i32) -> ()> -> f64
        call @lifted_lambda.54(%0, %arg1) : (f64, i32) -> ()
      }

      // Outlined body of original lambda
      func.func private @lifted_lambda.54(%88: f64, %arg0: i32) {
       ^entry:
        ... %88 ...
        cc.return
      }
    ```

    In the case when the `nocapture` attribute is present, the closure pointer
    value will be set to `nullptr`. A `cc.call_callable` can thus test the
    closure value to determine whether to omit prepending the pointer to the
    callable pair or not.

    See also CallCallableOp.
  }];

  let arguments = (ins
    SymbolRefAttr:$callee,
    Variadic<AnyType>:$closure_data,
    OptionalAttr<UnitAttr>:$no_capture
  );
  let results = (outs cc_CallableType:$signature);

  let builders = [
    OpBuilder<(ins "mlir::Type":$signature, "mlir::SymbolRefAttr":$callee,
      "mlir::ValueRange":$closure_data), [{
      return build($_builder, $_state, signature, callee, closure_data,
        mlir::UnitAttr{});
    }]> 
  ];

  let assemblyFormat = [{
    $callee `(` $closure_data `)` ( `nocapture` $no_capture^ )?
      `:` functional-type(operands, results) attr-dict
  }];
}

def cc_CallableFuncOp : CCOp<"callable_func", [Pure]> {
  let summary = "Extract the function from a callable object.";
  let description = [{
    A callable object contains both code (a function) and data (a closure). A
    CallableFuncOp can be used to retrieve the function (pointer) from the
    callable object.

    Example:

    ```mlir
      %0 = ... : !cc.callable<(i32) -> i8>
      %1 = ... : i32
      %2 = cc.callable_func %0 : (!cc.callable<(i32) -> i8>) -> ((!cc.callable<(i32) -> i8>, i32) -> i8)
      %3 = call_indirect %2(%0, %1) : (!cc.callable<(i32) -> i8>, i32) -> i8
    ```

    See also InstantiateCallableOp, CallableClosureOp.
  }];

  let arguments = (ins cc_CallableType:$callable);
  let results = (outs FunctionType:$function);

  let assemblyFormat = [{
    $callable `:` functional-type(operands, results) attr-dict
  }];
}

def cc_CallableClosureOp : CCOp<"callable_closure", [Pure]> {
  let summary = "Extract the closure data from a callable object.";
  let description = [{
    A callable object contains both code (a function) and data (a closure). The
    CallableClosureOp can be used to retrieve the closure (data) from the
    callable object.

    For a callable instantiated from a λ expression, the closure is unpacked
    and returned as transparently as a set of values. This is possible because
    a closure has a compile-time constant number of values and cannot be
    extended dynamically.

    For a callable instantiated from a C++ class, the closure is the `this`
    pointer to the object instance.

    For a callable instantiated from a plain old function, the closure is empty
    and contains no data.

    Example:

    ```mlir
      %0 = ... !cc.callable<(i32) -> i8>
      %1 = ... : i32
      %2:2 = cc.callable_closure %0 : (!cc.callable<(i32) -> i8>) -> (!quake.wire, i1)
      %3 = call @lifted_lambda.72(%2#0, %2#1, %1) : (!quake.wire, i1, i32) -> i8
    ```

    See also CallableFuncOp, InstantiateCallableOp.
  }];

  let arguments = (ins cc_CallableType:$callable);
  let results = (outs Variadic<AnyType>:$closure_data);

  let assemblyFormat = [{
    $callable `:` functional-type(operands, results) attr-dict
  }];
}

def cc_CreateStringLiteralOp : CCOp<"string_literal"> {
  let summary = "Create a constant string literal.";
  let description = [{
    This operation creates a ASCIIZ string literal value. It's argument is a
    constant MLIR String Attribute. The literal will have a null character
    appended automatically.

    ```mlir
      %0 = cc.string_literal "Quantum Computing" : !cc.ptr<!cc.array<i8 x 18>>
    ```
  }];

  let arguments = (ins StrAttr:$stringLiteral);
  let results = (outs cc_PointerType:$result);
  let assemblyFormat = [{
     $stringLiteral `:` qualified(type(results)) attr-dict
  }];
}
#endif // CUDAQ_OPTIMIZER_DIALECT_CC_OPS
