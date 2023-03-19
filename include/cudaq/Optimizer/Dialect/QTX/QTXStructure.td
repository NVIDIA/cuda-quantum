/********************************************************** -*- tablegen -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#ifndef CUDAQ_OPTIMIZER_DIALECT_QTX_QTX_STRUCTURE
#define CUDAQ_OPTIMIZER_DIALECT_QTX_QTX_STRUCTURE
include "QTXTypes.td"

include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// QTX quantum circuit
//===----------------------------------------------------------------------===//

def qtx_CircuitOp : QTXOp<"circuit", [
    HasParent<"mlir::ModuleOp">, IsolatedFromAbove, OpAsmOpInterface, Symbol,
    CallableOpInterface, AutomaticAllocationScope
  ]> {
  let summary = "An operation with a name containing a single `SSACFG` region";
  let description = [{
    The `qtx.circuit` operation represents a quantum circuit.  The signature
    of a circuit operation has the following components:

      * A symbol `name` that specifies the MLIR name for the circuit.
      * An optional list of classical arguments.
      * An optional list of control qubits.
      * An optional list of target qubits.
      * A circuit body

    Operations within the circuit cannot implicitly capture values defined
    outside of it, i.e. a circuit is `IsolatedFromAbove`.  All external
    references must use circuit arguments or attributes that establish a
    symbolic connection (e.g. symbols referenced by name via a string
    attribute like SymbolRefAttr).
  }];
  let arguments = (ins OptionalAttr<StrAttr>:$sym_visibility,
                       SymbolNameAttr:$sym_name,
                       TypeAttrOf<OperatorType>:$operator_type);
  let regions = (region AnyRegion:$body);
  let builders = [
    OpBuilder<(ins "mlir::StringAttr":$sym_name)>,

    OpBuilder<(ins "mlir::StringAttr":$sym_visibility,
                   "mlir::StringAttr":$sym_name,
                   "mlir::ValueRange":$classical_args,
                   "mlir::ValueRange":$quantum_args,
                   "mlir::TypeRange":$classical_return_types)>
  ];
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // Attribute names
    //===------------------------------------------------------------------===//

    constexpr static mlir::StringRef getArgSegmentSizesAttrName() {
      return "argument_segment_sizes";
    }

    //===------------------------------------------------------------------===//
    // Name
    //===------------------------------------------------------------------===//

    mlir::StringRef getName() {
      return mlir::SymbolTable::getSymbolName(*this);
    }

    //===------------------------------------------------------------------===//
    // CallableOpInterface
    //===------------------------------------------------------------------===//

    /// Returns the region on the current operation that is callable. This may
    /// return null in the case of an external callable object, e.g. an external
    /// function.
    mlir::Region *getCallableRegion() {
      return &getBody();
    }

    /// Returns the results types that the callable region produces when
    /// executed.
    mlir::ArrayRef<mlir::Type> getCallableResults() {
      return getOperatorType().getResults();
    }

    //===------------------------------------------------------------------===//
    // Type Handling
    //===------------------------------------------------------------------===//

    mlir::TypeAttr _getOperatorTypeAttr() {
      return this->getOperation()->template getAttrOfType<mlir::TypeAttr>(
          "operator_type");
    }

    OperatorType _getOperatorType() {
      return getOperatorTypeAttr().getValue().cast<OperatorType>();
    }

    //===------------------------------------------------------------------===//
    // Body Handling
    //===------------------------------------------------------------------===//

    mlir::Region &_getBody() { return getRegion(); }

    mlir::OpBuilder getBodyBuilder() {
      assert(!getBody().empty() && "Unexpected empty 'body' region.");
      mlir::Block &bodyBlock = getBody().front();
      return mlir::OpBuilder::atBlockEnd(&bodyBlock);
    }

    //===------------------------------------------------------------------===//
    // Argument and Result Handling
    //===------------------------------------------------------------------===//

    // Block argument iterator types.
    using BlockArgListType = ::mlir::Region::BlockArgListType;
    using args_iterator = BlockArgListType::iterator;

    // Support argument iteration.
    args_iterator args_begin() { return getBody().args_begin(); }
    args_iterator args_end() { return getBody().args_end(); }

    std::pair<unsigned, unsigned> getArgumentIndexAndLength(unsigned index) {
      using namespace ::mlir::impl;
      auto sizeAttr = getAttrFromSortedRange((*this)->getAttrs().begin(),
                                             (*this)->getAttrs().end(),
                                             "argument_segment_sizes")
                      .cast<::mlir::DenseI32ArrayAttr>();

      unsigned start = 0;
      for (unsigned i = 0; i < index; ++i)
        start += sizeAttr[i];
      return {start, sizeAttr[index]};
    }

    BlockArgListType getArguments() { return getBody().getArguments(); }

    unsigned getNumParameters() { return getArgumentIndexAndLength(0).second; }

    unsigned getNumTargets() { return getArgumentIndexAndLength(1).second; }

    BlockArgListType getParameters() {
      return BlockArgListType(getArguments().data(), getNumParameters());
    }

    BlockArgListType getTargets() {
      return BlockArgListType(getArguments().data() + getNumParameters(),
                              getNumTargets());
    }

    mlir::ArrayRef<mlir::Type> getClassicalResultTypes() {
      return getOperatorType().getClassicalResults();
    }

    mlir::ArrayRef<mlir::Type> getTargetTypes() {
      return getOperatorType().getTargets();
    }

    //===------------------------------------------------------------------===//
    // OpAsmOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Allow the dialect prefix to be omitted.
    static llvm::StringRef getDefaultDialect() { return "qtx"; }
  }];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

def qtx_ReturnOp : QTXOp<"return", [
    AttrSizedOperandSegments, HasParent<"CircuitOp">, Pure, ReturnLike,
    Terminator
  ]> {
  let summary = "circuit return operation";
  let description = [{
    The `qtx.return` operation represents a return operation within a circuit.
    The operation takes two variable number of operands and produces no results.

    The operands are divided into classical and quantum (also known as targets).
    Classical operands are delimited by `<` and `>`.  The number of operands and
    their types must match the signature of the circuit that contains the
    operation.

    NOTE: A circuit might return an array of wires.  In such cases, the array
    must not contain dead wires, i.e., have borrowed wires that were not yield
    back to the array (for more information about dead wires refer to the
    description of `qtx.array_borrow`).

    Example:

    ```mlir
    // Returns a wire and an array
    return %wire, %array : !qtx.wire, !qtx.wire_array<5>

    // Rertuns a classical result.
    return <%measurement_result> : <i32>

    // Returns classical and quantum results.
    return <%measurement_result> %wire : <i32> !qtx.wire
    ```
  }];
  let arguments = (ins
    Variadic<AnyType>:$classical,
    Variadic<AnyQType>:$targets
  );
  // We need custom assembly format because we can have just `return` (without
  // any operands)---a case in which we need to omit the `:`.
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def qtx_UnrealizedReturnOp : QTXOp<"unrealized_return", [
    AttrSizedOperandSegments, Pure
  ]> {
  let summary = "unrealized return operation";
  let description = [{
    This operation does not have execution semantics and is only intended to aid
    the two-step conversion from quantum kernels to QTX's circuit.

    In the first step, the conversion will convert the kernel region
    (`mlir::FuncOp` body), and insert this instruction before all `func.return`
    operations.  This instruction is used to save the live values corresponding
    to input quantum references that the QTX circuit will return.  It does this
    to unburden the second step from analyzing the function body again.
  }];
  let arguments = (ins
    Variadic<AnyType>:$classical,
    Variadic<AnyQType>:$targets
  );
  // We need custom assembly format because we can have just `return` (without
  // any operands)---a case in which we need to omit the `:`.
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// ApplyOp
//===----------------------------------------------------------------------===//

def qtx_ApplyOp : QTXOp<"apply", [
    AttrSizedOperandSegments, AttrSizedResultSegments, CallOpInterface,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>,
    AllMatch<["getTargets().getType()", "getNewTargets().getType()"],
              "target input types match new target types (results)">
  ]> {
  let summary = "apply a circuit";
  let description = [{
    The `qtx.apply` operation represents calling a circuit that is within the
    same symbol scope, i.e., `ModuleOp` (in our case).  The operands and result
    types of the operation must match the specified circuit type.  The circuit
    is encoded as a symbol reference attribute named "callee".

    Example:
    ```mlir
    %w0 = alloca : !qtx.wire
    %a0 = alloca : !qtx.wire_array<2>
    %w1, %a1 = apply @foo(%w0, %a0) : (!qtx.wire, !qtx.wire_array<2>)
        -> (!qtx.wire, !qtx.wire_array<2>)
    ```
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    UnitAttr:$is_adj,
    Variadic<WireType>:$controls,
    Variadic<AnyType>:$parameters,
    Variadic<AnyQType>:$targets
  );
  let results = (outs
    Variadic<AnyType>:$classical_results,
    Variadic<AnyQType>:$new_targets
  );
  let assemblyFormat = [{
    (```<` `adj` $is_adj^ `>`)?  (`[` $controls^ `]`)?
    $callee  (```<` $parameters^ `>```)? (```(` $targets^ `)```)? attr-dict
    `:` custom<OperatorType>(type($parameters), type($targets),
                             type($classical_results), type($new_targets))
  }];
  let extraClassDeclaration = [{
    static constexpr llvm::StringRef getCalleeAttrNameStr() { return "callee"; }

    OperatorType getCalleeType();

    unsigned getNumParameters() { return getParameters().size(); }
    unsigned getNumControls() { return getControls().size(); }
    unsigned getNumTargets() { return getTargets().size(); }
    unsigned getNumClassicalResults() { return getClassicalResults().size(); }

    mlir::Value getParameter(unsigned i) { return getParameters()[i]; }
    mlir::Value getControl(unsigned i) { return getControls()[i]; }
    mlir::Value getTarget(unsigned i) { return getTargets()[i]; }
    mlir::Value getClassicalResult(unsigned i) {
      return getClassicalResults()[i];
    }
    mlir::Value getNewTarget(unsigned i) { return getNewTargets()[i]; }

    using operand_iterator = mlir::Operation::operand_iterator;
    using operand_range = mlir::Operation::operand_range;

    operand_iterator arg_operand_begin() {
      return getOperation()->operand_begin();
    }
    operand_iterator arg_operand_end() {
      return getOperation()->operand_end();
    }

    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    mlir::CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<mlir::SymbolRefAttr>(getCalleeAttrName());
    }
  }];
}

#endif // CUDAQ_OPTIMIZER_DIALECT_QTX_QTX_STRUCTURE
