/********************************************************** -*- tablegen -*- ***
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#ifndef CUDAQ_OPTIMIZER_DIALECT_QUAKE_IR_INTERFACES
#define CUDAQ_OPTIMIZER_DIALECT_QUAKE_IR_INTERFACES

include "mlir/IR/OpBase.td"

def OperatorInterface : OpInterface<"OperatorInterface"> {
  let description = [{
    Conceptually, a `quantum operator` is an effect that _might_ modify the
    state of a subset of wires.  Most often, this effect is unitary evolution
    ---in this case, we say that the operator is a unitary.

    Note that every quantum operation (instruction) can be seen as a operator.
    There are some passes that simply don't need to know which specific
    operation they are dealing with,  i.e., they only care how this operation is
    connected to others through wires.  Hence, this interface provides a way of
    getting this information without the need of casting an operation to its
    actual type.
  }];
  let cppNamespace = "cudaq::quake";

  let methods = [
    InterfaceMethod<
      /*desc=*/ "Returns whether this is the adjoint of an operator",
      /*retType=*/ "bool",
      /*methodName=*/ "isAdj",
      /*args=*/ (ins),
      /*methodBody=*/ "return $_op.getIsAdj();"
    >,
    InterfaceMethod<
      /*desc=*/ "Returns the classical parameters of this operator",
      /*retType=*/ "mlir::Operation::operand_range",
      /*methodName=*/ "getParameters",
      /*args=*/ (ins),
      /*methodBody=*/ "return $_op.getParameters();"
    >,
    InterfaceMethod<
      /*desc=*/ "Returns the controls of this operator",
      /*retType=*/ "mlir::Operation::operand_range",
      /*methodName=*/ "getControls",
      /*args=*/ (ins),
      /*methodBody=*/ "return $_op.getControls();"
    >,
    InterfaceMethod<
      /*desc=*/ "Returns the set of qubits that were operated upon",
      /*retType=*/ "mlir::ValueRange",
      /*methodName=*/ "getWires",
      /*args=*/ (ins),
      /*methodBody=*/ "return $_op.getWires();"
    >,
    InterfaceMethod<
      /*desc=*/ "Returns an array indicating whether controls are negated",
      /*retType=*/ "std::optional<mlir::ArrayRef<bool>>",
      /*methodName=*/ "getNegatedControls",
      /*args=*/ (ins),
      /*methodBody=*/ "return $_op.getNegatedQubitControls();"
    >,
    InterfaceMethod<
      /*desc=*/ "Returns the targets (input) of this operator",
      /*retType=*/ "mlir::Operation::operand_range",
      /*methodName=*/ "getTargets",
      /*args=*/ (ins),
      /*methodBody=*/ "return $_op.getTargets();"
    >,
    InterfaceMethod<
      /*desc=*/ "Return a paticular target (input) of this operator",
      /*retType=*/ "mlir::Value",
      /*methodName=*/ "getTarget",
      /*args=*/ (ins "unsigned":$index),
      /*methodBody=*/ "return $_op.getTargets()[index];"
    >,
    InterfaceMethod<
      /*desc=*/ [{ Set `matrix` to the operator's unitary matrix as a
                   column-major array if it can be computed at compilation time,
                   i.e., if the parameter is a compile-time constant.}],
      /*retType=*/ "void",
      /*methodName=*/ "getOperatorMatrix",
      /*args=*/ (ins "mlir::SmallVectorImpl<std::complex<double>>&":$matrix),
      /*methodBody=*/ "return $_op.getOperatorMatrix(matrix);"
    >,
  ];
}

def MeasurementInterface : OpInterface<"MeasurementInterface"> {
  let description = [{
    A measurement operation takes a qubit (qudit) as input, measures it, and
    returns a classical value, resp. a bit (bitstring). A measurement "destroys"
    the value of the qubit in the sense the information that the qubit encoded
    in its state is not recoverable.

    A measurement either measures each of its targets independently, giving
    one classical result per target (e.g., `mz`), or it measures all of its
    targets together as a single joint observable, giving exactly one classical
    result no matter how many targets there are (e.g., `mpp`). Code that
    assumes the first form, such as code that pairs each target with a result
    or that treats each target as an independent single-qubit measurement, must
    consult `isJointMeasurement` before treating a measurement generically.
  }];
  let cppNamespace = "cudaq::quake";

  let methods = [
    InterfaceMethod<
      /*desc=*/ "Returns the set of qubits to be measured",
      /*retType=*/ "mlir::Operation::operand_range",
      /*methodName=*/ "getTargets",
      /*args=*/ (ins),
      /*methodBody=*/ "return $_op.getTargets();"
    >,
    InterfaceMethod<
      /*desc=*/ "Returns the set of qubits that were measured",
      /*retType=*/ "mlir::ValueRange",
      /*methodName=*/ "getWires",
      /*args=*/ (ins),
      /*methodBody=*/ "return $_op.getWires();"
    >,
    InterfaceMethod<
      /*desc=*/ [{ Returns true if the targets are measured together as one
                   joint observable that yields a single classical result,
                   and false if each target is measured independently. }],
      /*retType=*/ "bool",
      /*methodName=*/ "isJointMeasurement",
      /*args=*/ (ins),
      /*methodBody=*/ [{}],
      /*defaultImplementation=*/ "return false;"
    >,
    InterfaceMethod<
      /*desc=*/ "Returns the register name, if any.",
      /*retType=*/ "std::optional<std::string>",
      /*methodName=*/ "getOptionalRegisterName",
      /*args=*/ (ins),
      /*methodBody=*/ [{
        if (auto attr = $_op.getRegisterNameAttr())
          return {attr.getValue().str()};
        return {};
      }]
    >,
    InterfaceMethod<
      /*desc=*/ "Set the registerName attribute",
      /*retType=*/ "void",
      /*methodName=*/ "setRegisterName",
      /*args=*/ (ins "llvm::StringRef":$name),
      /*methodBody=*/ "$_op.setRegisterName(name);"
    >
  ];
}

#endif // CUDAQ_OPTIMIZER_DIALECT_QUAKE_IR_INTERFACES
