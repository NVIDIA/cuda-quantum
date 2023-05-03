/********************************************************** -*- tablegen -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

// This files defines a set of common quantum instructions that part of the 
// Quake and QTX dialects.  The dialects must implement the `Operator` and 
// `Measurement` classes which are used as base.  Adding an operation here will
// add this operation to both dialects.

#ifndef CUDAQ_OPTIMIZER_DIALECT_COMMON_OPS
#define CUDAQ_OPTIMIZER_DIALECT_COMMON_OPS

include "cudaq/Optimizer/Dialect/Common/Traits.td"

//===----------------------------------------------------------------------===//
// Generic operator classes
//===----------------------------------------------------------------------===//

class GenOperator<string mnemonic, list<Trait> traits = []>
    : Operator<mnemonic, !listconcat([QuantumGate], traits)> {
  code OpBaseDeclaration = OperatorExtraClassDeclaration # [{
    //===------------------------------------------------------------------===//
    // Properties
    //===------------------------------------------------------------------===//

    bool isAdj() { return getIsAdj(); }

    //===------------------------------------------------------------------===//
    // Element Access
    //===------------------------------------------------------------------===//

    mlir::Value getParameter(unsigned i = 0u) { return getParameters()[i]; }

    /// If the parameter is known at compilation-time, set the result value and
    /// returns success. Otherwise, returns failure.
    mlir::LogicalResult getParameterAsDouble(double& result, unsigned i = 0) {
      auto paramDefOp = getParameter(i).getDefiningOp();
      if (!paramDefOp)
        return mlir::failure();
      if (auto constOp = mlir::dyn_cast<mlir::arith::ConstantOp>(paramDefOp)) {
        if (auto value = dyn_cast<mlir::FloatAttr>(constOp.getValue())) {
          result = value.getValueAsDouble();
          return mlir::success();
        }
      }
      return mlir::failure();
    }

    mlir::Value getTarget(unsigned i = 0u) { return getTargets()[i]; }
  }];
}

class OneTargetOp<string mnemonic, list<Trait> traits = []> :
    GenOperator<mnemonic, !listconcat([NumParameters<0>, NumTargets<1>], traits)>;

class OneTargetParamOp<string mnemonic, list<Trait> traits = []> :
    GenOperator<mnemonic, !listconcat([NumParameters<1>, NumTargets<1>], traits)>;

class TwoTargetOp<string mnemonic, list<Trait> traits = []> :
    GenOperator<mnemonic, !listconcat([NumParameters<0>, NumTargets<2>], traits)>;

//===----------------------------------------------------------------------===//
// Pauli instructions
//===----------------------------------------------------------------------===//

def XOp : OneTargetOp<"x", [Hermitian]> {
  let summary = "Pauli-X operation (aka, NOT)";
  let description = [{
    Matrix representation:
    ```
    X = | 0  1 |
        | 1  0 |
    ```

    Circuit symbol:
    ```
        ┌───┐
    q: ─┤ X ├─
        └───┘
    ```
  }];
  let extraClassDeclaration = OpBaseDeclaration # [{
    //===------------------------------------------------------------------===//
    // Operator interface
    //===------------------------------------------------------------------===//
    using Matrix = mlir::SmallVectorImpl<std::complex<double>>;

    /// Set `matrix` to the operator's unitary matrix as a column-major array.
    void getOperatorMatrix(Matrix &matrix) {
      matrix.assign({0, 1, 1, 0});
    }

    bool isClifford() {
      return getControls().size() < 2;
    }
  }];
}

def YOp : OneTargetOp<"y", [Hermitian]> {
  let summary = "Pauli-Y operation";
  let description = [{
    Matrix representation:
    ```
    Y = | 0  -i |
        | i   0 |
    ```

    Circuit symbol:
    ```
        ┌───┐
    q: ─┤ Y ├─
        └───┘
    ```
  }];
  let extraClassDeclaration = OpBaseDeclaration # [{
    //===------------------------------------------------------------------===//
    // Operator interface
    //===------------------------------------------------------------------===//
    using Matrix = mlir::SmallVectorImpl<std::complex<double>>;

    /// Set `matrix` to the operator's unitary matrix as a column-major array.
    void getOperatorMatrix(Matrix &matrix) {
      using namespace std::complex_literals;
      matrix.assign({0, 1i, -1i, 0});
    }

    bool isClifford() {
      return getControls().size() < 2;
    }
  }];
}

def ZOp : OneTargetOp<"z", [Hermitian]> {
  let summary = "Pauli-Z operation.";
  let description = [{
    Matrix representation:
    ```
    Z = | 1   0 |
        | 0  -1 |
    ```

    Circuit symbol:
    ```
        ┌───┐
    q: ─┤ Z ├─
        └───┘
    ```
  }];
  let extraClassDeclaration = OpBaseDeclaration # [{
    //===------------------------------------------------------------------===//
    // Operator interface
    //===------------------------------------------------------------------===//
    using Matrix = mlir::SmallVectorImpl<std::complex<double>>;

    /// Set `matrix` to the operator's unitary matrix as a column-major array.
    void getOperatorMatrix(Matrix &matrix) {
      matrix.assign({1, 0, 0, -1});
    }

    bool isClifford() {
      return getControls().size() < 2;
    }
  }];
}

//===----------------------------------------------------------------------===//
// Clifford+T instructions (Pauli's are also part of the Clifford group)
//===----------------------------------------------------------------------===//

def HOp : OneTargetOp<"h", [Hermitian]> {
  let summary = "Hadamard operation";
  let description = [{
    This is a pi rotation about the X+Z axis, and has the effect of changing
    computation basis from |0> (|1>) to |+> (|->) and vice-versa.  Meaning that
    it enables one to create a superposition of basis states.

    Matrix representation:
    ```
    H = (1 / sqrt(2)) * | 1   1 |
                        | 1  -1 |
    ```

    Circuit symbol:
    ```
        ┌───┐
    q: ─┤ H ├─
        └───┘
    ```
  }];
  let extraClassDeclaration = OpBaseDeclaration # [{
    //===------------------------------------------------------------------===//
    // Operator interface
    //===------------------------------------------------------------------===//
    using Matrix = mlir::SmallVectorImpl<std::complex<double>>;

    /// Set `matrix` to the operator's unitary matrix as a column-major array.
    void getOperatorMatrix(Matrix &matrix) {
      using namespace llvm::numbers;
      matrix.assign({inv_sqrt2, inv_sqrt2, inv_sqrt2, -inv_sqrt2});
    }

    bool isClifford() {
      return getControls().size() == 0;
    }
  }];
}

def SOp : OneTargetOp<"s"> {
  let summary = "S operation (aka, P or Sqrt(Z))";
  let description = [{
    This operation applies to its target a pi/2 rotation about the Z axis.

    Matrix representation:
    ```
    S = | 1   0 |
        | 0   i |
    ```

    Circuit symbol:
    ```
        ┌───┐
    q: ─┤ S ├─
        └───┘
    ```
  }];
  let extraClassDeclaration = OpBaseDeclaration # [{
    //===------------------------------------------------------------------===//
    // Operator interface
    //===------------------------------------------------------------------===//
    using Matrix = mlir::SmallVectorImpl<std::complex<double>>;

    /// Set `matrix` to the operator's unitary matrix as a column-major array.
    void getOperatorMatrix(Matrix &matrix) {
      using namespace llvm::numbers;
      using namespace std::complex_literals;
      if (getIsAdj())
        matrix.assign({1, 0, 0, -1i});
      else
        matrix.assign({1, 0, 0, 1i});
    }

    bool isClifford() {
      return getControls().size() == 0;
    }
  }];
}

def TOp : OneTargetOp<"t"> {
  let summary = "T operation";
  let description = [{
    This operation applies to its target a π/4 rotation about the Z axis.

    Matrix representation:
    ```
    T = | 1      0     |
        | 0  exp(iπ/4) |
    ```

    Circuit symbol:
    ```
        ┌───┐
    q: ─┤ T ├─
        └───┘
    ```
  }];
  let extraClassDeclaration = OpBaseDeclaration # [{
    //===------------------------------------------------------------------===//
    // Operator interface
    //===------------------------------------------------------------------===//
    using Matrix = mlir::SmallVectorImpl<std::complex<double>>;

    /// Set `matrix` to the operator's unitary matrix as a column-major array.
    void getOperatorMatrix(Matrix &matrix) {
      using namespace llvm::numbers;
      if (getIsAdj())
        matrix.assign({1, 0, 0, {inv_sqrt2, -inv_sqrt2}});
      else
        matrix.assign({1, 0, 0, {inv_sqrt2, inv_sqrt2}});
    }
  }];
}

def SwapOp : TwoTargetOp<"swap", [Hermitian]> {
  let summary = "Swap operation";
  let description = [{
    This operation swaps the states of two wires.

    Matrix representation:
    ```
    Swap = | 1 0 0 0 |
           | 0 0 1 0 |
           | 0 1 0 0 |
           | 0 0 0 1 |
    ```

    Circuit symbol:
    ```
    q: ─X─
        │
    q: ─X─
    ```
  }];
  let extraClassDeclaration = OpBaseDeclaration # [{
    //===------------------------------------------------------------------===//
    // Operator interface
    //===------------------------------------------------------------------===//
    using Matrix = mlir::SmallVectorImpl<std::complex<double>>;

    /// Set `matrix` to the operator's unitary matrix as a column-major array.
    void getOperatorMatrix(Matrix &matrix) {
      matrix.assign({1, 0, 0, 0,
                     0, 0, 1, 0,
                     0, 1, 0, 0,
                     0, 0, 0, 1});
    }

    bool isClifford() {
      return getControls().size() == 0;
    }
  }];
}

//===----------------------------------------------------------------------===//
// Arbitrary rotation instructions
//===----------------------------------------------------------------------===//

def R1Op : OneTargetParamOp<"r1"> {
  let summary = "an arbitrary rotation about the |1> state";
  let description = [{
    Matrix representation:
    ```
    R1(λ) = | 1     0    |
            | 0  exp(iλ) |
    ```

    Circuit symbol:
    ```
        ┌───────┐
    q: ─┤ R1(λ) ├─
        └───────┘
    ```
  }];
  let extraClassDeclaration = OpBaseDeclaration # [{
    //===------------------------------------------------------------------===//
    // Operator interface
    //===------------------------------------------------------------------===//
    using Matrix = mlir::SmallVectorImpl<std::complex<double>>;

    /// Set `matrix` to the operator's unitary matrix as a column-major array if
    /// it can be computed at compilation time, i.e., if the parameter is a
    /// compile-time constant.
    void getOperatorMatrix(Matrix &matrix) {
      using namespace std::complex_literals;
      double theta;
      if (failed(getParameterAsDouble(theta)))
        return;
      if (getIsAdj())
        theta *= -1;
      matrix.assign({1, 0, 0, std::exp(theta * 1i)});
    }
  }];
}

def RxOp : OneTargetParamOp<"rx"> {
  let summary = "an arbitrary rotation about the X axis";
  let description = [{
    Matrix representation:
    ```
    Rx(ϴ) =  exp(-(i * ϴ/2) * X) = |  cos(ϴ/2)  -isin(ϴ/2) |
                                   | -isin(ϴ/2)  cos(ϴ/2)  |
    ```

    Circuit symbol:
    ```
        ┌───────┐
    q: ─┤ Rx(ϴ) ├─
        └───────┘
    ```
  }];
  let extraClassDeclaration = OpBaseDeclaration # [{
    //===------------------------------------------------------------------===//
    // Operator interface
    //===------------------------------------------------------------------===//
    using Matrix = mlir::SmallVectorImpl<std::complex<double>>;

    /// Set `matrix` to the operator's unitary matrix as a column-major array if
    /// it can be computed at compilation time, i.e., if the parameter is a
    /// compile-time constant.
    void getOperatorMatrix(Matrix &matrix) {
      using namespace std::complex_literals;
      double theta;
      if (failed(getParameterAsDouble(theta)))
        return;
      if (getIsAdj())
        theta *= -1;
      matrix.assign({      std::cos(theta / 2.), -1i * std::sin(theta / 2.),
                     -1i * std::sin(theta / 2.), std::cos(theta / 2.)});
    }
  }];
}

def PhasedRxOp : GenOperator<"phased_rx", [NumParameters<2>, NumTargets<1>]> {
  let summary = "Phased rx gate. Two parameters and one target qubit.";
  let description = [{
    Matrix representation:
    ```
    PhasedRx(ϴ,φ) = |        cos(ϴ/2)        -iexp(-iφ) * sin(ϴ/2) |
                    | -iexp(iφ)) * sin(ϴ/2)         cos(ϴ/2)       |
    ```

    Circuit symbol:
    ```
        ┌───────────────┐
    q: ─┤ PhasedRx(ϴ,φ) ├─
        └───────────────┘
    ```
  }];
  let extraClassDeclaration = OpBaseDeclaration # [{
    //===------------------------------------------------------------------===//
    // Operator interface
    //===------------------------------------------------------------------===//
    using Matrix = mlir::SmallVectorImpl<std::complex<double>>;

    /// Set `matrix` to the operator's unitary matrix as a column-major array if
    /// it can be computed at compilation time, i.e., if the parameter is a
    /// compile-time constant.
    void getOperatorMatrix(Matrix &matrix) {
      using namespace std::complex_literals;

      // Get parameters
      double theta;
      double phi;
      if (failed(getParameterAsDouble(theta)) ||
          failed(getParameterAsDouble(phi, /*i =*/1)))
        return;

      if (getIsAdj())
        theta *= -1;

      matrix.assign({
        std::cos(theta / 2.),
        -1i * std::exp(1i * phi ) * std::sin(theta / 2.),
        -1i * std::exp(-1i * phi) * std::sin(theta / 2.),
        std::cos(theta / 2.)
      });
    }
  }];
}

def RyOp : OneTargetParamOp<"ry"> {
  let summary = "an arbitrary rotation about the Y axis";
  let description = [{
    Matrix representation:
    ```
    Ry(ϴ) = exp(-(i * ϴ/2) * Y) = | cos(ϴ/2)  -sin(ϴ/2) |
                                  | sin(ϴ/2)   cos(ϴ/2) |
    ```

    Circuit symbol:
    ```
        ┌───────┐
    q: ─┤ Ry(ϴ) ├─
        └───────┘
    ```
  }];
  let extraClassDeclaration = OpBaseDeclaration # [{
    //===------------------------------------------------------------------===//
    // Operator interface
    //===------------------------------------------------------------------===//
    using Matrix = mlir::SmallVectorImpl<std::complex<double>>;

    /// Set `matrix` to the operator's unitary matrix as a column-major array if
    /// it can be computed at compilation time, i.e., if the parameter is a
    /// compile-time constant.
    void getOperatorMatrix(Matrix &matrix) {
      // Get parameter
      double theta;
      if (failed(getParameterAsDouble(theta)))
        return;

      if (getIsAdj())
        theta *= -1;

      matrix.assign({std::cos(theta / 2.), std::sin(theta / 2.),
                     -std::sin(theta / 2.), std::cos(theta / 2.)});
    }
  }];
}

def RzOp : OneTargetParamOp<"rz"> {
  let summary = "an arbitrary rotation about the Z axis";
  let description = [{
    Matrix representation:
    ```
    Rz(λ) = exp(-iλ/2 * Z) = | exp(-iλ/2)      0     |
                             |     0       exp(iλ/2) |
    ```

    Circuit symbol:
    ```
        ┌───────┐
    q: ─┤ Rz(λ) ├─
        └───────┘
    ```
  }];
  let extraClassDeclaration = OpBaseDeclaration # [{
    //===------------------------------------------------------------------===//
    // Operator interface
    //===------------------------------------------------------------------===//
    using Matrix = mlir::SmallVectorImpl<std::complex<double>>;

    /// Set `matrix` to the operator's unitary matrix as a column-major array if
    /// it can be computed at compilation time, i.e., if the parameter is a
    /// compile-time constant.
    void getOperatorMatrix(Matrix &matrix) {
      using namespace std::complex_literals;

      // Get parameter
      double theta;
      if (failed(getParameterAsDouble(theta)))
        return;

      if (getIsAdj())
        theta *= -1;

      matrix.assign({std::exp(-1i * theta / 2.), 0,
                     0, std::exp(1i * theta / 2.)});
    }
  }];
}

//===----------------------------------------------------------------------===//
// Universal intructions
//===----------------------------------------------------------------------===//

def U2Op : GenOperator<"u2", [NumParameters<2>, NumTargets<1>]> {
  let summary = "generic rotation about the X+Z axis";
  let description = [{
    The two parameters are Euler angles: φ and λ.

    Matrix representation:
    ```
    U2(φ,λ) = 1/sqrt(2) * | 1        -exp(iλ)       |
                          | exp(iφ)   exp(i(λ + φ)) |
    ```

    Circuit symbol:
    ```
        ┌─────────┐
    q: ─┤ U2(φ,λ) ├─
        └─────────┘
    ```
  }];
  let extraClassDeclaration = OpBaseDeclaration # [{
    //===------------------------------------------------------------------===//
    // Operator interface
    //===------------------------------------------------------------------===//
    using Matrix = mlir::SmallVectorImpl<std::complex<double>>;

    /// Set `matrix` to the operator's unitary matrix as a column-major array if
    /// it can be computed at compilation time, i.e., if the parameter is a
    /// compile-time constant.
    void getOperatorMatrix(Matrix &matrix) {
      using namespace llvm::numbers;
      using namespace std::complex_literals;

      // Get parameters
      double phi;
      double lambda;
      if (failed(getParameterAsDouble(phi)) ||
          failed(getParameterAsDouble(lambda, /*i =*/1)))
        return;

      if (getIsAdj()) {
        phi *= -1;
        lambda *= -1;
      }

      matrix.assign({
        inv_sqrt2,
        inv_sqrt2 * std::exp(phi * 1i),
        -inv_sqrt2 * std::exp(lambda * 1i),
        inv_sqrt2 * std::exp(1i * (phi + lambda))
      });
    }
  }];
}

def U3Op : GenOperator<"u3", [NumParameters<3>, NumTargets<1>]> {
  let summary = "the universal three-parameters operator";
  let description = [{
    The three parameters are Euler angles: ϴ, φ, and λ.

    NOTE: U3 is a generalization of U2 that covers all single-qubit rotations.

    Matrix representation:
    ```
    U3(ϴ,φ,λ) = | cos(ϴ/2)            -exp(iλ) * sin(ϴ/2)       |
                | exp(iφ) * sin(ϴ/2)   exp(i(λ + φ)) * cos(ϴ/2) |
    ```

    Circuit symbol:
    ```
        ┌───────────┐
    q: ─┤ U3(ϴ,φ,λ) ├─
        └───────────┘
    ```
  }];
  let extraClassDeclaration = OpBaseDeclaration # [{
    //===------------------------------------------------------------------===//
    // Operator interface
    //===------------------------------------------------------------------===//
    using Matrix = mlir::SmallVectorImpl<std::complex<double>>;

    /// Set `matrix` to the operator's unitary matrix as a column-major array if
    /// it can be computed at compilation time, i.e., if the parameter is a
    /// compile-time constant.
    void getOperatorMatrix(Matrix &matrix) {
      using namespace std::complex_literals;

      // Get parameters
      double theta;
      double phi;
      double lambda;
      if (failed(getParameterAsDouble(theta)) ||
          failed(getParameterAsDouble(phi, /*i =*/1)) ||
          failed(getParameterAsDouble(lambda, /*i =*/2)))
        return;

      if (getIsAdj()) {
        theta *= -1;
        phi *= -1;
        lambda *= -1;
      }

      matrix.assign({
        std::cos(theta / 2.),
        std::exp(phi * 1i) * std::sin(theta / 2.),
        -std::exp(lambda * 1i) * std::sin(theta / 2.),
        std::exp(1i * (phi + lambda)) * std::cos(theta / 2.)
      });
    }
  }];
}

//===----------------------------------------------------------------------===//
// Measurements
//===----------------------------------------------------------------------===//

def MxOp : Measurement<"mx"> {
  let summary = "Measure a single qubit along the x-axis";
  let description = [{
    The `mx` operation measures the state of qubits into classical bits
    represented by a `i1` (or a vector of `i1`), along the x-axis.

    The state of the qubits is collapsed into one of the computational basis
    states, i.e., either |0> or |1>.  A `reset` operation can guarantee that the
    qubit returns to a |0> state, and thus it can be used for further
    computation.  Another option is to deallocate the qubit using `dealloc`.
  }];
}

def MyOp : Measurement<"my"> {
  let summary = "Measure a single qubit along the y-axis";
  let description = [{
    The `mx` operation measures the state of qubits into classical bits
    represented by a `i1` (or a vector of `i1`), along the y-axis.

    The state of the qubit is collapsed into one of the computational basis
    states, i.e., either |0> or |1>.  A `reset` operation can guarantee that the
    qubit returns to a |0> state, and thus it can be used for further
    computation.  Another option is to deallocate the qubit using `dealloc`.
  }];
}

def MzOp : Measurement<"mz"> {
  let summary = "Measure a single qubit along the z-axis";
  let description = [{
    The `mz` operation measures the state of qubits into a classical bits
    represented by a `i1` (or a vector of `i1`), along the z-axis---the
    so-called computational basis.

    The state of the qubit is collapsed into one of the computational basis
    states, i.e., either |0> or |1>.  A `reset` operation can guarantee that the
    qubit returns to a |0> state, and thus it can be used for further
    computation.  Another option is to deallocate the qubit using `dealloc`.
  }];
}

#endif // CUDAQ_OPTIMIZER_DIALECT_COMMON_OPS
