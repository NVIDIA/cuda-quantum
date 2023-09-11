/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "llvm/ADT/ScopedHashTable.h"
#if (defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER))
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
#include "llvm/Support/FormatVariadic.h"
#if (defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER))
#pragma GCC diagnostic pop
#endif
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/IndentedOstream.h"
#include <stack>

namespace cudaq {

/// Try to convert `value`, which is used as an index, to an integer
inline std::optional<int64_t> getIndexValueAsInt(mlir::Value value) {
  // The value might not have an defining operation, e.g., when the value is an
  // circuit argument
  if (auto constOp =
          dyn_cast_if_present<mlir::arith::ConstantOp>(value.getDefiningOp())) {
    if (auto index = dyn_cast<mlir::IntegerAttr>(constOp.getValue())) {
      return index.getInt();
    }
  }
  return {};
}

/// Try to convert `value`, which is used as a operator's parameter, to a double
inline std::optional<double> getParameterValueAsDouble(mlir::Value value) {
  // The value might not have an defining operation, e.g., when the value is an
  // circuit argument.
  if (auto constOp =
          dyn_cast_if_present<mlir::arith::ConstantOp>(value.getDefiningOp())) {
    if (auto index = dyn_cast<mlir::FloatAttr>(constOp.getValue())) {
      return index.getValueAsDouble();
    }
  }
  return {};
}

/// Base emitter to translate Quake to quantum assembly-like languages.
struct Emitter {
  explicit Emitter(mlir::raw_ostream &os) : os(os) {
    valuesInScopeCount.push(0);
  }

  /// Returns a new name by combining `prefix` and the number of values in scope
  std::string createName(mlir::StringRef prefix = "var",
                         int64_t index_offset = 0) {
    return llvm::formatv("{0}{1}", prefix,
                         valuesInScopeCount.top() + index_offset);
  }

  /// Return the existing name for a value, or assign `name` to it.
  mlir::StringRef getOrAssignName(mlir::Value value,
                                  std::string const &name = "") {
    if (!valueToName.count(value)) {
      assert(!name.empty() && "Cannot give a value an empty name");
      valueToName.insert(value, name);
      valuesInScopeCount.top() += 1;
    }
    return *valueToName.begin(value);
  }

  /// Assigns to `unamedValue` the name same name of `namedValue`.
  void mapValuesName(mlir::Value namedValue, mlir::Value unamedValue) {
    assert(valueToName.count(namedValue) && "Unknown named value");
    valueToName.insert(unamedValue, *valueToName.begin(namedValue));
  }

  void mapValuesName(mlir::ValueRange namedValues,
                     mlir::ValueRange unamedValues) {
    for (auto [named, unamed] : llvm::zip(namedValues, unamedValues)) {
      mapValuesName(named, unamed);
    }
  }

  /// Helper struct to manage entering and exiting scopes.
  struct Scope {
    Scope(Emitter &emitter, bool isEntryPoint = false)
        : isEntryPoint(isEntryPoint), valueToName(emitter.valueToName),
          emitter(emitter) {
      emitter.valuesInScopeCount.push(emitter.valuesInScopeCount.top());
    }
    ~Scope() { emitter.valuesInScopeCount.pop(); }

    bool isEntryPoint;

  private:
    llvm::ScopedHashTableScope<mlir::Value, std::string> valueToName;
    Emitter &emitter;
  };

  /// Output stream to emit to.
  mlir::raw_indented_ostream os;

  /// Map from value to a variable name.
  llvm::ScopedHashTable<mlir::Value, std::string> valueToName;

  /// The number of values in the current scope. This is used to declare the
  /// variable names of values in a scope, e.g, `q0`, `q1` etc.
  std::stack<int64_t> valuesInScopeCount;
};

} // namespace cudaq
