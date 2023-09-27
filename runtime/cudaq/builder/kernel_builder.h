/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/qis/modifiers.h"
#include "cudaq/qis/qreg.h"
#include "cudaq/qis/qvector.h"
#include "cudaq/utils/cudaq_utils.h"
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "QuakeValue.h"

// Goal here is to keep MLIR out of user code!
namespace mlir {
class Type;
class Block;
class ImplicitLocOpBuilder;
class MLIRContext;
class DialectRegistry;
class Value;
class ExecutionEngine;
class PassManager;
} // namespace mlir

using namespace mlir;

namespace cudaq {
std::string get_quake_by_name(const std::string &);

/// @brief Define a floating point concept
template <typename T>
concept NumericType = requires(T param) { std::is_floating_point_v<T>; };

/// @brief Define a Quake-constructable floating point value concept
// i.e., it could be a `QuakeValue` type or a floating point number (convertible
// to a `QuakeValue` with `ConstantFloatOp`).
template <typename T>
concept QuakeValueOrNumericType = requires(T param) {
  std::is_floating_point_v<T> ||
      std::is_same_v<std::remove_cvref_t<T>, QuakeValue>;
};

/// @brief Define a floating point concept
template <typename T>
concept IntegralType = requires(T param) { std::is_integral_v<T>; };

// Helper template type to check if type is in a variadic pack
template <typename T, typename... Ts>
concept KernelBuilderArgTypeIsValid =
    std::disjunction_v<std::is_same<T, Ts>...>;

// If you want to add to the list of valid kernel argument types
// first add it here, then add `details::mapArgToType()` function
#define CUDAQ_VALID_BUILDER_ARGS_FOLD()                                        \
  requires(                                                                    \
      KernelBuilderArgTypeIsValid<                                             \
          Args, float, double, std::size_t, int, std::vector<int>,             \
          std::vector<float>, std::vector<std::size_t>, std::vector<double>,   \
          cudaq::qubit, cudaq::qreg<>, cudaq::qvector<>> &&                    \
      ...)

namespace details {

// Define a `mlir::Type` generator in the `cudaq` namespace,
// this helps us keep MLIR out of this public header

/// @brief The `kernel_builder::Type` allows us to track
/// input C++ types representing the quake function argument types
/// in a way that does not expose MLIR Type to the CUDA Quantum code.
/// This type keeps track of a functor that generates the MLIR Type
/// in implementation code when create() is invoked.
class KernelBuilderType {
protected:
  /// @brief For this type instance, create an MLIR Type
  std::function<mlir::Type(mlir::MLIRContext *)> creator;

public:
  /// @brief The constructor, take the Type generation functor
  KernelBuilderType(std::function<mlir::Type(mlir::MLIRContext *ctx)> &&f);

  /// Create the MLIR Type
  mlir::Type create(mlir::MLIRContext *ctx);
};

/// Map a `double` to a `KernelBuilderType`
KernelBuilderType mapArgToType(double &e);

/// Map a `float` to a `KernelBuilderType`
KernelBuilderType mapArgToType(float &e);

/// Map a `int` to a `KernelBuilderType`
KernelBuilderType mapArgToType(int &e);

/// Map a `size_t` to a `KernelBuilderType`
KernelBuilderType mapArgToType(std::size_t &e);

/// Map a `std::vector<int>` to a `KernelBuilderType`
KernelBuilderType mapArgToType(std::vector<int> &e);

/// Map a `std::vector<std::size_t>` to a `KernelBuilderType`
KernelBuilderType mapArgToType(std::vector<std::size_t> &e);

/// Map a `std::vector<float>` to a `KernelBuilderType`
KernelBuilderType mapArgToType(std::vector<float> &e);

/// Map a `vector<double>` to a `KernelBuilderType`
KernelBuilderType mapArgToType(std::vector<double> &e);

/// Map a `qubit` to a `KernelBuilderType`
KernelBuilderType mapArgToType(cudaq::qubit &e);

/// @brief  Map a `qreg` to a `KernelBuilderType`
KernelBuilderType mapArgToType(cudaq::qreg<> &e);

/// @brief  Map a qvector to a `KernelBuilderType`
KernelBuilderType mapArgToType(cudaq::qvector<> &e);

/// @brief Initialize the `MLIRContext`, return the raw
/// pointer which we'll wrap in an `unique_ptr`.
MLIRContext *initializeContext();

/// @brief Delete function for the context pointer,
/// also given to the `unique_ptr`
void deleteContext(MLIRContext *);

/// @brief Initialize the `OpBuilder`, return the raw
/// pointer which we'll wrap in an `unique_ptr`.
ImplicitLocOpBuilder *initializeBuilder(MLIRContext *,
                                        std::vector<KernelBuilderType> &,
                                        std::vector<QuakeValue> &,
                                        std::string &kernelName);

/// @brief Delete function for the builder pointer,
/// also given to the `unique_ptr`
void deleteBuilder(ImplicitLocOpBuilder *builder);

/// @brief Delete function for the JIT pointer,
/// also given to the `unique_ptr`
void deleteJitEngine(ExecutionEngine *jit);

/// @brief Allocate a single `qubit`
QuakeValue qalloc(ImplicitLocOpBuilder &builder);

/// @brief Allocate a `qvector`.
QuakeValue qalloc(ImplicitLocOpBuilder &builder, const std::size_t nQubits);

/// @brief Allocate a `qvector` from existing `QuakeValue` size
QuakeValue qalloc(ImplicitLocOpBuilder &builder, QuakeValue &size);

// In the following macros + instantiations, we define the functions
// that create Quake Quantum Ops + Measures

#define CUDAQ_DETAILS_QIS_DECLARATION(NAME)                                    \
  void NAME(ImplicitLocOpBuilder &builder, std::vector<QuakeValue> &ctrls,     \
            const QuakeValue &target, bool adjoint = false);

CUDAQ_DETAILS_QIS_DECLARATION(h)
CUDAQ_DETAILS_QIS_DECLARATION(s)
CUDAQ_DETAILS_QIS_DECLARATION(t)
CUDAQ_DETAILS_QIS_DECLARATION(x)
CUDAQ_DETAILS_QIS_DECLARATION(y)
CUDAQ_DETAILS_QIS_DECLARATION(z)

#define CUDAQ_DETAILS_ONEPARAM_QIS_DECLARATION(NAME)                           \
  void NAME(ImplicitLocOpBuilder &builder, QuakeValue &parameter,              \
            std::vector<QuakeValue> &ctrls, QuakeValue &target);

CUDAQ_DETAILS_ONEPARAM_QIS_DECLARATION(rx)
CUDAQ_DETAILS_ONEPARAM_QIS_DECLARATION(ry)
CUDAQ_DETAILS_ONEPARAM_QIS_DECLARATION(rz)
CUDAQ_DETAILS_ONEPARAM_QIS_DECLARATION(r1)

#define CUDAQ_DETAILS_MEASURE_DECLARATION(NAME)                                \
  QuakeValue NAME(ImplicitLocOpBuilder &builder, QuakeValue &target,           \
                  std::string regName = "");

CUDAQ_DETAILS_MEASURE_DECLARATION(mx)
CUDAQ_DETAILS_MEASURE_DECLARATION(my)
CUDAQ_DETAILS_MEASURE_DECLARATION(mz)

void exp_pauli(ImplicitLocOpBuilder &builder, const QuakeValue &theta,
               const std::vector<QuakeValue> &qubits,
               const std::string &pauliWord);

void swap(ImplicitLocOpBuilder &builder, const std::vector<QuakeValue> &ctrls,
          const std::vector<QuakeValue> &targets, bool adjoint = false);

void reset(ImplicitLocOpBuilder &builder, const QuakeValue &qubitOrQvec);

void c_if(ImplicitLocOpBuilder &builder, QuakeValue &conditional,
          std::function<void()> &thenFunctor);

/// @brief Return the name of this `kernel_builder`,
/// it is also the name of the function
std::string name(std::string_view kernelName);

/// @brief Apply our MLIR passes before JIT execution
void applyPasses(PassManager &);

/// @brief Create the `ExecutionEngine` and return a raw
/// pointer, which we will wrap in a `unique_ptr`
std::tuple<bool, ExecutionEngine *>
jitCode(ImplicitLocOpBuilder &, ExecutionEngine *,
        std::unordered_map<ExecutionEngine *, std::size_t> &, std::string,
        std::vector<std::string>);

/// @brief Invoke the function with the given kernel name.
void invokeCode(ImplicitLocOpBuilder &builder, ExecutionEngine *jit,
                std::string kernelName, void **argsArray,
                std::vector<std::string> extraLibPaths);

/// @brief Invoke the provided kernel function.
void call(ImplicitLocOpBuilder &builder, std::string &name,
          std::string &quakeCode, std::vector<QuakeValue> &values);

/// @brief Apply the given kernel controlled on the provided qubit value.
void control(ImplicitLocOpBuilder &builder, std::string &name,
             std::string &quakeCode, QuakeValue &control,
             std::vector<QuakeValue> &values);

/// @brief Apply the adjoint of the given kernel
void adjoint(ImplicitLocOpBuilder &builder, std::string &name,
             std::string &quakeCode, std::vector<QuakeValue> &values);

/// @brief Add a for loop that starts from the given `start` integer index,
/// ends at the given `end` integer index, and applies the given `body` as a
/// callable function. This callable function must take as input an index
/// variable that can be used within the body.
void forLoop(ImplicitLocOpBuilder &builder, std::size_t start, std::size_t end,
             std::function<void(QuakeValue &)> &body);

/// @brief Add a for loop that starts from the given `start` integer index,
/// ends at the given `end` index, and applies the given `body` as a callable
/// function. This callable function must take as input an index variable that
/// can be used within the body.
void forLoop(ImplicitLocOpBuilder &builder, std::size_t start, QuakeValue &end,
             std::function<void(QuakeValue &)> &body);

/// @brief Add a for loop that starts from the given `start` index,
/// ends at the given `end` integer index, and applies the given `body` as a
/// callable function. This callable function must take as input an index
/// variable that can be used within the body.
void forLoop(ImplicitLocOpBuilder &builder, QuakeValue &start, std::size_t end,
             std::function<void(QuakeValue &)> &body);

/// @brief Add a for loop that starts from the given `start` index,
/// ends at the given `end` index, and applies the given `body` as a
/// callable function. This callable function must take as input an index
/// variable that can be used within the body.
void forLoop(ImplicitLocOpBuilder &builder, QuakeValue &start, QuakeValue &end,
             std::function<void(QuakeValue &)> &body);

/// @brief Return the quake representation as a string
std::string to_quake(ImplicitLocOpBuilder &builder);

/// @brief Returns `true` if the argument to the `kernel_builder`
/// is a `cc::StdvecType`. Returns `false` otherwise.
bool isArgStdVec(std::vector<QuakeValue> &args, std::size_t idx);

/// @brief The `ArgumentValidator` provides a way validate the input
/// arguments when the kernel is invoked (via a fold expression).
template <typename T>
struct ArgumentValidator {
  static void validate(std::size_t &argCounter, std::vector<QuakeValue> &args,
                       T &val) {
    // Default case, do nothing for now
    argCounter++;
  }
};

/// @brief The `ArgumentValidator` provides a way validate the input
/// arguments when the kernel is invoked (via a fold expression). Here
/// we explicitly validate `std::vector<T>` and its size.
template <typename T>
struct ArgumentValidator<std::vector<T>> {
  static void validate(std::size_t &argCounter, std::vector<QuakeValue> &args,
                       std::vector<T> &input) {
    if (argCounter >= args.size())
      throw std::runtime_error("Error validating stdvec input to "
                               "kernel_builder. argCounter >= args.size()");

    // Get the argument, increment the counter
    auto &arg = args[argCounter];
    argCounter++;

    // Validate the input vector<T> if possible
    if (auto nRequiredElements = arg.getRequiredElements();
        arg.canValidateNumElements())
      if (input.size() != nRequiredElements)
        throw std::runtime_error(
            "Invalid vector<T> input. Number of elements provided != "
            "number of elements required (" +
            std::to_string(nRequiredElements) + " required).\n");
  }
};

/// @brief The `kernel_builder_base` provides a
/// base type for the templated kernel builder so that
/// we can get a single handle on an instance within the runtime.
class kernel_builder_base {
public:
  virtual std::string to_quake() const = 0;
  virtual void jitCode(std::vector<std::string> extraLibPaths = {}) = 0;
  virtual ~kernel_builder_base() = default;

  /// @brief Write the kernel_builder to the given output stream.
  /// This outputs the Quake representation.
  friend std::ostream &operator<<(std::ostream &stream,
                                  const kernel_builder_base &builder) {
    stream << builder.to_quake();
    return stream;
  }
};

} // namespace details

template <class... Ts>
concept AllAreQuakeValues =
    sizeof...(Ts) < 2 ||
    (std::conjunction_v<
         std::is_same<std::tuple_element_t<0, std::tuple<Ts...>>, Ts>...> &&
     std::is_same_v<
         std::remove_reference_t<std::tuple_element<0, std::tuple<Ts...>>>,
         QuakeValue>);

template <typename... Args>
class kernel_builder : public details::kernel_builder_base {
private:
  /// @brief Handle to the MLIR Context, stored
  /// as a pointer here to keep implementation details
  /// out of CUDA Quantum code
  std::unique_ptr<MLIRContext, void (*)(MLIRContext *)> context;

  /// @brief Handle to the MLIR `OpBuilder`, stored
  /// as a pointer here to keep implementation details
  /// out of CUDA Quantum code
  std::unique_ptr<ImplicitLocOpBuilder, void (*)(ImplicitLocOpBuilder *)>
      opBuilder;

  /// @brief Handle to the MLIR `ExecutionEngine`, stored
  /// as a pointer here to keep implementation details
  /// out of CUDA Quantum code
  std::unique_ptr<ExecutionEngine, void (*)(ExecutionEngine *)> jitEngine;

  /// @brief Map created ExecutionEngines to a unique hash of the
  /// ModuleOp they derive from.
  std::unordered_map<ExecutionEngine *, std::size_t> jitEngineToModuleHash;

  /// @brief Name of the CUDA Quantum kernel Quake function
  std::string kernelName = "__nvqpp__mlirgen____nvqppBuilderKernel";

  /// @brief The CUDA Quantum Quake function arguments stored
  /// as `QuakeValue`s.
  std::vector<QuakeValue> arguments;

public:
  /// @brief The constructor, takes the input
  /// `KernelBuilderType`s which is used to create the MLIR
  /// function type
  kernel_builder(std::vector<details::KernelBuilderType> &types)
      : context(details::initializeContext(), details::deleteContext),
        opBuilder(nullptr, [](ImplicitLocOpBuilder *) {}),
        jitEngine(nullptr, [](ExecutionEngine *) {}) {
    auto *ptr =
        details::initializeBuilder(context.get(), types, arguments, kernelName);
    opBuilder =
        std::unique_ptr<ImplicitLocOpBuilder, void (*)(ImplicitLocOpBuilder *)>(
            ptr, details::deleteBuilder);
  }

  /// @brief Return the `QuakeValue` arguments
  /// @return
  auto &getArguments() { return arguments; }

  /// @brief Return `true` if the argument to the
  /// kernel is a `std::vector`, `false` otherwise.
  bool isArgStdVec(std::size_t idx) {
    return details::isArgStdVec(arguments, idx);
  }

  /// @brief Return the name of this kernel
  std::string name() { return details::name(kernelName); }

  /// @brief Return the number of function arguments.
  /// @return
  std::size_t getNumParams() { return arguments.size(); }

  /// @brief Return a `QuakeValue` representing the allocated qubit.
  QuakeValue qalloc() { return details::qalloc(*opBuilder.get()); }

  /// @brief Return a `QuakeValue` representing the allocated `QVec`.
  QuakeValue qalloc(const std::size_t nQubits) {
    return details::qalloc(*opBuilder.get(), nQubits);
  }

  /// @brief Return a `QuakeValue` representing the allocated `Veq`,
  /// size is from a pre-allocated size `QuakeValue` or `BlockArgument`.
  QuakeValue qalloc(QuakeValue size) {
    return details::qalloc(*opBuilder.get(), size);
  }

  // In the following macros + instantiations, we define the kernel_builder
  // methods that create Quake Quantum Ops + Measures

#define CUDAQ_BUILDER_ADD_ONE_QUBIT_OP(NAME)                                   \
  void NAME(QuakeValue &qubit) {                                               \
    std::vector<QuakeValue> empty;                                             \
    details::NAME(*opBuilder, empty, qubit);                                   \
  }                                                                            \
  void NAME(QuakeValue &&qubit) { NAME(qubit); }                               \
  void NAME(std::vector<QuakeValue> &ctrls, QuakeValue &target) {              \
    details::NAME(*opBuilder, ctrls, target);                                  \
  }                                                                            \
  template <typename mod, typename... QubitValues,                             \
            typename = typename std::enable_if_t<sizeof...(QubitValues) >= 2>, \
            typename =                                                         \
                typename std::enable_if_t<std::is_same_v<mod, cudaq::ctrl> ||  \
                                          std::is_same_v<mod, cudaq::base>>>   \
  void NAME(QubitValues... args) {                                             \
    std::vector<QuakeValue> values{args...};                                   \
    if constexpr (std::is_same_v<mod, cudaq::ctrl>) {                          \
      std::vector<QuakeValue> ctrls(values.begin(), values.end() - 1);         \
      auto &target = values.back();                                            \
      NAME(ctrls, target);                                                     \
      return;                                                                  \
    }                                                                          \
    for (auto &v : values) {                                                   \
      NAME(v);                                                                 \
    }                                                                          \
  }                                                                            \
  template <typename mod,                                                      \
            typename =                                                         \
                typename std::enable_if_t<std::is_same_v<mod, cudaq::adj>>>    \
  void NAME(const QuakeValue &qubit) {                                         \
    std::vector<QuakeValue> empty;                                             \
    details::NAME(*opBuilder, empty, qubit, true);                             \
  }

  CUDAQ_BUILDER_ADD_ONE_QUBIT_OP(h)
  CUDAQ_BUILDER_ADD_ONE_QUBIT_OP(s)
  CUDAQ_BUILDER_ADD_ONE_QUBIT_OP(t)
  CUDAQ_BUILDER_ADD_ONE_QUBIT_OP(x)
  CUDAQ_BUILDER_ADD_ONE_QUBIT_OP(y)
  CUDAQ_BUILDER_ADD_ONE_QUBIT_OP(z)

#define CUDAQ_BUILDER_ADD_ONE_QUBIT_PARAM(NAME)                                \
  void NAME(QuakeValue parameter, QuakeValue qubit) {                          \
    std::vector<QuakeValue> empty;                                             \
    details::NAME(*opBuilder, parameter, empty, qubit);                        \
  }                                                                            \
  void NAME(QuakeValue parameter, std::vector<QuakeValue> &ctrls,              \
            QuakeValue &target) {                                              \
    details::NAME(*opBuilder, parameter, ctrls, target);                       \
  }                                                                            \
  void NAME(double param, QuakeValue qubit) {                                  \
    std::vector<QuakeValue> empty;                                             \
    QuakeValue v(*opBuilder, param);                                           \
    details::NAME(*opBuilder, v, empty, qubit);                                \
  }                                                                            \
  template <typename mod, QuakeValueOrNumericType ParamT,                      \
            typename =                                                         \
                typename std::enable_if_t<std::is_same_v<mod, cudaq::adj>>>    \
  void NAME(const ParamT &parameter, QuakeValue qubit) {                       \
    if constexpr (std::is_floating_point_v<ParamT>)                            \
      NAME(QuakeValue(*opBuilder, -parameter), qubit);                         \
    else                                                                       \
      NAME(-parameter, qubit);                                                 \
  }                                                                            \
  template <typename mod, QuakeValueOrNumericType ParamT,                      \
            typename... QubitValues,                                           \
            typename = typename std::enable_if_t<sizeof...(QubitValues) >= 2>> \
  void NAME(const ParamT &parameter, QubitValues... args) {                    \
    std::vector<QuakeValue> values{args...};                                   \
    if constexpr (std::is_same_v<mod, cudaq::ctrl>) {                          \
      std::vector<QuakeValue> ctrls(values.begin(), values.end() - 1);         \
      auto &target = values.back();                                            \
      if constexpr (std::is_floating_point_v<ParamT>)                          \
        NAME(QuakeValue(*opBuilder, parameter), ctrls, target);                \
      else                                                                     \
        NAME(parameter, ctrls, target);                                        \
      return;                                                                  \
    }                                                                          \
  }

  CUDAQ_BUILDER_ADD_ONE_QUBIT_PARAM(rx)
  CUDAQ_BUILDER_ADD_ONE_QUBIT_PARAM(ry)
  CUDAQ_BUILDER_ADD_ONE_QUBIT_PARAM(rz)
  CUDAQ_BUILDER_ADD_ONE_QUBIT_PARAM(r1)

#define CUDAQ_BUILDER_ADD_MEASURE(NAME)                                        \
  QuakeValue NAME(QuakeValue qubitOrQvec) {                                    \
    return details::NAME(*opBuilder, qubitOrQvec);                             \
  }                                                                            \
  auto NAME(QuakeValue qubit, const std::string &regName) {                    \
    return details::NAME(*opBuilder, qubit, regName);                          \
  }                                                                            \
  auto NAME(QuakeValue qubit, const std::string &&regName) {                   \
    return NAME(qubit, regName);                                               \
  }                                                                            \
  template <                                                                   \
      typename... QubitValues,                                                 \
      typename = typename std::enable_if_t<                                    \
          sizeof...(QubitValues) >= 2 &&                                       \
          !std::is_same_v<decltype(std::get<1>(std::tuple<QubitValues...>())), \
                          std::string>>>                                       \
  auto NAME(QubitValues... args) {                                             \
    std::vector<QuakeValue> values{args...}, results;                          \
    for (auto &value : values) {                                               \
      results.emplace_back(NAME(value));                                       \
    }                                                                          \
    return results;                                                            \
  }

  CUDAQ_BUILDER_ADD_MEASURE(mx)
  CUDAQ_BUILDER_ADD_MEASURE(my)
  CUDAQ_BUILDER_ADD_MEASURE(mz)

  /// @brief SWAP operation for swapping the quantum states of qubits.
  /// Currently only support swaps between two qubits.
  void swap(const QuakeValue &first, const QuakeValue &second) {
    const std::vector<QuakeValue> empty;
    const std::vector<QuakeValue> &qubits{first, second};
    details::swap(*opBuilder, empty, qubits);
  }

  /// @brief Reset the given qubit or qubits.
  void reset(const QuakeValue &qubit) { details::reset(*opBuilder, qubit); }

  /// @brief Apply a conditional statement on a
  /// measure result, if true apply the `thenFunctor`.
  void c_if(QuakeValue result, std::function<void()> &&thenFunctor) {
    details::c_if(*opBuilder, result, thenFunctor);
  }

  /// @brief Apply a general pauli rotation, exp(i theta P),
  /// takes a QuakeValue representing a register of qubits.
  template <QuakeValueOrNumericType ParamT>
  void exp_pauli(const ParamT &theta, const QuakeValue &qubits,
                 const std::string &pauliWord) {
    std::vector<QuakeValue> qubitValues{qubits};
    if constexpr (std::is_floating_point_v<ParamT>)
      details::exp_pauli(*opBuilder, QuakeValue(*opBuilder, theta), qubitValues,
                         pauliWord);
    else
      details::exp_pauli(*opBuilder, theta, qubitValues, pauliWord);
  }

  /// @brief Apply a general pauli rotation, exp(i theta P),
  /// takes a variadic list of QuakeValues representing a individual qubits.
  template <QuakeValueOrNumericType ParamT, typename... QubitArgs>
  void exp_pauli(const ParamT &theta, const std::string &pauliWord,
                 QubitArgs &&...qubits) {
    std::vector<QuakeValue> qubitValues{qubits...};
    if constexpr (std::is_floating_point_v<ParamT>)
      details::exp_pauli(*opBuilder, QuakeValue(*opBuilder, theta), qubitValues,
                         pauliWord);
    else
      details::exp_pauli(*opBuilder, theta, qubitValues, pauliWord);
  }

  /// @brief Apply the given `otherKernel` with the provided `QuakeValue`
  /// arguments.
  template <typename OtherKernelBuilder>
  void call(OtherKernelBuilder &kernel, std::vector<QuakeValue> &values) {
    // This should work for regular c++ kernels too
    std::string name = "", quake = "";
    if constexpr (std::is_base_of_v<
                      details::kernel_builder_base,
                      std::remove_reference_t<OtherKernelBuilder>>) {
      name = kernel.name();
      quake = kernel.to_quake();
    } else {
      name = cudaq::getKernelName(kernel);
      quake = cudaq::get_quake_by_name(name);
    }
    details::call(*opBuilder, name, quake, values);
  }

  /// @brief Apply the given `otherKernel` with the provided `QuakeValue`
  /// arguments.
  template <typename OtherKernelBuilder, typename... QuakeValues>
    requires(AllAreQuakeValues<QuakeValues...>)
  void call(OtherKernelBuilder &&kernel, QuakeValues &...values) {
    // static_assert(kernel)
    std::vector<QuakeValue> vecValues{values...};
    call(kernel, vecValues);
  }

  /// @brief Apply the given kernel controlled on the provided qubit value.
  /// This overload takes a vector of `QuakeValue`s and is primarily meant
  /// to be used internally.
  template <typename OtherKernelBuilder>
  void control(OtherKernelBuilder &kernel, QuakeValue &control,
               std::vector<QuakeValue> &args) {
    std::string name = "", quake = "";
    if constexpr (std::is_base_of_v<
                      details::kernel_builder_base,
                      std::remove_reference_t<OtherKernelBuilder>>) {
      name = kernel.name();
      quake = kernel.to_quake();
    } else {
      name = cudaq::getKernelName(kernel);
      quake = cudaq::get_quake_by_name(name);
    }

    details::control(*opBuilder, name, quake, control, args);
  }

  /// @brief Apply the given kernel controlled on the provided qubit value.
  template <typename OtherKernelBuilder, typename... QuakeValues>
    requires(AllAreQuakeValues<QuakeValues...>)
  void control(OtherKernelBuilder &kernel, QuakeValue &ctrl,
               QuakeValues &...values) {
    std::vector<QuakeValue> vecValues{values...};
    control(kernel, ctrl, vecValues);
  }

  /// @brief Apply the adjoint of the given kernel.
  /// This overload takes a vector of `QuakeValue`s and is primarily meant
  /// to be used internally.
  template <typename OtherKernelBuilder>
  void adjoint(OtherKernelBuilder &kernel, std::vector<QuakeValue> &args) {
    std::string name = "", quake = "";
    if constexpr (std::is_base_of_v<
                      details::kernel_builder_base,
                      std::remove_reference_t<OtherKernelBuilder>>) {
      name = kernel.name();
      quake = kernel.to_quake();
    } else {
      name = cudaq::getKernelName(kernel);
      quake = cudaq::get_quake_by_name(name);
    }

    details::adjoint(*opBuilder, name, quake, args);
  }

  /// @brief Apply the adjoint of the given kernel.
  template <typename OtherKernelBuilder, typename... QuakeValues>
    requires(AllAreQuakeValues<QuakeValues...>)
  void adjoint(OtherKernelBuilder &kernel, QuakeValues &...values) {
    std::vector<QuakeValue> vecValues{values...};
    adjoint(kernel, vecValues);
  }

  /// @brief Apply the for loop with given start and end (non inclusive) indices
  /// that contains the instructions provided via the given body callable.
  template <typename StartType, typename EndType>
  void for_loop(StartType &&start, EndType &&end,
                std::function<void(QuakeValue &)> &&body) {
    details::forLoop(*opBuilder, start, end, body);
  }

  /// @brief Return the string representation of the quake code.
  std::string to_quake() const override {
    return details::to_quake(*opBuilder);
  }

  /// @brief Lower the Quake code to the LLVM Dialect, call
  /// `PassManager`.
  void jitCode(std::vector<std::string> extraLibPaths = {}) override {
    auto [wasChanged, ptr] =
        details::jitCode(*opBuilder, jitEngine.get(), jitEngineToModuleHash,
                         kernelName, extraLibPaths);
    // If we had a jitEngine, but the code changed,
    // delete the one we had.
    if (jitEngine && wasChanged)
      details::deleteJitEngine(jitEngine.release());

    // Store for the next time if we haven't already
    if (!jitEngine)
      jitEngine = std::unique_ptr<ExecutionEngine, void (*)(ExecutionEngine *)>(
          ptr, details::deleteJitEngine);
  }

  /// @brief Invoke JIT compilation and extract a function pointer and execute.
  void jitAndInvoke(void **argsArray,
                    std::vector<std::string> extraLibPaths = {}) {
    jitCode(extraLibPaths);
    details::invokeCode(*opBuilder, jitEngine.get(), kernelName, argsArray,
                        extraLibPaths);
  }

  /// @brief The call operator for the kernel_builder,
  /// takes as input the constructed function arguments.
  void operator()(Args... args) {
    [[maybe_unused]] std::size_t argCounter = 0;
    (details::ArgumentValidator<Args>::validate(argCounter, arguments, args),
     ...);
    void *argsArr[sizeof...(Args)] = {&args...};
    return operator()(argsArr);
  }

  /// @brief Call operator that takes an array of opaque pointers
  /// for the function arguments
  void operator()(void **argsArray) { jitAndInvoke(argsArray); }

  /// Expose the `get<N>()` method necessary for
  /// enabling structured bindings on a custom type
  template <std::size_t N>
  decltype(auto) get() {
    if constexpr (N == 0)
      return *this;
    else
      return arguments[N - 1];
  }
};
} // namespace cudaq

/// The following std functions are necessary to enable
/// structured bindings on the `kernel_builder` type.
/// e.g.
/// `auto [kernel, theta, phi] = std::make_kernel<double,double>();`
namespace std {

template <typename... Args>
struct tuple_size<cudaq::kernel_builder<Args...>>
    : std::integral_constant<std::size_t, sizeof...(Args) + 1> {};

template <std::size_t N, typename... Args>
struct tuple_element<N, cudaq::kernel_builder<Args...>> {
  using type = std::conditional_t<N == 0, cudaq::kernel_builder<Args...>,
                                  cudaq::QuakeValue>;
};

} // namespace std

namespace cudaq {

/// @brief Return a new kernel_builder that takes no arguments
inline auto make_kernel() {
  std::vector<details::KernelBuilderType> empty;
  return kernel_builder<>(empty);
}

/// Factory function for creating a new `kernel_builder` with specified argument
/// types. This requires programmers specify the concrete argument types of the
/// kernel being built. The return type is meant to be acquired via C++17
/// structured binding with the first element representing the builder, and the
/// remaining bound variables representing the kernel argument handles.
template <typename... Args>
CUDAQ_VALID_BUILDER_ARGS_FOLD()
auto make_kernel() {
  std::vector<details::KernelBuilderType> types;
  cudaq::tuple_for_each(std::tuple<Args...>(), [&](auto &&el) {
    types.push_back(details::mapArgToType(el));
  });
  return kernel_builder<Args...>(types);
}

} // namespace cudaq
