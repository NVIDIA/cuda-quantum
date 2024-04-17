/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/builder/QuakeValue.h"
#include "cudaq/host_config.h"
#include "cudaq/qis/modifiers.h"
#include "cudaq/qis/qvector.h"
#include "cudaq/utils/cudaq_utils.h"
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <variant>
#include <vector>

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

namespace cudaq {
std::string get_quake_by_name(const std::string &);

#if CUDAQ_USE_STD20
/// @brief Define a floating point concept
template <typename T>
concept NumericType = requires(T param) { std::is_floating_point_v<T>; };

/// @brief Define a Quake-`constructable` floating point value concept; i.e., it
/// could be a `QuakeValue` type or a floating point number (convertible
/// to a `QuakeValue` with `ConstantFloatOp`).
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

// If you want to add to the list of valid kernel argument types first add it
// here, then add `details::convertArgumentTypeToMLIR()` function
#define CUDAQ_VALID_BUILDER_ARGS_FOLD()                                        \
  requires(                                                                    \
      KernelBuilderArgTypeIsValid<                                             \
          Args, float, double, std::size_t, int, std::vector<int>,             \
          std::vector<float>, std::vector<std::size_t>, std::vector<double>,   \
          std::vector<std::complex<float>>, std::vector<std::complex<double>>, \
          std::vector<cudaq::simulation_scalar>, cudaq::qubit,                 \
          cudaq::qvector<>> &&                                                 \
      ...)
#else
// Not C++ 2020: stub these out.
#define QuakeValueOrNumericType typename
#define CUDAQ_VALID_BUILDER_ARGS_FOLD()
#endif

namespace details {
/// Use parametric type: `initializations` must be vectors of complex float or
/// double. No other type is allowed.
using StateVectorVariant = std::variant<std::vector<std::complex<float>> *,
                                        std::vector<std::complex<double>> *>;

/// Type describing user-provided state vector data. This is a list of the state
/// vector variables used in a kernel with at least one `qvector` with initial
/// state.
using StateVectorStorage = std::vector<StateVectorVariant>;

// Define a `mlir::Type` generator in the `cudaq` namespace, this helps us keep
// MLIR out of this public header

/// @brief The `kernel_builder::Type` allows us to track input C++ types
/// representing the quake function argument types in a way that does not expose
/// MLIR Type to the CUDA Quantum code. This type keeps track of a functor that
/// generates the MLIR Type in implementation code when create() is invoked.
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
KernelBuilderType convertArgumentTypeToMLIR(double &e);

/// Map a `float` to a `KernelBuilderType`
KernelBuilderType convertArgumentTypeToMLIR(float &e);

/// Map a `int` to a `KernelBuilderType`
KernelBuilderType convertArgumentTypeToMLIR(int &e);

/// Map a `size_t` to a `KernelBuilderType`
KernelBuilderType convertArgumentTypeToMLIR(std::size_t &e);

/// Map a `std::vector<int>` to a `KernelBuilderType`
KernelBuilderType convertArgumentTypeToMLIR(std::vector<int> &e);

/// Map a `std::vector<std::size_t>` to a `KernelBuilderType`
KernelBuilderType convertArgumentTypeToMLIR(std::vector<std::size_t> &e);

/// Map a `std::vector<float>` to a `KernelBuilderType`
KernelBuilderType convertArgumentTypeToMLIR(std::vector<float> &e);

/// Map a `vector<double>` to a `KernelBuilderType`
KernelBuilderType convertArgumentTypeToMLIR(std::vector<double> &e);

/// Map a `vector<std::complex<double>>` to a `KernelBuilderType`
KernelBuilderType
convertArgumentTypeToMLIR(std::vector<std::complex<double>> &e);

/// Map a `vector<std::complex<double>>` to a `KernelBuilderType`
KernelBuilderType
convertArgumentTypeToMLIR(std::vector<std::complex<float>> &e);

/// Map a `qubit` to a `KernelBuilderType`
KernelBuilderType convertArgumentTypeToMLIR(cudaq::qubit &e);

/// @brief  Map a `qvector` to a `KernelBuilderType`
KernelBuilderType convertArgumentTypeToMLIR(cudaq::qvector<> &e);

/// @brief Initialize the `MLIRContext`, return the raw pointer which we'll wrap
/// in an `unique_ptr`.
mlir::MLIRContext *initializeContext();

/// @brief Delete function for the context pointer, also given to the
/// `unique_ptr`
void deleteContext(mlir::MLIRContext *);

/// @brief Initialize the `OpBuilder`, return the raw pointer which we'll wrap
/// in an `unique_ptr`.
mlir::ImplicitLocOpBuilder *initializeBuilder(mlir::MLIRContext *,
                                              std::vector<KernelBuilderType> &,
                                              std::vector<QuakeValue> &,
                                              std::string &kernelName);

/// @brief Delete function for the builder pointer, also given to the
/// `unique_ptr`
void deleteBuilder(mlir::ImplicitLocOpBuilder *builder);

/// @brief Delete function for the JIT pointer, also given to the `unique_ptr`
void deleteJitEngine(mlir::ExecutionEngine *jit);

/// @brief Allocate a single `qubit`
QuakeValue qalloc(mlir::ImplicitLocOpBuilder &builder);

/// @brief Allocate a `qvector`.
QuakeValue qalloc(mlir::ImplicitLocOpBuilder &builder,
                  const std::size_t nQubits);

/// @brief Allocate a `qvector` from existing `QuakeValue` size
QuakeValue qalloc(mlir::ImplicitLocOpBuilder &builder, QuakeValue &size);

/// @brief Allocate a `qvector` from a user provided state vector.
QuakeValue qalloc(mlir::ImplicitLocOpBuilder &builder,
                  StateVectorStorage &stateVectorData,
                  StateVectorVariant &&state, simulation_precision precision);

/// @brief Create a QuakeValue representing a constant floating-point number
QuakeValue constantVal(mlir::ImplicitLocOpBuilder &builder, double val);

// In the following macros + instantiations, we define the functions that create
// Quake Quantum Ops + Measures

#define CUDAQ_DETAILS_QIS_DECLARATION(NAME)                                    \
  void NAME(mlir::ImplicitLocOpBuilder &builder,                               \
            std::vector<QuakeValue> &ctrls, const QuakeValue &target,          \
            bool adjoint = false);

CUDAQ_DETAILS_QIS_DECLARATION(h)
CUDAQ_DETAILS_QIS_DECLARATION(s)
CUDAQ_DETAILS_QIS_DECLARATION(t)
CUDAQ_DETAILS_QIS_DECLARATION(x)
CUDAQ_DETAILS_QIS_DECLARATION(y)
CUDAQ_DETAILS_QIS_DECLARATION(z)

#define CUDAQ_DETAILS_ONEPARAM_QIS_DECLARATION(NAME)                           \
  void NAME(mlir::ImplicitLocOpBuilder &builder, QuakeValue &parameter,        \
            std::vector<QuakeValue> &ctrls, QuakeValue &target);               \
  void NAME(mlir::ImplicitLocOpBuilder &builder, double &parameter,            \
            std::vector<QuakeValue> &ctrls, QuakeValue &target);

CUDAQ_DETAILS_ONEPARAM_QIS_DECLARATION(rx)
CUDAQ_DETAILS_ONEPARAM_QIS_DECLARATION(ry)
CUDAQ_DETAILS_ONEPARAM_QIS_DECLARATION(rz)
CUDAQ_DETAILS_ONEPARAM_QIS_DECLARATION(r1)

#define CUDAQ_DETAILS_MEASURE_DECLARATION(NAME)                                \
  QuakeValue NAME(mlir::ImplicitLocOpBuilder &builder, QuakeValue &target,     \
                  std::string regName = "");

CUDAQ_DETAILS_MEASURE_DECLARATION(mx)
CUDAQ_DETAILS_MEASURE_DECLARATION(my)
CUDAQ_DETAILS_MEASURE_DECLARATION(mz)

void exp_pauli(mlir::ImplicitLocOpBuilder &builder, const QuakeValue &theta,
               const std::vector<QuakeValue> &qubits,
               const std::string &pauliWord);

void swap(mlir::ImplicitLocOpBuilder &builder,
          const std::vector<QuakeValue> &ctrls,
          const std::vector<QuakeValue> &targets, bool adjoint = false);

void reset(mlir::ImplicitLocOpBuilder &builder, const QuakeValue &qubitOrQvec);

void c_if(mlir::ImplicitLocOpBuilder &builder, QuakeValue &conditional,
          std::function<void()> &thenFunctor);

/// @brief Return the name of this `kernel_builder`, it is also the name of the
/// function
std::string name(std::string_view kernelName);

/// @brief Apply our MLIR passes before JIT execution
void applyPasses(mlir::PassManager &);

/// @brief Create the `ExecutionEngine` and return a raw pointer, which we will
/// wrap in a `unique_ptr`
std::tuple<bool, mlir::ExecutionEngine *>
jitCode(mlir::ImplicitLocOpBuilder &, mlir::ExecutionEngine *,
        std::unordered_map<mlir::ExecutionEngine *, std::size_t> &, std::string,
        std::vector<std::string>, StateVectorStorage &);

/// @brief Invoke the function with the given kernel name.
void invokeCode(mlir::ImplicitLocOpBuilder &builder, mlir::ExecutionEngine *jit,
                std::string kernelName, void **argsArray,
                std::vector<std::string> extraLibPaths,
                StateVectorStorage &storage);

/// @brief Invoke the provided kernel function.
void call(mlir::ImplicitLocOpBuilder &builder, std::string &name,
          std::string &quakeCode, std::vector<QuakeValue> &values);

/// @brief Apply the given kernel controlled on the provided qubit value.
void control(mlir::ImplicitLocOpBuilder &builder, std::string &name,
             std::string &quakeCode, QuakeValue &control,
             std::vector<QuakeValue> &values);

/// @brief Apply the adjoint of the given kernel
void adjoint(mlir::ImplicitLocOpBuilder &builder, std::string &name,
             std::string &quakeCode, std::vector<QuakeValue> &values);

/// @brief Add a for loop that starts from the given `start` integer index, ends
/// at the given `end` integer index, and applies the given `body` as a callable
/// function. This callable function must take as input an index variable that
/// can be used within the body.
void forLoop(mlir::ImplicitLocOpBuilder &builder, std::size_t start,
             std::size_t end, std::function<void(QuakeValue &)> &body);

/// @brief Add a for loop that starts from the given `start` integer index, ends
/// at the given `end` index, and applies the given `body` as a callable
/// function. This callable function must take as input an index variable that
/// can be used within the body.
void forLoop(mlir::ImplicitLocOpBuilder &builder, std::size_t start,
             QuakeValue &end, std::function<void(QuakeValue &)> &body);

/// @brief Add a for loop that starts from the given `start` index, ends at the
/// given `end` integer index, and applies the given `body` as a callable
/// function. This callable function must take as input an index variable that
/// can be used within the body.
void forLoop(mlir::ImplicitLocOpBuilder &builder, QuakeValue &start,
             std::size_t end, std::function<void(QuakeValue &)> &body);

/// @brief Add a for loop that starts from the given `start` index, ends at the
/// given `end` index, and applies the given `body` as a callable function. This
/// callable function must take as input an index variable that can be used
/// within the body.
void forLoop(mlir::ImplicitLocOpBuilder &builder, QuakeValue &start,
             QuakeValue &end, std::function<void(QuakeValue &)> &body);

/// @brief Return the quake representation as a string
std::string to_quake(mlir::ImplicitLocOpBuilder &builder);

/// @brief Returns `true` if the argument to the `kernel_builder` is a
/// `cc::StdvecType`. Returns `false` otherwise.
bool isArgStdVec(std::vector<QuakeValue> &args, std::size_t idx);

/// @brief The `ArgumentValidator` provides a way validate the input arguments
/// when the kernel is invoked (via a fold expression).
template <typename T>
struct ArgumentValidator {
  static void validate(std::size_t &argCounter, std::vector<QuakeValue> &args,
                       T &val) {
    // Default case, do nothing for now
    argCounter++;
  }
};

/// @brief The `ArgumentValidator` provides a way validate the input arguments
/// when the kernel is invoked (via a fold expression). Here we explicitly
/// validate `std::vector<T>` and its size.
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

    // Validate the input vector<T> if possible. If getRequiredElements()
    // returns 0, any size vector is ok.
    auto nRequiredElements = arg.getRequiredElements();
    if (nRequiredElements && arg.canValidateNumElements() &&
        input.size() < nRequiredElements)
      throw std::runtime_error(
          "Invalid vector<T> input. Number of elements provided (" +
          std::to_string(input.size()) + ") != number of elements required (" +
          std::to_string(nRequiredElements) + ").\n");
  }
};

/// @brief The `kernel_builder_base` provides a base type for the templated
/// kernel builder so that we can get a single handle on an instance within the
/// runtime.
class kernel_builder_base {
public:
  virtual std::string to_quake() const = 0;
  virtual void jitCode(std::vector<std::string> extraLibPaths = {}) = 0;
  virtual ~kernel_builder_base() = default;

  /// @brief Write the kernel_builder to the given output stream. This outputs
  /// the Quake representation.
  friend std::ostream &operator<<(std::ostream &stream,
                                  const kernel_builder_base &builder);
};

} // namespace details

#if CUDAQ_USE_STD20
template <class... Ts>
concept AllAreQuakeValues =
    sizeof...(Ts) < 2 ||
    (std::conjunction_v<
         std::is_same<std::tuple_element_t<0, std::tuple<Ts...>>, Ts>...> &&
     std::is_same_v<
         std::remove_reference_t<std::tuple_element<0, std::tuple<Ts...>>>,
         QuakeValue>);
#endif

template <typename... Args>
class kernel_builder : public details::kernel_builder_base {
private:
  /// @brief Handle to the MLIR Context, stored as a pointer here to keep
  /// implementation details out of CUDA Quantum code
  std::unique_ptr<mlir::MLIRContext, void (*)(mlir::MLIRContext *)> context;

  /// @brief Handle to the MLIR `OpBuilder`, stored as a pointer here to keep
  /// implementation details out of CUDA Quantum code
  std::unique_ptr<mlir::ImplicitLocOpBuilder,
                  void (*)(mlir::ImplicitLocOpBuilder *)>
      opBuilder;

  /// @brief Handle to the MLIR `ExecutionEngine`, stored as a pointer here to
  /// keep implementation details out of CUDA Quantum code
  std::unique_ptr<mlir::ExecutionEngine, void (*)(mlir::ExecutionEngine *)>
      jitEngine;

  /// @brief Map created ExecutionEngines to a unique hash of the
  /// ModuleOp they derive from.
  std::unordered_map<mlir::ExecutionEngine *, std::size_t>
      jitEngineToModuleHash;

  /// @brief Name of the CUDA Quantum kernel Quake function
  std::string kernelName = "__nvqpp__mlirgen____nvqppBuilderKernel";

  /// @brief The CUDA Quantum Quake function arguments stored as `QuakeValue`s.
  std::vector<QuakeValue> arguments;

  /// @brief Return a string representation of the given spin operator.
  /// Throw an exception if the spin operator provided is not a single term.
  auto toPauliWord(const std::variant<std::string, spin_op> &term) {
    if (term.index()) {
      auto op = std::get<spin_op>(term);
      if (op.num_terms() > 1)
        throw std::runtime_error(
            "exp_pauli requires a spin operator with a single term.");
      return op.to_string(false);
    }
    return std::get<std::string>(term);
  }

  /// @brief Storage for any user-provided state-vector data.
  details::StateVectorStorage stateVectorStorage;

public:
  /// @brief The constructor, takes the input `KernelBuilderType`s which is
  /// used to create the MLIR function type
  kernel_builder(std::vector<details::KernelBuilderType> &types)
      : context(details::initializeContext(), details::deleteContext),
        opBuilder(nullptr, [](mlir::ImplicitLocOpBuilder *) {}),
        jitEngine(nullptr, [](mlir::ExecutionEngine *) {}) {
    auto *ptr =
        details::initializeBuilder(context.get(), types, arguments, kernelName);
    opBuilder = std::unique_ptr<mlir::ImplicitLocOpBuilder,
                                void (*)(mlir::ImplicitLocOpBuilder *)>(
        ptr, details::deleteBuilder);
  }

  /// @brief Return the `QuakeValue` arguments
  auto &getArguments() { return arguments; }

  /// @brief Return `true` if the argument to the kernel is a `std::vector`,
  /// `false` otherwise.
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

  /// @brief Return a `QuakeValue` representing the allocated `Veq`, size is
  /// from a pre-allocated size `QuakeValue` or `BlockArgument`.
  QuakeValue qalloc(QuakeValue size) {
    return details::qalloc(*opBuilder.get(), size);
  }

  /// Return a `QuakeValue` representing the allocated quantum register,
  /// initialized to the given state vector, \p state.
  ///
  /// Note: input argument is a \e true reference here, the calling context has
  /// to own the data. Specifically, the builder object will capture variables
  /// by reference (implemented as a container of pointers for simplicity) but
  /// the builder does not create, own, or copy these vectors. This implies that
  /// if the captured vector goes out of scope before the kernel is invoked, the
  /// reference may contain garbage. This behavior is identical to a C++ lambda
  /// capture by reference.
  QuakeValue qalloc(std::vector<std::complex<double>> &state) {
    return details::qalloc(*opBuilder.get(), stateVectorStorage,
                           details::StateVectorVariant{&state},
                           simulation_precision::fp64);
  }
  // Overload for complex<float> vector.
  QuakeValue qalloc(std::vector<std::complex<float>> &state) {
    return details::qalloc(*opBuilder.get(), stateVectorStorage,
                           details::StateVectorVariant{&state},
                           simulation_precision::fp32);
  }

  /// @brief Return a `QuakeValue` representing the constant floating-point
  /// value.
  QuakeValue constantVal(double val) {
    return details::constantVal(*opBuilder.get(), val);
  }

  // In the following macros + instantiations, we define the kernel_builder
  // methods that create Quake Quantum Ops + Measures

#define CUDAQ_BUILDER_ADD_ONE_QUBIT_OP(NAME)                                   \
  void NAME(QuakeValue &qubit) {                                               \
    std::vector<QuakeValue> empty;                                             \
    details::NAME(*opBuilder, empty, qubit);                                   \
  }                                                                            \
  void NAME(QuakeValue &&qubit) { NAME(qubit); }                               \
  [[deprecated("In the future, passing `ctrls` to " #NAME                      \
               " will require an explicit `<cudaq::ctrl>` template argument. " \
               "Upon the next release, this will be interpreted as a single "  \
               "qubit gate broadcast across all input qubits, per the CUDA "   \
               "Quantum Specification.")]] void                                \
  NAME(std::vector<QuakeValue> &ctrls, QuakeValue &target) {                   \
    details::NAME(*opBuilder, ctrls, target);                                  \
  }                                                                            \
  template <typename mod,                                                      \
            typename =                                                         \
                typename std::enable_if_t<std::is_same_v<mod, cudaq::ctrl>>>   \
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
      NAME<cudaq::ctrl>(ctrls, target);                                        \
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
  [[deprecated("In the future, passing `ctrls` to " #NAME                      \
               " will require an explicit `<cudaq::ctrl>` template argument. " \
               "Upon the next release, this will be interpreted as a single "  \
               "qubit gate broadcast across all input qubits, per the CUDA "   \
               "Quantum Specification.")]] void                                \
  NAME(QuakeValue parameter, std::vector<QuakeValue> &ctrls,                   \
       QuakeValue &target) {                                                   \
    details::NAME(*opBuilder, parameter, ctrls, target);                       \
  }                                                                            \
  template <typename mod,                                                      \
            typename =                                                         \
                typename std::enable_if_t<std::is_same_v<mod, cudaq::ctrl>>>   \
  void NAME(QuakeValue parameter, std::vector<QuakeValue> &ctrls,              \
            QuakeValue &target) {                                              \
    details::NAME(*opBuilder, parameter, ctrls, target);                       \
  }                                                                            \
  [[deprecated("In the future, passing `ctrls` to " #NAME                      \
               " will require an explicit `<cudaq::ctrl>` template argument. " \
               "Upon the next release, this will be interpreted as a single "  \
               "qubit gate broadcast across all input qubits, per the CUDA "   \
               "Quantum Specification.")]] void                                \
  NAME(double parameter, std::vector<QuakeValue> &ctrls, QuakeValue &target) { \
    QuakeValue v(*opBuilder, parameter);                                       \
    details::NAME(*opBuilder, v, ctrls, target);                               \
  }                                                                            \
  template <typename mod,                                                      \
            typename =                                                         \
                typename std::enable_if_t<std::is_same_v<mod, cudaq::ctrl>>>   \
  void NAME(double parameter, std::vector<QuakeValue> &ctrls,                  \
            QuakeValue &target) {                                              \
    QuakeValue v(*opBuilder, parameter);                                       \
    details::NAME(*opBuilder, v, ctrls, target);                               \
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
        NAME<cudaq::ctrl>(QuakeValue(*opBuilder, parameter), ctrls, target);   \
      else                                                                     \
        NAME<cudaq::ctrl>(parameter, ctrls, target);                           \
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

  /// @brief SWAP operation for swapping the quantum states of two qubits.
  void swap(const QuakeValue &first, const QuakeValue &second) {
    const std::vector<QuakeValue> empty;
    const std::vector<QuakeValue> &qubits{first, second};
    details::swap(*opBuilder, empty, qubits);
  }

  /// @brief SWAP operation for performing a Fredkin gate between two qubits,
  /// based on the state of input `control` qubit/s.
  template <typename mod, typename = typename std::enable_if_t<
                              std::is_same_v<mod, cudaq::ctrl>>>
  void swap(const QuakeValue &control, const QuakeValue &first,
            const QuakeValue &second) {
    const std::vector<QuakeValue> ctrl{control};
    const std::vector<QuakeValue> targets{first, second};
    details::swap(*opBuilder, ctrl, targets);
  }

  /// @brief SWAP operation for performing a Fredkin gate between two qubits,
  /// based on the state of an input vector of `controls`.
  template <typename mod, typename = typename std::enable_if_t<
                              std::is_same_v<mod, cudaq::ctrl>>>
  void swap(const std::vector<QuakeValue> &controls, const QuakeValue &first,
            const QuakeValue &second) {
    const std::vector<QuakeValue> targets{first, second};
    details::swap(*opBuilder, controls, targets);
  }

  /// @brief SWAP operation for performing a Fredkin gate between two qubits,
  /// based on the state of a variadic input of control qubits and registers.
  /// Note: the final two qubits in the variadic list will always be the qubits
  /// that undergo a SWAP. This requires >=3 qubits in the arguments.
  template <
      typename mod, typename... QubitValues,
      typename = typename std::enable_if_t<sizeof...(QubitValues) >= 3>,
      typename = typename std::enable_if_t<std::is_same_v<mod, cudaq::ctrl>>>
  void swap(QubitValues... args) {
    std::vector<QuakeValue> values{args...};
    // Up until the last two arguments will be our controls.
    const std::vector<QuakeValue> controls(values.begin(), values.end() - 2);
    // The last two args will be the two qubits to swap.
    const std::vector<QuakeValue> targets(values.end() - 2, values.end());
    details::swap(*opBuilder, controls, targets);
  }

  /// @brief Reset the given qubit or qubits.
  void reset(const QuakeValue &qubit) { details::reset(*opBuilder, qubit); }

  /// @brief Apply a conditional statement on a measure result, if true apply
  /// the `thenFunctor`.
  void c_if(QuakeValue result, std::function<void()> &&thenFunctor) {
    details::c_if(*opBuilder, result, thenFunctor);
  }

  /// @brief Apply a general Pauli rotation, exp(i theta P), takes a QuakeValue
  /// representing a register of qubits.
  template <QuakeValueOrNumericType ParamT>
  void exp_pauli(const ParamT &theta, const QuakeValue &qubits,
                 const std::variant<std::string, spin_op> &op) {
    auto pauliWord = toPauliWord(op);
    std::vector<QuakeValue> qubitValues{qubits};
    if constexpr (std::is_floating_point_v<ParamT>)
      details::exp_pauli(*opBuilder, QuakeValue(*opBuilder, theta), qubitValues,
                         pauliWord);
    else
      details::exp_pauli(*opBuilder, theta, qubitValues, pauliWord);
  }

  /// @brief Apply a general Pauli rotation, exp(i theta P), takes a variadic
  /// list of QuakeValues representing a individual qubits.
  template <QuakeValueOrNumericType ParamT, typename... QubitArgs>
  void exp_pauli(const ParamT &theta,
                 const std::variant<std::string, spin_op> &op,
                 QubitArgs &&...qubits) {
    auto pauliWord = toPauliWord(op);
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
#if CUDAQ_USE_STD20
  template <typename OtherKernelBuilder, typename... QuakeValues>
    requires AllAreQuakeValues<QuakeValues...>
#else
  template <typename OtherKernelBuilder, typename... QuakeValues,
            typename = std::enable_if_t<std::conjunction_v<std::is_same<
                std::remove_reference_t<QuakeValues>, cudaq::QuakeValue>...>>>
#endif
  void call(OtherKernelBuilder &&kernel, QuakeValues &...values) {
    // static_assert(kernel)
    std::vector<QuakeValue> vecValues{values...};
    call(kernel, vecValues);
  }

  /// @brief Apply the given kernel controlled on the provided qubit value. This
  /// overload takes a vector of `QuakeValue`s and is primarily meant to be used
  /// internally.
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
#if CUDAQ_USE_STD20
  template <typename OtherKernelBuilder, typename... QuakeValues>
    requires AllAreQuakeValues<QuakeValues...>
#else
  template <typename OtherKernelBuilder, typename... QuakeValues,
            typename = std::enable_if_t<std::conjunction_v<std::is_same<
                std::remove_reference_t<QuakeValues>, cudaq::QuakeValue>...>>>
#endif
  void control(OtherKernelBuilder &kernel, QuakeValue &ctrl,
               QuakeValues &...values) {
    std::vector<QuakeValue> vecValues{values...};
    control(kernel, ctrl, vecValues);
  }

  /// @brief Apply the adjoint of the given kernel. This overload takes a vector
  /// of `QuakeValue`s and is primarily meant to be used internally.
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
#if CUDAQ_USE_STD20
  template <typename OtherKernelBuilder, typename... QuakeValues>
    requires AllAreQuakeValues<QuakeValues...>
#else
  template <typename OtherKernelBuilder, typename... QuakeValues,
            typename = std::enable_if_t<std::conjunction_v<std::is_same<
                std::remove_reference_t<QuakeValues>, cudaq::QuakeValue>...>>>
#endif
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

  /// @brief Lower the Quake code to the LLVM Dialect, call `PassManager`.
  void jitCode(std::vector<std::string> extraLibPaths = {}) override {
    auto [wasChanged, ptr] =
        details::jitCode(*opBuilder, jitEngine.get(), jitEngineToModuleHash,
                         kernelName, extraLibPaths, stateVectorStorage);
    // If we had a jitEngine, but the code changed, delete the one we had.
    if (jitEngine && wasChanged)
      details::deleteJitEngine(jitEngine.release());

    // Store for the next time if we haven't already
    if (!jitEngine)
      jitEngine = std::unique_ptr<mlir::ExecutionEngine,
                                  void (*)(mlir::ExecutionEngine *)>(
          ptr, details::deleteJitEngine);
  }

  /// @brief Invoke JIT compilation and extract a function pointer and execute.
  void jitAndInvoke(void **argsArray,
                    std::vector<std::string> extraLibPaths = {}) {
    static std::mutex jitMutex;
    {
      std::scoped_lock<std::mutex> lock(jitMutex);
      // Scoped locking since jitCode is not thread-safe while this jitAndInvoke
      // can be invoked by kernel_builder::operator()(Args... args) in a
      // multi-threaded context.
      jitCode(extraLibPaths);
    }
    details::invokeCode(*opBuilder, jitEngine.get(), kernelName, argsArray,
                        extraLibPaths, stateVectorStorage);
  }

  /// @brief The call operator for the kernel_builder, takes as input the
  /// constructed function arguments.
  void operator()(Args... args) {
    [[maybe_unused]] std::size_t argCounter = 0;
    (details::ArgumentValidator<Args>::validate(argCounter, arguments, args),
     ...);
    void *argsArr[sizeof...(Args)] = {&args...};
    return operator()(argsArr);
  }

  /// @brief Call operator that takes an array of opaque pointers for the
  /// function arguments
  void operator()(void **argsArray) { jitAndInvoke(argsArray); }

  /// Expose the `get<N>()` method necessary for enabling structured bindings on
  /// a custom type
  template <std::size_t N>
  decltype(auto) get() {
    if constexpr (N == 0)
      return *this;
    else
      return arguments[N - 1];
  }
};
} // namespace cudaq

/// The following std functions are necessary to enable structured bindings on
/// the `kernel_builder` type.
/// e.g. `auto [kernel, theta, phi] = std::make_kernel<double,double>();`
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
    types.push_back(details::convertArgumentTypeToMLIR(el));
  });
  return kernel_builder<Args...>(types);
}

} // namespace cudaq
