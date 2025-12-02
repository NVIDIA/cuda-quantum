/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file
/// @brief CUDA-Q quantum compiler interface and kernel compilation system
/// @details Provides interfaces for compiling quantum kernels into
/// device-specific control programs for quantum processing units. Supports
/// multiple quantum devices and resource management for compiled kernels.

#pragma once

#include "utils/extension_point.h"

#include "device.h"

/// @brief CUDA-Q quantum compiler and kernel management namespace
/// @details Contains interfaces and implementations for compiling quantum
/// kernels into executable control programs for quantum processing units.
namespace cudaq::nvqlink {

/// @brief Internal implementation details for quantum kernel compilation
namespace details {
/// @brief Removes non-entrypoint functions from MLIR code manually
/// @param mlirCode The input MLIR code string to process
/// @return Processed MLIR code with non-entrypoint functions removed
/// @details This function manually parses and filters MLIR code to retain only
/// the quantum kernel entrypoint functions, removing auxiliary functions that
/// are not needed for device execution.
std::string removeNonEntrypointFunctionsManual(const std::string &mlirCode);
} // namespace details

/// @brief Binary program representation for quantum control systems
/// @details Encapsulates a compiled quantum program as binary data along with
/// the target quantum device identifier. Used to store device-specific compiled
/// quantum kernels ready for execution.
struct qcontrol_program {
  /// @brief Compiled binary program data
  /// @details Raw binary representation of the quantum program compiled for
  /// the specific quantum control system architecture.
  std::vector<std::byte> binary;

  /// @brief Target quantum device identifier
  /// @details Specifies which quantum processing unit this program is compiled
  /// for. Must match the device ID when executing the program.
  std::size_t qdevice_id;
};

/// @brief Compiled quantum kernel with associated control programs and
/// resources
/// @details Represents a fully compiled quantum kernel that can be executed on
/// one or more quantum devices. Manages compiled binary programs and tracks
/// associated resources that need cleanup when the kernel is destroyed.
class compiled_kernel {
protected:
  /// @brief Name of the original quantum kernel
  /// @details Stores the kernel identifier used during compilation for
  /// debugging and identification purposes.
  std::string kernel_name;

  /// @brief Collection of device-specific control programs
  /// @details Each program in this vector corresponds to a different quantum
  /// device or quantum control system that can execute this kernel.
  std::vector<qcontrol_program> control_programs;

  /// @brief Resource tracking for automatic cleanup
  /// @details Maps resource pointers to their corresponding deleter functions.
  /// When the compiled_kernel is destroyed, all tracked resources are properly
  /// cleaned up using their associated deleter functions.
  std::unordered_map<void *, std::function<void(void *)>> tracked_resources;

public:
  /// @brief Constructs a compiled kernel with programs and tracked resources
  /// @param name The name of the quantum kernel
  /// @param programs Vector of compiled control programs for different devices
  /// @param tr Map of resources to their deleter functions for cleanup
  /// @details Creates a compiled kernel instance with all necessary data for
  /// execution on quantum devices. Takes ownership of the tracked resources.
  compiled_kernel(
      std::string name, std::vector<qcontrol_program> programs,
      const std::unordered_map<void *, std::function<void(void *)>> &tr)
      : kernel_name(name), control_programs(programs), tracked_resources(tr) {}

  /// @brief Destructor that cleans up all tracked resources
  /// @details Automatically calls the deleter function for each tracked
  /// resource to ensure proper cleanup when the compiled kernel goes out of
  /// scope.
  ~compiled_kernel() {
    for (auto &[resource, deleter] : tracked_resources)
      deleter(resource);
  }

  /// @brief Gets the kernel name
  /// @return Const reference to the kernel name string
  /// @details Returns the identifier of the original quantum kernel that was
  /// compiled.
  const std::string &name() const { return kernel_name; }

  /// @brief Gets the compiled control programs
  /// @return Const reference to the vector of control programs
  /// @details Returns all device-specific binary programs that were generated
  /// during compilation of this quantum kernel.
  const std::vector<qcontrol_program> &get_programs() const {
    return control_programs;
  };
};

/// @brief Abstract base class for quantum kernel compilers
/// @details Defines the interface that all quantum compiler implementations
/// must provide. Uses the extension point pattern to allow pluggable compiler
/// backends for different quantum architectures and programming models.
class compiler : public cudaqx::extension_point<compiler> {
public:
  /// @brief Virtual destructor for proper cleanup of derived classes
  virtual ~compiler() = default;

  /// @brief Determines if this compiler can handle the given code
  /// @param code The quantum kernel code to analyze
  /// @return True if this compiler understands and can compile the code, false
  /// otherwise
  /// @details Each compiler implementation should examine the code format,
  /// language, or other characteristics to determine compatibility. This
  /// enables automatic selection of the appropriate compiler for different code
  /// types.
  virtual bool understands_code(const std::string &code) const = 0;

  /// @brief Compiles quantum kernel code into device-specific control programs
  /// @param code The source code of the quantum kernel to compile
  /// @param kernel_name The name identifier for the kernel
  /// @param num_qcs_devices The number of quantum control system devices to
  /// target
  /// @return Unique pointer to the compiled kernel with all control programs
  /// @details Performs the full compilation pipeline from source code to
  /// executable binary programs for the specified number of quantum devices.
  /// The returned compiled_kernel contains all necessary data for execution.
  virtual std::unique_ptr<compiled_kernel>
  compile(const std::string &code, const std::string &kernel_name,
          std::size_t num_qcs_devices) = 0;
};

/// @brief Extracts MLIR code and kernel name from a quantum kernel callable
/// @tparam QuantumKernel Type of the quantum kernel callable object
/// @param kernel The quantum kernel object to extract information from
/// @return Tuple containing the processed MLIR code and kernel name
/// @details This template function works with any quantum kernel type to
/// extract the underlying MLIR representation and kernel identifier. The MLIR
/// code is processed to remove non-entrypoint functions, preparing it for
/// compilation.
/// @note The kernel parameter is forwarded to preserve value category and
/// enable perfect forwarding for both lvalue and rvalue kernel objects.
// template <typename QuantumKernel>
// std::tuple<std::string, std::string> extract_code(QuantumKernel &&kernel) {
//   std::string kernelName{
//       cudaq::details::getKernelName(std::forward<QuantumKernel>(kernel))};
//   auto code = details::removeNonEntrypointFunctionsManual(
//       cudaq::get_quake_by_name(kernelName));
//   return std::make_tuple(code, kernelName);
// }

} // namespace cudaq::nvqlink
