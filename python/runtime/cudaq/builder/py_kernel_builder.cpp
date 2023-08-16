/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <pybind11/stl.h>

#include "py_kernel_builder.h"
#include "utils/OpaqueArguments.h"

#include "cudaq/builder/kernel_builder.h"
#include "cudaq/platform.h"

#include "common/ExecutionContext.h"
#include "common/MeasureCounts.h"

#include <any>

namespace cudaq {
struct PyQubit {};
struct PyQreg {};

void bindMakeKernel(py::module &mod) {

  py::class_<PyQubit>(mod, "qubit",
                      "The data-type representing a qubit argument to a "
                      ":class:`Kernel` function.\n"
                      "\n.. code-block:: python\n\n"
                      "  # Example:\n"
                      "  kernel, qubit = cudaq.make_kernel(cudaq.qubit)\n");
  py::class_<PyQreg>(mod, "qreg",
                     "The data-type representing a register of qubits as an "
                     "argument to a :class:`Kernel` function.\n"
                     "\n.. code-block:: python\n\n"
                     "  # Example:\n"
                     "  kernel, qreg = cudaq.make_kernel(cudaq.qreg)\n");

  mod.def(
      "make_kernel",
      []() {
        std::vector<details::KernelBuilderType> empty;
        return std::make_unique<kernel_builder<>>(empty);
      },
      "Create and return a :class:`Kernel` that accepts no arguments.\n"
      "\nReturns:\n"
      "  :class:`Kernel` : An empty kernel function to be used for quantum "
      "program construction."
      "  This kernel is non-parameterized and accepts no arguments.\n"
      "\n.. code-block:: python\n\n"
      "  # Example:\n"
      "  # Non-parameterized kernel.\n"
      "  kernel = cudaq.make_kernel()\n");

  mod.def(
      "make_kernel",
      [](py::args arguments) {
        // Transform the py::args (which must be py::types) into
        // our KernelBuilderType
        std::vector<details::KernelBuilderType> types;
        std::transform(
            arguments.begin(), arguments.end(), std::back_inserter(types),
            [&](auto &&arguments) {
              auto name =
                  arguments.attr("__name__").template cast<std::string>();
              if (name == "float") {
                double tmp = 0.0;
                return details::mapArgToType(tmp);
              } else if (name == "int") {
                int tmp = 0;
                return details::mapArgToType(tmp);
              } else if (name == "list" || name == "List") {
                std::vector<double> tmp;
                return details::mapArgToType(tmp);
              } else if (name == "qubit") {
                cudaq::qubit q;
                return details::mapArgToType(q);
              } else if (name == "qreg") {
                cudaq::qreg<cudaq::dyn, 2> q;
                return details::mapArgToType(q);
              } else
                throw std::runtime_error("Invalid builder parameter type (must "
                                         "be a int, float, list/List,"
                                         "`cudaq.qubit`, or `cudaq.qreg`).");
            });

        auto kernelBuilder = std::make_unique<kernel_builder<>>(types);
        auto &quakeValues = kernelBuilder->getArguments();
        auto ret = py::tuple(quakeValues.size() + 1);
        ret[0] = std::move(kernelBuilder);
        for (std::size_t i = 0; i < quakeValues.size(); i++) {
          ret[i + 1] = quakeValues[i];
        }
        return ret;
      },
      "Create a :class:`Kernel` that takes the provided types as arguments. "
      "Returns a tuple containing the kernel and a :class:`QuakeValue` for "
      "each kernel argument.\n"
      "\nNote:\n"
      "  The following types are supported as kernel arguments: `int`, "
      "`float`, `list`/`List`, `cudaq.qubit`, or `cudaq.qreg`.\n"
      "\nArgs:\n"
      "  *arguments : A variable amount of types for the kernel function to "
      "accept as arguments.\n"
      "\nReturns:\n"
      "  `tuple[Kernel, QuakeValue, ...]` : "
      "A tuple containing an empty kernel function and a "
      ":class:`QuakeValue` "
      "handle for each argument that was passed into :func:`make_kernel`.\n"
      "\n.. code-block:: python\n\n"
      "  # Example:\n"
      "  # Parameterized kernel that accepts an `int` and `float` as "
      "arguments.\n"
      "  kernel, int_value, float_value = cudaq.make_kernel(int, float)\n");
}

/// Useful macros for defining builder.QIS(...) functions
#define ADD_BUILDER_QIS_METHOD(NAME)                                           \
  .def(                                                                        \
      #NAME,                                                                   \
      [](kernel_builder<> &self, QuakeValue &target) { self.NAME(target); },   \
      py::arg("target"),                                                       \
      "Apply a " #NAME " gate to the given target qubit or qubits.\n"          \
      "\nArgs:\n"                                                              \
      "  target (:class:`QuakeValue`): The qubit or qubits to apply " #NAME    \
      " to.\n"                                                                 \
      "\n.. code-block:: python\n\n"                                           \
      "  # Example:\n"                                                         \
      "  kernel = cudaq.make_kernel()\n"                                       \
      "  # Allocate qubit/s to the `kernel`.\n"                                \
      "  qubits = kernel.qalloc(5)\n"                                          \
      "  # Apply a " #NAME " gate to the qubit/s.\n"                           \
      "  kernel." #NAME "(qubits)\n")                                          \
      .def(                                                                    \
          "c" #NAME,                                                           \
          [](kernel_builder<> &self, QuakeValue &control,                      \
             QuakeValue &target) {                                             \
            std::vector<QuakeValue> controls{control};                         \
            self.NAME(controls, target);                                       \
          },                                                                   \
          py::arg("control"), py::arg("target"),                               \
          "Apply a controlled-" #NAME " operation"                             \
          " to the given target qubit, with the provided control qubit.\n"     \
          "\nArgs:\n"                                                          \
          "  control (:class:`QuakeValue`): The control qubit for the "        \
          "operation. Must be a single qubit, registers are not a valid "      \
          "`control` argument.\n"                                              \
          "  target (:class:`QuakeValue`): The target qubit of the "           \
          "operation.\n"                                                       \
          "\n.. code-block:: python\n\n"                                       \
          "  # Example:\n"                                                     \
          "  kernel = cudaq.make_kernel()\n"                                   \
          "  control = kernel.qalloc()\n"                                      \
          "  target = kernel.qalloc()\n"                                       \
          "  # Apply a controlled-" #NAME " between the two qubits.\n"         \
          "  kernel.c" #NAME "(control=control, target=target)\n")             \
      .def(                                                                    \
          "c" #NAME,                                                           \
          [](kernel_builder<> &self, std::vector<QuakeValue> &controls,        \
             QuakeValue &target) { self.NAME(controls, target); },             \
          py::arg("controls"), py::arg("target"),                              \
          "Apply a controlled-" #NAME " operation"                             \
          " to the given target qubits, with the provided list of control "    \
          "qubits.\n"                                                          \
          "\nArgs:\n"                                                          \
          "  controls (:class:`QuakeValue`): The list of qubits to use as "    \
          "controls for the operation.\n"                                      \
          "  target (:class:`QuakeValue`): The target qubit of the "           \
          "operation.\n"                                                       \
          "\n.. code-block:: python\n\n"                                       \
          "  # Example:\n"                                                     \
          "  kernel = cudaq.make_kernel()\n"                                   \
          "  controls = kernel.qalloc(2)\n"                                    \
          "  target = kernel.qalloc()\n"                                       \
          "  # Apply a controlled-" #NAME                                      \
          " to the target qubit, with the two control qubits.\n"               \
          "  kernel.c" #NAME                                                   \
          "(controls=[controls[0], controls[1]], target=target)\n")

#define ADD_BUILDER_PARAM_QIS_METHOD(NAME)                                     \
  .def(                                                                        \
      #NAME,                                                                   \
      [](kernel_builder<> &self, QuakeValue &parameter, QuakeValue &target) {  \
        self.NAME(parameter, target);                                          \
      },                                                                       \
      py::arg("parameter"), py::arg("target"),                                 \
      "Apply " #NAME                                                           \
      " to the given target qubit, parameterized by the provided "             \
      "kernel argument (`parameter`).\n"                                       \
      "\nArgs:\n"                                                              \
      "  parameter (:class:`QuakeValue`): The kernel argument to "             \
      "parameterize "                                                          \
      "the " #NAME " gate over.\n"                                             \
      "  target (:class:`QuakeValue`): The target qubit of the " #NAME         \
      " gate.\n"                                                               \
      "\n.. code-block:: python\n\n"                                           \
      "  # Example:\n"                                                         \
      "  # Create a kernel that accepts a float, `angle`, as its argument.\n"  \
      "  kernel, angle = cudaq.make_kernel(float)\n"                           \
      "  qubit = kernel.qalloc()\n"                                            \
      "  # Apply an " #NAME " to the kernel at `angle`.\n"                     \
      "  kernel." #NAME "(parameter=angle, target=qubit)\n")                   \
      .def(                                                                    \
          #NAME,                                                               \
          [](kernel_builder<> &self, double parameter, QuakeValue &target) {   \
            self.NAME(parameter, target);                                      \
          },                                                                   \
          py::arg("parameter"), py::arg("target"),                             \
          "Apply " #NAME                                                       \
          " to the given target qubit, parameterized by the provided "         \
          "double value (`parameter`).\n"                                      \
          "\nArgs:\n"                                                          \
          "  parameter (float): The double value to "                          \
          "parameterize "                                                      \
          "the " #NAME " gate over.\n"                                         \
          "  target (:class:`QuakeValue`): The target qubit of the " #NAME     \
          " gate.\n"                                                           \
          "\n.. code-block:: python\n\n"                                       \
          "  # Example:\n"                                                     \
          "  kernel = cudaq.make_kernel() \n"                                  \
          "  # Apply an " #NAME                                                \
          " to the kernel at a concrete parameter value.\n"                    \
          "  kernel." #NAME "(parameter=3.14, target=qubit)\n")

void bindKernel(py::module &mod) {
  py::class_<kernel_builder<>>(
      mod, "Kernel",
      "The :class:`Kernel` provides an API for dynamically constructing "
      "quantum "
      "circuits. The :class:`Kernel` programmatically represents the circuit "
      "as "
      "an MLIR function using the Quake dialect.\n"
      "\nNote:\n"
      "  See :func:`make_kernel` for the :class:`Kernel` constructor.\n"
      "\nAttributes:\n"
      "  name (str): The name of the :class:`Kernel` function. Read-only.\n"
      "  arguments (List[:class:`QuakeValue`]): The arguments accepted by the "
      ":class:`Kernel` function. Read-only.\n"
      "  argument_count (int): The number of arguments accepted by the "
      ":class:`Kernel` function. Read-only.\n")
      .def_property_readonly("name", &cudaq::kernel_builder<>::name)
      .def_property_readonly("arguments",
                             &cudaq::kernel_builder<>::getArguments)
      .def_property_readonly("argument_count",
                             &cudaq::kernel_builder<>::getNumParams)
      /// @brief Bind overloads for `qalloc()`.
      .def(
          "qalloc", [](kernel_builder<> &self) { return self.qalloc(); },
          "Allocate a single qubit and return a handle to it as a "
          ":class:`QuakeValue`.\n"
          "\nReturns:\n"
          "  :class:`QuakeValue` : A handle to the allocated qubit in the "
          "MLIR.\n"
          "\n.. code-block:: python\n\n"
          "  # Example:\n"
          "  kernel = cudaq.make_kernel()\n"
          "  qubit = kernel.qalloc()\n")
      .def(
          "qalloc",
          [](kernel_builder<> &self, std::size_t qubit_count) {
            return self.qalloc(qubit_count);
          },
          py::arg("qubit_count"),
          "Allocate a register of qubits of size `qubit_count` and return a "
          "handle to them as a :class:`QuakeValue`.\n"
          "\nArgs:\n"
          "  qubit_count (`int`): The number of qubits to allocate.\n"
          "\nReturns:\n"
          "  :class:`QuakeValue` : A handle to the allocated qubits in the "
          "MLIR.\n"
          "\n.. code-block:: python\n\n"
          "  # Example:\n"
          "  kernel = cudaq.make_kernel()\n"
          "  qubits = kernel.qalloc(10)\n")
      .def(
          "qalloc",
          [](kernel_builder<> &self, QuakeValue &qubit_count) {
            return self.qalloc(qubit_count);
          },
          py::arg("qubit_count"),
          "Allocate a register of qubits of size `qubit_count` (where "
          "`qubit_count` is an existing :class:`QuakeValue`) and return a "
          "handle to "
          "them as a new :class:`QuakeValue`.\n"
          "\nArgs:\n"
          "  qubit_count (:class:`QuakeValue`): The parameterized number of "
          "qubits to allocate.\n"
          "\nReturns:\n"
          "  :class:`QuakeValue` : A handle to the allocated qubits in the "
          "MLIR.\n"
          "\n.. code-block:: python\n\n"
          "  # Example:\n"
          "  # Create a kernel that takes an int as its argument.\n"
          "  kernel, qubit_count = cudaq.make_kernel(int)\n"
          "  # Allocate the variable number of qubits.\n"
          "  qubits = kernel.qalloc(qubit_count)\n")
      /// @brief Bind the qubit reset method.
      .def(
          "reset",
          [](kernel_builder<> &self, QuakeValue &qubitOrQreg) {
            self.reset(qubitOrQreg);
          },
          "Reset the provided qubit or qubits.")
      /// @brief Allow for JIT compilation of kernel in python via call to
      /// `builder(args)`.
      .def(
          "__call__",
          [&](kernel_builder<> &self, py::args arguments) {
            auto validatedArgs = validateInputArguments(self, arguments);
            OpaqueArguments argData;
            packArgs(argData, validatedArgs);
            self.jitAndInvoke(argData.data());
          },
          "Just-In-Time (JIT) compile `self` (:class:`Kernel`), and call "
          "the kernel function at the provided concrete arguments.\n"
          "\nArgs:\n"
          "  *arguments (Optional[Any]): The concrete values to evaluate the "
          "kernel function at. Leave empty if the `target` kernel doesn't "
          "accept "
          "any arguments.\n"
          "\n.. code-block:: python\n\n"
          "  # Example:\n"
          "  # Create a kernel that accepts an int and float as its "
          "arguments.\n"
          "  kernel, qubit_count, angle = cudaq.make_kernel(int, float)\n"
          "  # Parameterize the number of qubits by `qubit_count`.\n"
          "  qubits = kernel.qalloc(qubit_count)\n"
          "  # Apply an `rx` rotation on the first qubit by `angle`.\n"
          "  kernel.rx(angle, qubits[0])\n"
          "  # Call the `Kernel` on the given number of qubits (5) and at "
          " a concrete angle (pi).\n"
          "  kernel(5, 3.14)\n")
      // clang-format off
      /// @brief Bind single-qubit gates to the kernel builder.
      ADD_BUILDER_QIS_METHOD(h)
      ADD_BUILDER_QIS_METHOD(x)
      ADD_BUILDER_QIS_METHOD(y)
      ADD_BUILDER_QIS_METHOD(z)
      ADD_BUILDER_QIS_METHOD(t)
      ADD_BUILDER_QIS_METHOD(s)
      /// @brief Bind parameterized single-qubit gates.
      ADD_BUILDER_PARAM_QIS_METHOD(rx)
      ADD_BUILDER_PARAM_QIS_METHOD(ry)
      ADD_BUILDER_PARAM_QIS_METHOD(rz)
      ADD_BUILDER_PARAM_QIS_METHOD(r1)
      // clang-format on

      .def(
          "sdg",
          [](kernel_builder<> &self, const QuakeValue &target) {
            return self.s<cudaq::adj>(target);
          },
          "Apply a rotation on the z-axis of -90 degrees to the given target "
          "qubit/s.\n")
      .def(
          "tdg",
          [](kernel_builder<> &self, const QuakeValue &target) {
            return self.t<cudaq::adj>(target);
          },
          "Apply a rotation on the z-axis of -45 degrees to the given target "
          "qubit/s.\n")

      /// @brief Bind the SWAP gate.
      .def(
          "swap",
          [](kernel_builder<> &self, const QuakeValue &first,
             const QuakeValue &second) { return self.swap(first, second); },
          py::arg("first"), py::arg("second"),
          "Swap the states of the provided qubits.\n "
          "\n.. code-block:: python\n\n"
          "  # Example:\n"
          "  kernel = cudaq.make_kernel()\n"
          "  # Allocate qubit/s to the `kernel`.\n"
          "  qubits = kernel.qalloc(2)\n"
          "  # Place the 0th qubit in the 1-state.\n"
          "  kernel.x(qubits[0])\n\n"
          "  # Swap their states.\n"
          "  kernel.swap(qubits[0], qubits[1])\n")
      /// @brief Allow for conditional statements on measurements.
      .def(
          "c_if",
          [&](kernel_builder<> &self, QuakeValue &measurement,
              py::function thenFunction) {
            self.c_if(measurement, [&]() { thenFunction(); });
          },
          py::arg("measurement"), py::arg("function"),
          "Apply the `function` to the :class:`Kernel` if the provided "
          "single-qubit `measurement` returns the 1-state.\n "
          "\nArgs:\n"
          "  measurement (:class:`QuakeValue`): The handle to the "
          "single qubit measurement instruction.\n"
          "  function (Callable): The function to conditionally "
          "apply to the :class:`Kernel`.\n"
          "\nRaises:\n"
          "  RuntimeError: If the provided `measurement` is on more than 1 "
          "qubit.\n"
          "\n.. code-block:: python\n\n"
          "  # Example:\n"
          "  # Create a kernel and allocate a single qubit.\n"
          "  kernel = cudaq.make_kernel()\n"
          "  qubit = kernel.qalloc()\n\n"
          "  # Define a function that performs certain operations on the\n"
          "  # kernel and the qubit.\n"
          "  def then_function():\n"
          "      kernel.x(qubit)\n"
          "  kernel.x(qubit)\n\n"
          "  # Measure the qubit.\n"
          "  measurement = kernel.mz(qubit)\n"
          "  # Apply `then_function` to the `kernel` if the qubit was "
          "measured\n"
          "  # in the 1-state.\n"
          "  kernel.c_if(measurement, then_function)\n")
      /// @brief Bind overloads for measuring qubits and registers.
      .def(
          "mx",
          [](kernel_builder<> &self, QuakeValue &target,
             const std::string &registerName) {
            return self.mx(target, registerName);
          },
          py::arg("target"), py::arg("register_name") = "",
          "Measure the given qubit or qubits in the X-basis. The optional "
          "`register_name` may be used to retrieve results of this measurement"
          " after execution on the QPU. If the measurement call is saved as a "
          "variable,"
          " it will return a :class:`QuakeValue` handle to the measurement "
          "instruction.\n"
          "\nArgs:\n"
          "  target (:class:`QuakeValue`): The qubit or qubits to measure.\n"
          "  register_name (Optional[str]): The optional name to provide the "
          "results of the measurement. Defaults to an empty string.\n "
          "\nReturns:\n"
          "  :class:`QuakeValue` : A handle to this measurement operation in "
          "the MLIR.\n"
          "\nNote:\n"
          "  Measurements may be applied both mid-circuit and at the end of "
          "the circuit. Mid-circuit measurements are currently only supported "
          "through the use of :func:`c_if`.\n"
          "\n.. code-block:: python\n\n"
          "  # Example:\n"
          "  kernel = cudaq.make_kernel()\n"
          "  # Allocate qubit/s to measure.\n"
          "  qubit = kernel.qalloc()\n"
          "  # Measure the qubit/s in the X-basis.\n"
          "  kernel.mx(qubit)\n")
      .def(
          "my",
          [](kernel_builder<> &self, QuakeValue &target,
             const std::string &registerName) {
            return self.my(target, registerName);
          },
          py::arg("target"), py::arg("register_name") = "",
          "Measure the given qubit or qubits in the Y-basis. The optional "
          "`register_name` may be used to retrieve results of this measurement"
          " after execution on the QPU. If the measurement call is saved as a "
          "variable,"
          " it will return a :class:`QuakeValue` handle to the measurement "
          "instruction.\n"
          "\nArgs:\n"
          "  target (:class:`QuakeValue`): The qubit or qubits to measure.\n"
          "  register_name (Optional[str]): The optional name to provide the "
          "results of the measurement. Defaults to an empty string.\n "
          "\nReturns:\n"
          "  :class:`QuakeValue` : A handle to this measurement operation in "
          "the MLIR.\n"
          "\nNote:\n"
          "  Measurements may be applied both mid-circuit and at the end of "
          "the circuit. Mid-circuit measurements are currently only supported "
          "through the use of :func:`c_if`.\n"
          "\n.. code-block:: python\n\n"
          "  # Example:\n"
          "  kernel = cudaq.make_kernel()\n"
          "  # Allocate qubit/s to measure.\n"
          "  qubit = kernel.qalloc()\n"
          "  # Measure the qubit/s in the Y-basis.\n"
          "  kernel.my(qubit)\n")
      .def(
          "mz",
          [](kernel_builder<> &self, QuakeValue &target,
             const std::string &registerName) {
            return self.mz(target, registerName);
          },
          py::arg("target"), py::arg("register_name") = "",
          "Measure the given qubit or qubits in the Z-basis. The optional "
          "`register_name` may be used to retrieve results of this measurement"
          " after execution on the QPU. If the measurement call is saved as a "
          "variable,"
          " it will return a :class:`QuakeValue` handle to the measurement "
          "instruction.\n"
          "\nArgs:\n"
          "  target (:class:`QuakeValue`): The qubit or qubits to measure.\n"
          "  register_name (Optional[str]): The optional name to provide the "
          "results of the measurement. Defaults to an empty string.\n "
          "\nReturns:\n"
          "  :class:`QuakeValue` : A handle to this measurement operation in "
          "the MLIR.\n"
          "\nNote:\n"
          "  Measurements may be applied both mid-circuit and at the end of "
          "the circuit. Mid-circuit measurements are currently only supported "
          "through the use of :func:`c_if`.\n"
          "\n.. code-block:: python\n\n"
          "  # Example:\n"
          "  kernel = cudaq.make_kernel()\n"
          "  # Allocate qubit/s to measure.\n"
          "  qubit = kernel.qalloc()\n"
          "  # Measure the qubit/s in the Z-basis.\n"
          "  kernel.mz(target=qubit)\n")
      .def(
          "adjoint",
          [](kernel_builder<> &self, kernel_builder<> &target,
             py::args target_arguments) {
            std::vector<QuakeValue> values;
            for (auto &value : target_arguments) {
              values.push_back(value.cast<QuakeValue>());
            }
            self.adjoint(target, values);
          },
          py::arg("target"),
          "Apply the adjoint of the `target` kernel in-place to `self`.\n"
          "\nArgs:\n"
          "  target (:class:`Kernel`): The kernel to take the adjoint of.\n"
          "  *target_arguments (Optional[QuakeValue]): The arguments to the "
          "`target` kernel. Leave empty if the `target` kernel doesn't accept "
          "any arguments.\n"
          "\nRaises:\n"
          "  RuntimeError: if the `*target_arguments` passed to the adjoint "
          "call don't match the argument signature of `target`.\n"
          "\n.. code-block:: python\n\n"
          "  # Example:\n"
          "  target_kernel = cudaq.make_kernel()\n"
          "  qubit = target_kernel.qalloc()\n"
          "  target_kernel.x(qubit)\n\n"
          "  # Apply the adjoint of `target_kernel` to `kernel`.\n"
          "  kernel = cudaq.make_kernel()\n"
          "  kernel.adjoint(target_kernel)\n")
      .def(
          "control",
          [](kernel_builder<> &self, kernel_builder<> &target,
             QuakeValue &control, py::args target_arguments) {
            std::vector<QuakeValue> values;
            for (auto &value : target_arguments) {
              values.push_back(value.cast<QuakeValue>());
            }
            self.control(target, control, values);
          },
          py::arg("target"), py::arg("control"),
          "Apply the `target` kernel as a controlled operation in-place to "
          "`self`."
          "Uses the provided `control` as control qubit/s for the operation.\n"
          "\nArgs:\n"
          "  target (:class:`Kernel`): The kernel to apply as a controlled "
          "operation in-place to self.\n"
          "  control (:class:`QuakeValue`): The control qubit or register to "
          "use when applying `target`.\n"
          "  *target_arguments (Optional[QuakeValue]): The arguments to the "
          "`target` kernel. Leave empty if the `target` kernel doesn't accept "
          "any arguments.\n"
          "\nRaises:\n"
          "  RuntimeError: if the `*target_arguments` passed to the control "
          "call don't match the argument signature of `target`.\n"
          "\n.. code-block:: python\n\n"
          "  # Example:\n"
          "  # Create a `Kernel` that accepts a qubit as an argument.\n"
          "  # Apply an X-gate on that qubit.\n"
          "  target_kernel, qubit = cudaq.make_kernel(cudaq.qubit)\n"
          "  target_kernel.x(qubit)\n\n"
          "  # Create another `Kernel` that will apply `target_kernel`\n"
          "  # as a controlled operation.\n"
          "  kernel = cudaq.make_kernel()\n"
          "  control_qubit = kernel.qalloc()\n"
          "  target_qubit = kernel.qalloc()\n"
          "  # In this case, `control` performs the equivalent of a \n"
          "  # controlled-X gate between `control_qubit` and `target_qubit`.\n"
          "  kernel.control(target_kernel, control_qubit, "
          "target_qubit)\n")
      .def(
          "apply_call",
          [](kernel_builder<> &self, kernel_builder<> &target,
             py::args target_arguments) {
            std::vector<QuakeValue> values;
            for (auto &value : target_arguments) {
              values.push_back(value.cast<QuakeValue>());
            }
            self.call(target, values);
          },
          py::arg("target"),
          "Apply a call to the given `target` kernel within the function-body "
          "of "
          "`self` at the provided `target_arguments`.\n"
          "\nArgs:\n"
          "  target (:class:`Kernel`): The kernel to call from within `self`.\n"
          "  *target_arguments (Optional[QuakeValue]): The arguments to the "
          "`target` kernel. Leave empty if the `target` kernel doesn't accept "
          "any arguments.\n"
          "\nRaises:\n"
          "  RuntimeError: if the `*target_arguments` passed to the apply "
          "call don't match the argument signature of `target`.\n"
          "\n.. code-block:: python\n\n"
          "  # Example:\n"
          "  # Build a `Kernel` that's parameterized by a `cudaq.qubit`.\n"
          "  target_kernel, other_qubit = cudaq.make_kernel(cudaq.qubit)\n"
          "  target_kernel.x(other_qubit)\n\n"
          "  # Build a `Kernel` that will call `target_kernel` within its\n"
          "  # own function body.\n"
          "  kernel = cudaq.make_kernel()\n"
          "  qubit = kernel.qalloc()\n"
          "  # Use `qubit` as the argument to `target_kernel`.\n"
          "  kernel.apply_call(target_kernel, qubit)\n"
          "  # The final measurement of `qubit` should return the 1-state.\n"
          "  kernel.mz(qubit)\n")
      .def(
          "for_loop",
          [](kernel_builder<> &self, std::size_t start, std::size_t end,
             py::function body) { self.for_loop(start, end, body); },
          "Add a for loop that starts from the given `start` integer index, "
          "ends at the given `end` integer index (non inclusive), and applies "
          "the given `body` "
          "as a callable function. This callable function must take as input "
          "an index variable that can be used within the body.")
      .def(
          "for_loop",
          [](kernel_builder<> &self, std::size_t start, QuakeValue &end,
             py::function body) { self.for_loop(start, end, body); },
          "Add a for loop that starts from the given `start` integer index, "
          "ends at the given `end` QuakeValue index (non inclusive), and "
          "applies the given "
          "`body` as a callable function. This callable function must take as "
          "input an index variable that can be used within the body.")
      .def(
          "for_loop",
          [](kernel_builder<> &self, QuakeValue &start, std::size_t end,
             py::function body) { self.for_loop(start, end, body); },
          "Add a for loop that starts from the given `start` QuakeValue index, "
          "ends at the given `end` integer index (non inclusive), and applies "
          "the given `body` "
          "as a callable function. This callable function must take as input "
          "an index variable that can be used within the body.")
      .def(
          "for_loop",
          [](kernel_builder<> &self, QuakeValue &start, QuakeValue &end,
             py::function body) { self.for_loop(start, end, body); },
          "Add a for loop that starts from the given `start` QuakeValue index, "
          "ends at the given `end` QuakeValue index (non inclusive), and "
          "applies the given "
          "`body` as a callable function. This callable function must take as "
          "input an index variable that can be used within the body.")
      /// @brief Convert kernel to a Quake string.
      .def("to_quake", &kernel_builder<>::to_quake, "See :func:`__str__`.")
      .def("__str__", &kernel_builder<>::to_quake,
           "Return the :class:`Kernel` as a string in its MLIR representation "
           "using the Quake dialect.\n");
}

void bindBuilder(py::module &mod) {
  bindMakeKernel(mod);
  bindKernel(mod);
}

} // namespace cudaq
