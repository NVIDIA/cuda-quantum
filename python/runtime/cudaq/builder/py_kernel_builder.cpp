/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "py_kernel_builder.h"
#include "utils/OpaqueArguments.h"

#include "cudaq/builder/kernel_builder.h"
#include "cudaq/builder/kernels.h"
#include "cudaq/platform.h"

#include "common/ExecutionContext.h"
#include "common/MeasureCounts.h"

#include <any>

namespace cudaq {
struct PyQubit {};
struct PyQreg {};

void bindMakeKernel(py::module &mod) {

  py::class_<PyQubit>(
      mod, "qubit",
      R"#(The data-type representing a qubit argument to a :class:`Kernel`
function.
                      
.. code-block:: python

  # Example:
  kernel, qubit = cudaq.make_kernel(cudaq.qubit))#");
  py::class_<PyQreg>(mod, "qreg",
                     R"#(The data-type representing a register of qubits as an 
argument to a :class:`Kernel` function.

.. code-block:: python

  # Example:
  kernel, qreg = cudaq.make_kernel(cudaq.qreg)
  
)#");

  mod.def(
      "make_kernel",
      []() {
        std::vector<details::KernelBuilderType> empty;
        return std::make_unique<kernel_builder<>>(empty);
      },
      R"#(Create and return a :class:`Kernel` that accepts no arguments.

Returns:
  :class:`Kernel`: An empty kernel function to be used for quantum program 
  construction. This kernel is non-parameterized and accepts no arguments.

.. code-block:: python

  # Example:
  # Non-parameterized kernel.
  kernel = cudaq.make_kernel()
  
)#");

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
      R"#(Create a :class:`Kernel` that takes the provided types as arguments. 
Returns a tuple containing the kernel and a :class:`QuakeValue` for each 
kernel argument.

Note:
  The following types are supported as kernel arguments: `int`, `float`, 
  `list`/`List`, `cudaq.qubit`, or `cudaq.qreg`.

Args:
  *arguments : A variable amount of types for the kernel function to accept as 
    arguments.

Returns:
  `tuple[Kernel, QuakeValue, ...]`: 
  A tuple containing an empty kernel function and a :class:`QuakeValue` 
  handle for each argument that was passed into :func:`make_kernel`.

.. code-block:: python

  # Example:
  # Parameterized kernel that accepts an `int`
  # and `float` as arguments.
  kernel, int_value, float_value = cudaq.make_kernel(int, float)

)#");

  mod.def(
      "from_state",
      [](kernel_builder<> &kernel, QuakeValue &qubits,
         py::array_t<std::complex<double>> &data) {
        std::vector<std::complex<double>> tmp(data.data(),
                                              data.data() + data.size());
        cudaq::from_state(kernel, qubits, tmp);
      },
      py::arg("kernel"), py::arg("qubits"), py::arg("state"),
      R"#(Decompose the input state vector to a set of controlled operations and 
rotations within the provided `kernel` body.

.. code-block:: python

  # Example:
  import numpy as np
  # Define our kernel.
  kernel = cudaq.make_kernel()
  # Allocate some qubits.
  qubits = kernel.qalloc(3)
  # Define a simple state vector.
  state = np.array([0,1], dtype=np.complex128)
  
  # Now calling `from_state`, we will provide the `kernel` and the 
  # qubit/s that we'd like to evolve to the given `state`.
  cudaq.from_state(kernel, qubits, state)

)#");

  mod.def(
      "from_state",
      [](py::array_t<std::complex<double>> &data) {
        std::vector<std::complex<double>> tmp(data.data(),
                                              data.data() + data.size());
        return cudaq::from_state(tmp);
      },
      py::arg("state"),
      R"#(Decompose the given state vector into a set of controlled operations 
and rotations and return a valid, callable, CUDA Quantum kernel.
      
.. code-block:: python

  # Example:
  import numpy as np
  # Define a simple state vector.
  state = np.array([0,1], dtype=np.complex128)
  # Create and return a kernel that produces the given `state`.
  kernel = cudaq.from_state(state)

)#");
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
      R"#(The :class:`Kernel` provides an API for dynamically constructing quantum 
circuits. The :class:`Kernel` programmatically represents the circuit as an MLIR 
function using the Quake dialect.

Note:
  See :func:`make_kernel` for the :class:`Kernel` constructor.

Attributes:
  name (str): The name of the :class:`Kernel` function. Read-only.
  arguments (List[:class:`QuakeValue`]): The arguments accepted by the 
    :class:`Kernel` function. Read-only.
  argument_count (int): The number of arguments accepted by the 
    :class:`Kernel` function. Read-only.)#")
      .def_property_readonly("name", &cudaq::kernel_builder<>::name)
      .def_property_readonly("arguments",
                             &cudaq::kernel_builder<>::getArguments)
      .def_property_readonly("argument_count",
                             &cudaq::kernel_builder<>::getNumParams)
      /// @brief Bind overloads for `qalloc()`.
      .def(
          "qalloc", [](kernel_builder<> &self) { return self.qalloc(); },
          R"#(Allocate a single qubit and return a handle to it as a 
:class:`QuakeValue`.

Returns:
  :class:`QuakeValue`: A handle to the allocated qubit in the MLIR.

.. code-block:: python

  # Example:
  kernel = cudaq.make_kernel()
  qubit = kernel.qalloc()
)#")
      .def(
          "qalloc",
          [](kernel_builder<> &self, std::size_t qubit_count) {
            return self.qalloc(qubit_count);
          },
          py::arg("qubit_count"),
          R"#(Allocate a register of qubits of size `qubit_count` and return a 
handle to them as a :class:`QuakeValue`.

Args:
  qubit_count (`int`): The number of qubits to allocate.

Returns:
  :class:`QuakeValue`: A handle to the allocated qubits in the MLIR.

.. code-block:: python

  # Example:
  kernel = cudaq.make_kernel()
  qubits = kernel.qalloc(10)

)#")
      .def(
          "qalloc",
          [](kernel_builder<> &self, QuakeValue &qubit_count) {
            return self.qalloc(qubit_count);
          },
          py::arg("qubit_count"),
          R"#(Allocate a register of qubits of size `qubit_count` (where 
`qubit_count` is an existing :class:`QuakeValue`) and return a handle to 
them as a new :class:`QuakeValue`.

Args:
  qubit_count (:class:`QuakeValue`): The parameterized number of 
    qubits to allocate.

Returns:
  :class:`QuakeValue`: A handle to the allocated qubits in the MLIR.

.. code-block:: python
  
  # Example:
  # Create a kernel that takes an int as its argument.
  kernel, qubit_count = cudaq.make_kernel(int)
  # Allocate the variable number of qubits.
  qubits = kernel.qalloc(qubit_count)
  
)#")
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
          R"#(Just-In-Time (JIT) compile `self` (:class:`Kernel`), and call 
the kernel function at the provided concrete arguments.

Args:
  *arguments (Optional[Any]): The concrete values to evaluate the 
    kernel function at. Leave empty if the `target` kernel doesn't 
    accept any arguments.

.. code-block:: python

  # Example:
  # Create a kernel that accepts an int and float as its 
  # arguments.
  kernel, qubit_count, angle = cudaq.make_kernel(int, float)
  # Parameterize the number of qubits by `qubit_count`.
  qubits = kernel.qalloc(qubit_count)
  # Apply an `rx` rotation on the first qubit by `angle`.
  kernel.rx(angle, qubits[0])
  # Call the `Kernel` on the given number of qubits (5) and at 
  a concrete angle (pi).
  kernel(5, 3.14))#")
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
          py::arg("target"),
          R"#(Apply a rotation on the z-axis of negative 90 degrees to the given
target qubit/s.

Args:
  target (:class:`QuakeValue`): The qubit or qubits to apply an sdg gate to.

.. code-block:: python

  # Example:
  kernel = cudaq.make_kernel()
  # Allocate qubit/s to the `kernel`.
  qubits = kernel.qalloc(5)
  # Apply a sdg gate to the qubit/s.
  kernel.sdg(qubit))#")
      .def(
          "tdg",
          [](kernel_builder<> &self, const QuakeValue &target) {
            return self.t<cudaq::adj>(target);
          },
          py::arg("target"),
          R"#(Apply a rotation on the z-axis of negative 45 degrees to the given
target qubit/s.

Args:
  target (:class:`QuakeValue`): The qubit or qubits to apply a tdg gate to.

.. code-block:: python

  # Example:
  kernel = cudaq.make_kernel()
  # Allocate qubit/s to the `kernel`.
  qubits = kernel.qalloc(5)
  # Apply a tdg gate to the qubit/s.
  kernel.tdg(qubit))#")

      /// @brief Bind the SWAP gate.
      .def(
          "swap",
          [](kernel_builder<> &self, const QuakeValue &first,
             const QuakeValue &second) { return self.swap(first, second); },
          py::arg("first"), py::arg("second"),
          R"#(Swap the states of the provided qubits. 

.. code-block:: python

  # Example:
  kernel = cudaq.make_kernel()
  # Allocate qubit/s to the `kernel`.
  qubits = kernel.qalloc(2)
  # Place the 0th qubit in the 1-state.
  kernel.x(qubits[0])
  # Swap their states.
  kernel.swap(qubits[0], qubits[1]))#")
      /// @brief Allow for conditional statements on measurements.
      .def(
          "c_if",
          [&](kernel_builder<> &self, QuakeValue &measurement,
              py::function thenFunction) {
            self.c_if(measurement, [&]() { thenFunction(); });
          },
          py::arg("measurement"), py::arg("function"),
          R"#(Apply the `function` to the :class:`Kernel` if the provided 
single-qubit `measurement` returns the 1-state. 

Args:
  measurement (:class:`QuakeValue`): The handle to the single qubit 
    measurement instruction.
  function (Callable): The function to conditionally apply to the 
    :class:`Kernel`.

Raises:
  RuntimeError: If the provided `measurement` is on more than 1 qubit.

.. code-block:: python

  # Example:
  # Create a kernel and allocate a single qubit.
  kernel = cudaq.make_kernel()
  qubit = kernel.qalloc()
  # Define a function that performs certain operations on the
  # kernel and the qubit.
  def then_function():
      kernel.x(qubit)
  kernel.x(qubit)
  # Measure the qubit.
  measurement = kernel.mz(qubit)
  # Apply `then_function` to the `kernel` if the qubit was measured
  # in the 1-state.
  kernel.c_if(measurement, then_function))#")
      /// @brief Bind overloads for measuring qubits and registers.
      .def(
          "mx",
          [](kernel_builder<> &self, QuakeValue &target,
             const std::string &registerName) {
            return self.mx(target, registerName);
          },
          py::arg("target"), py::arg("register_name") = "",
          R"#(Measure the given qubit or qubits in the X-basis. The optional 
`register_name` may be used to retrieve results of this measurement after 
execution on the QPU. If the measurement call is saved as a variable, it will 
return a :class:`QuakeValue` handle to the measurement instruction.

Args:
  target (:class:`QuakeValue`): The qubit or qubits to measure.
  register_name (Optional[str]): The optional name to provide the 
    results of the measurement. Defaults to an empty string. 

Returns:
  :class:`QuakeValue`: A handle to this measurement operation in the MLIR.

Note:
  Measurements may be applied both mid-circuit and at the end of 
  the circuit. Mid-circuit measurements are currently only supported 
  through the use of :func:`c_if`.

.. code-block:: python

  # Example:
  kernel = cudaq.make_kernel()
  # Allocate qubit/s to measure.
  qubit = kernel.qalloc()
  # Measure the qubit/s in the X-basis.
  kernel.mx(qubit))#")
      .def(
          "my",
          [](kernel_builder<> &self, QuakeValue &target,
             const std::string &registerName) {
            return self.my(target, registerName);
          },
          py::arg("target"), py::arg("register_name") = "",
          R"#(Measure the given qubit or qubits in the Y-basis. The optional 
`register_name` may be used to retrieve results of this measurement after 
execution on the QPU. If the measurement call is saved as a variable, it will
return a :class:`QuakeValue` handle to the measurement instruction.

Args:
  target (:class:`QuakeValue`): The qubit or qubits to measure.
  register_name (Optional[str]): The optional name to provide the 
    results of the measurement. Defaults to an empty string. 

Returns:
  :class:`QuakeValue`: A handle to this measurement operation in the MLIR.

Note:
  Measurements may be applied both mid-circuit and at the end of 
  the circuit. Mid-circuit measurements are currently only supported 
  through the use of :func:`c_if`.

.. code-block:: python

  # Example:
  kernel = cudaq.make_kernel()
  # Allocate qubit/s to measure.
  qubit = kernel.qalloc()
  # Measure the qubit/s in the Y-basis.
  kernel.my(qubit))#")
      .def(
          "mz",
          [](kernel_builder<> &self, QuakeValue &target,
             const std::string &registerName) {
            return self.mz(target, registerName);
          },
          py::arg("target"), py::arg("register_name") = "",
          R"#(Measure the given qubit or qubits in the Z-basis. The optional 
`register_name` may be used to retrieve results of this measurement after 
execution on the QPU. If the measurement call is saved as a variable, it will 
return a :class:`QuakeValue` handle to the measurement instruction.

Args:
  target (:class:`QuakeValue`): The qubit or qubits to measure.
  register_name (Optional[str]): The optional name to provide the 
    results of the measurement. Defaults to an empty string. 

Returns:
  :class:`QuakeValue`: A handle to this measurement operation in the MLIR.

Note:
  Measurements may be applied both mid-circuit and at the end of 
  the circuit. Mid-circuit measurements are currently only supported 
  through the use of :func:`c_if`.

.. code-block:: python

  # Example:
  kernel = cudaq.make_kernel()
  # Allocate qubit/s to measure.
  qubit = kernel.qalloc()
  # Measure the qubit/s in the Z-basis.
  kernel.mz(target=qubit))#")
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
          R"#(Apply the adjoint of the `target` kernel in-place to `self`.

Args:
  target (:class:`Kernel`): The kernel to take the adjoint of.
  *target_arguments (Optional[QuakeValue]): The arguments to the 
    `target` kernel. Leave empty if the `target` kernel doesn't accept 
    any arguments.

Raises:
  RuntimeError: if the `*target_arguments` passed to the adjoint call don't 
    match the argument signature of `target`.

.. code-block:: python

  # Example:
  target_kernel = cudaq.make_kernel()
  qubit = target_kernel.qalloc()
  target_kernel.x(qubit)
  # Apply the adjoint of `target_kernel` to `kernel`.
  kernel = cudaq.make_kernel()
  kernel.adjoint(target_kernel))#")
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
          R"#(Apply the `target` kernel as a controlled operation in-place to 
`self`.Uses the provided `control` as control qubit/s for the operation.

Args:
  target (:class:`Kernel`): The kernel to apply as a controlled 
    operation in-place to self.
  control (:class:`QuakeValue`): The control qubit or register to 
    use when applying `target`.
  *target_arguments (Optional[QuakeValue]): The arguments to the 
    `target` kernel. Leave empty if the `target` kernel doesn't accept 
    any arguments.

Raises:
  RuntimeError: if the `*target_arguments` passed to the control 
    call don't match the argument signature of `target`.

.. code-block:: python

  # Example:
  # Create a `Kernel` that accepts a qubit as an argument.
  # Apply an X-gate on that qubit.
  target_kernel, qubit = cudaq.make_kernel(cudaq.qubit)
  target_kernel.x(qubit)
  # Create another `Kernel` that will apply `target_kernel`
  # as a controlled operation.
  kernel = cudaq.make_kernel()
  control_qubit = kernel.qalloc()
  target_qubit = kernel.qalloc()
  # In this case, `control` performs the equivalent of a 
  # controlled-X gate between `control_qubit` and `target_qubit`.
  kernel.control(target_kernel, control_qubit, target_qubit))#")
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
          R"#(Apply a call to the given `target` kernel within the function-body 
of `self` at the provided `target_arguments`.

Args:
  target (:class:`Kernel`): The kernel to call from within `self`.
  *target_arguments (Optional[QuakeValue]): The arguments to the `target` kernel. 
    Leave empty if the `target` kernel doesn't accept any arguments.

Raises:
  RuntimeError: if the `*args` passed to the apply 
    call don't match the argument signature of `target`.

.. code-block:: python

  # Example:
  # Build a `Kernel` that's parameterized by a `cudaq.qubit`.
  target_kernel, other_qubit = cudaq.make_kernel(cudaq.qubit)
  target_kernel.x(other_qubit)
  # Build a `Kernel` that will call `target_kernel` within its
  # own function body.
  kernel = cudaq.make_kernel()
  qubit = kernel.qalloc()
  # Use `qubit` as the argument to `target_kernel`.
  kernel.apply_call(target_kernel, qubit)
  # The final measurement of `qubit` should return the 1-state.
  kernel.mz(qubit))#")
      .def(
          "for_loop",
          [](kernel_builder<> &self, std::size_t start, std::size_t stop,
             py::function function) { self.for_loop(start, stop, function); },
          py::arg("start"), py::arg("stop"), py::arg("function"),
          R"#(Add a for loop that starts from the given `start` integer index, 
ends at the given `stop` integer index (non inclusive), applying the provided 
`function` within `self` at each iteration.

Args:
  start (int): The beginning iterator value for the for loop.
  stop (int): The final iterator value (non-inclusive) for the for loop.
  function (Callable): The callable function to apply within the `kernel` at
    each iteration.

Note:
  This callable function must take as input an index variable that can
  be used within the body.
  
.. code-block:: python

  # Example:
  # Create a kernel and allocate (5) qubits to it.
  kernel = cudaq.make_kernel()
  qubits = kernel.qalloc(5)
  kernel.h(qubits[0])

  def foo(index: int):
      """A function that will be applied to `kernel` in a for loop."""
      kernel.cx(qubits[index], qubits[index+1])

  # Create a for loop in `kernel`, providing a concrete number
  # of iterations to run (4).
  kernel.for_loop(start=0, stop=4, function=foo)

  # Execute the kernel.
  result = cudaq.sample(kernel)
  print(result)

)#")
      .def(
          "for_loop",
          [](kernel_builder<> &self, std::size_t start, QuakeValue &end,
             py::function body) { self.for_loop(start, end, body); },
          py::arg("start"), py::arg("stop"), py::arg("function"),
          R"#(Add a for loop that starts from the given `start` integer index, 
ends at the given `stop` :class:`QuakeValue` index (non inclusive), applying the 
provided `function` within `self` at each iteration.

Args:
  start (int): The beginning iterator value for the for loop.
  stop (:class:`QuakeValue`): The final iterator value (non-inclusive) for the for loop.
  function (Callable): The callable function to apply within the `kernel` at
    each iteration.

.. code-block:: python

  # Example:
  # Create a kernel function that takes an `int` argument.
  kernel, size = cudaq.make_kernel(int)
  # Parameterize the allocated number of qubits by the int.
  qubits = kernel.qalloc(size)
  kernel.h(qubits[0])

  def foo(index: int):
      """A function that will be applied to `kernel` in a for loop."""
      kernel.cx(qubits[index], qubits[index+1])

  # Create a for loop in `kernel`, parameterized by the `size`
  # argument for its `stop` iterator.
  kernel.for_loop(start=0, stop=size-1, function=foo)

  # Execute the kernel, passing along a concrete value (5) for 
  # the `size` argument.
  counts = cudaq.sample(kernel, 5)
  print(counts)
    
)#")
      .def(
          "for_loop",
          [](kernel_builder<> &self, QuakeValue &start, std::size_t end,
             py::function body) { self.for_loop(start, end, body); },
          py::arg("start"), py::arg("stop"), py::arg("function"),
          R"#(Add a for loop that starts from the given `start` :class:`QuakeValue`
index, ends at the given `stop` integer index (non inclusive), applying the provided 
`function` within `self` at each iteration.

Args:
  start (:class:`QuakeValue`): The beginning iterator value for the for loop.
  stop (int): The final iterator value (non-inclusive) for the for loop.
  function (Callable): The callable function to apply within the `kernel` at
    each iteration.

.. code-block:: python

  # Example:
  # Create a kernel function that takes an `int` argument.
  kernel, start = cudaq.make_kernel(int)
  # Allocate 5 qubits.
  qubits = kernel.qalloc(5)
  kernel.h(qubits[0])

  def foo(index: int):
      """A function that will be applied to `kernel` in a for loop."""
      kernel.cx(qubits[index], qubits[index+1])

  # Create a for loop in `kernel`, with its start index being
  # parameterized by the kernel's `start` argument.
  kernel.for_loop(start=start, stop=4, function=foo)

  # Execute the kernel, passing along a concrete value (0) for 
  # the `start` argument.
  counts = cudaq.sample(kernel, 0)
  print(counts)
    
)#")
      .def(
          "for_loop",
          [](kernel_builder<> &self, QuakeValue &start, QuakeValue &end,
             py::function body) { self.for_loop(start, end, body); },
          py::arg("start"), py::arg("stop"), py::arg("function"),
          R"#(Add a for loop that starts from the given `start` :class:`QuakeValue`
index, and ends at the given `stop` :class:`QuakeValue` index (non inclusive). The
provided `function` will be applied within `self` at each iteration. 

Args:
  start (:class:`QuakeValue`): The beginning iterator value for the for loop.
  stop (:class:`QuakeValue`): The final iterator value (non-inclusive) for the for loop.
  function (Callable): The callable function to apply within the `kernel` at
    each iteration.

.. code-block:: python

    # Example:
    # Create a kernel function that takes two `int`'s as arguments.
    kernel, start, stop = cudaq.make_kernel(int, int)
    # Parameterize the allocated number of qubits by the int.
    qubits = kernel.qalloc(stop)
    kernel.h(qubits[0])

    def foo(index: int):
        """A function that will be applied to `kernel` in a for loop."""
        kernel.x(qubits[index])

    # Create a for loop in `kernel`, parameterized by the `start` 
    # and `stop` arguments.
    kernel.for_loop(start=start, stop=stop, function=foo)

    # Execute the kernel, passing along concrete values for the 
    # `start` and `stop` arguments.
    counts = cudaq.sample(kernel, 3, 8)
    print(counts))#")
      /// @brief Convert kernel to a Quake string.
      .def("to_quake", &kernel_builder<>::to_quake, "See :func:`__str__`.")
      .def("__str__", &kernel_builder<>::to_quake,
           "Return the :class:`Kernel` as a string in its MLIR representation "
           "using the Quake dialect.\n")
      .def(
          "exp_pauli",
          [](kernel_builder<> &self, py::object theta, const QuakeValue &qubits,
             const std::string &pauliWord) {
            if (py::isinstance<py::float_>(theta))
              self.exp_pauli(theta.cast<double>(), qubits, pauliWord);
            else if (py::isinstance<QuakeValue>(theta))
              self.exp_pauli(theta.cast<QuakeValue &>(), qubits, pauliWord);
            else
              throw std::runtime_error(
                  "Invalid `theta` argument type. Must be a "
                  "`float` or a `QuakeValue`.");
          },
          "Apply a general Pauli tensor product rotation, `exp(i theta P)`, on "
          "the specified qubit register. The Pauli tensor product is provided "
          "as a string, e.g. `XXYX` for a 4-qubit term. The angle parameter "
          "can be provided as a concrete float or a `QuakeValue`.");
}

void bindBuilder(py::module &mod) {
  bindMakeKernel(mod);
  bindKernel(mod);
}

} // namespace cudaq
