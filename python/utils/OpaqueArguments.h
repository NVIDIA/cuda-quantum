/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/FmtCore.h"
#include "cudaq/builder/kernel_builder.h"
#include <chrono>
#include <functional>
#include <future>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;
using namespace std::chrono_literals;

namespace cudaq {

/// @brief The OpaqueArguments type wraps a vector
/// of function arguments represented as opaque
/// pointers. For each element in the vector of opaque
/// pointers, we also track the arguments corresponding
/// deletion function - a function invoked upon destruction
/// of this OpaqueArguments to clean up the memory.
class OpaqueArguments {
public:
  using OpaqueArgDeleter = std::function<void(void *)>;

private:
  /// @brief The opaque argument pointers
  std::vector<void *> args;

  /// @brief Deletion functions for the arguments.
  std::vector<OpaqueArgDeleter> deleters;

public:
  /// @brief Add an opaque argument and its deleter to this OpaqueArguments
  template <typename ArgPointer, typename Deleter>
  void emplace_back(ArgPointer &&pointer, Deleter &&deleter) {
    args.emplace_back(pointer);
    deleters.emplace_back(deleter);
  }

  /// @brief Return the args as a pointer to void*.
  void **data() { return args.data(); }

  /// @brief Return the number of arguments
  std::size_t size() { return args.size(); }

  /// Destructor, clean up the memory
  ~OpaqueArguments() {
    for (std::size_t counter = 0; auto &ptr : args)
      deleters[counter++](ptr);

    args.clear();
    deleters.clear();
  }
};

/// @brief Validate that the number of arguments provided is
/// correct for the given kernel_builder. Throw an exception if not.
inline py::args validateInputArguments(kernel_builder<> &kernel,
                                       py::args &args) {
  // Ensure the user input is correct.
  auto nInputArgs = args.size();
  auto nRequiredParams = kernel.getNumParams();
  if (nRequiredParams != nInputArgs)
    throw std::runtime_error(
        fmt::format("Kernel requires {} input parameter{} but {} provided.",
                    nRequiredParams, nRequiredParams > 1 ? "s" : "",
                    nInputArgs == 0 ? "none" : std::to_string(nInputArgs)));

  // Look at the input arguments, validate they are ok
  // Specifically here we'll check if we've been given
  // other list-like types as input for a stdvec (like numpy array)
  py::args processed = py::tuple(args.size());
  for (std::size_t i = 0; i < args.size(); ++i) {
    auto arg = args[i];
    if (kernel.isArgStdVec(i)) {

      auto nRequiredElements = kernel.getArguments()[i].getRequiredElements();

      // Check if it has tolist, so it might be a 1d buffer (array / numpy
      // ndarray)
      if (py::hasattr(args[i], "tolist")) {
        // This is a valid ndarray if it has tolist and shape
        if (!py::hasattr(args[i], "shape"))
          throw std::runtime_error(
              "Invalid input argument type, could not get shape of array.");

        // This is an ndarray with tolist() and shape attributes
        // get the shape and check its size
        auto shape = args[i].attr("shape").cast<py::tuple>();
        if (shape.size() != 1)
          throw std::runtime_error("Cannot pass ndarray with shape != (N,).");

        arg = args[i].attr("tolist")();
      }

      // has to be a list if its not a nd array
      if (!py::isinstance<py::list>(arg))
        throw std::runtime_error(
            "Invalid list-like argument to Kernel.__call__()");

      auto nElements = arg.cast<py::list>().size();
      if (nRequiredElements != nElements)
        throw std::runtime_error("Kernel list argument requires " +
                                 std::to_string(nRequiredElements) + " but " +
                                 std::to_string(nElements) + " were provided.");
    }

    processed[i] = arg;
  }

  // TODO: Handle more type checking

  return processed;
}

/// @brief For general function broadcasting over many argument
/// sets, this function will create those argument sets from
/// the input args.
inline std::vector<py::args> createArgumentSet(py::args &args) {
  // we accept float, int, list so we will check here for
  // list[float], list[int], list[list], or ndarray for any,
  // where the ndarray could be 2D (list[list])

  // First we need to get the size of the arg set.
  std::size_t nArgSets = py::len(args[0]);

  // Now I want to build up vector<tuple (arg0, arg1, ...)>
  std::vector<py::args> argSet;
  for (std::size_t j = 0; j < nArgSets; j++) {
    py::args currentArgs = py::tuple(args.size());

    for (std::size_t i = 0; i < args.size(); ++i) {
      auto arg = args[i];

      if (py::isinstance<py::list>(arg)) {
        auto list = arg.cast<py::list>();
        if (list.size() != nArgSets)
          throw std::runtime_error(
              "Invalid argument to sample/observe broadcast, must be a list of "
              "argument instances.");
        currentArgs[i] = list[j];
      }

      // This is not a list, but see if we can get it as one
      if (py::hasattr(args[i], "tolist")) {
        // This is a valid ndarray if it has tolist and shape
        if (!py::hasattr(args[i], "shape"))
          throw std::runtime_error(
              "Invalid input argument type, could not get shape of array.");

        // This is an ndarray with tolist() and shape attributes
        // get the shape and check its size
        auto shape = args[i].attr("shape").cast<py::tuple>();
        if (shape.size() > 2)
          throw std::runtime_error(
              "Invalid kernel arg for sample_n / observe_n, shape.size() > 2");

        // Can handle 1d array and 2d matrix of data
        if (shape.size() == 2) {
          auto list =
              arg.attr("__getitem__")(j).attr("tolist")().cast<py::list>();
          currentArgs[i] = list;
        } else {
          auto list = arg.attr("tolist")().cast<py::list>();
          currentArgs[i] = list[j];
        }
      }
    }

    argSet.push_back(currentArgs);
  }

  return argSet;
}

/// @brief Convert py::args to an OpaqueArguments instance
inline void packArgs(OpaqueArguments &argData, py::args args) {
  for (auto &arg : args) {
    if (py::isinstance<py::float_>(arg)) {
      double *ourAllocatedArg = new double();
      *ourAllocatedArg = PyFloat_AsDouble(arg.ptr());
      argData.emplace_back(ourAllocatedArg, [](void *ptr) {
        delete static_cast<double *>(ptr);
      });
    }
    if (py::isinstance<py::int_>(arg)) {
      int *ourAllocatedArg = new int();
      *ourAllocatedArg = PyLong_AsLong(arg.ptr());
      argData.emplace_back(ourAllocatedArg,
                           [](void *ptr) { delete static_cast<int *>(ptr); });
    }
    if (py::isinstance<py::list>(arg)) {
      auto casted = py::cast<py::list>(arg);
      std::vector<double> *ourAllocatedArg =
          new std::vector<double>(casted.size());
      for (std::size_t counter = 0; auto el : casted) {
        (*ourAllocatedArg)[counter++] = PyFloat_AsDouble(el.ptr());
      }
      argData.emplace_back(ourAllocatedArg, [](void *ptr) {
        delete static_cast<std::vector<double> *>(ptr);
      });
    }
  }
}

/// @brief Return true if the given py::args represents a
/// request for broadcasting sample or observe over all argument sets.
/// Kernel arg types can be int, float, list, so
/// we should check if args[i] is a list or ndarray.
inline bool isBroadcastRequest(kernel_builder<> &builder, py::args &args) {
  if (args.empty())
    return false;

  auto arg = args[0];
  // Just need to check the leading argument
  if (py::isinstance<py::list>(arg) && !builder.isArgStdVec(0))
    return true;

  if (py::hasattr(arg, "tolist")) {
    if (!py::hasattr(arg, "shape"))
      return false;

    auto shape = arg.attr("shape").cast<py::tuple>();
    if (shape.size() == 1 && !builder.isArgStdVec(0))
      return true;

    // // If shape is 2, then we know its a list of list
    if (shape.size() == 2)
      return true;
  }

  return false;
}

} // namespace cudaq
