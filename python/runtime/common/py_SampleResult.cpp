/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <pybind11/stl.h>

#include "py_SampleResult.h"

#include "common/MeasureCounts.h"

#include <sstream>

namespace cudaq {

void bindMeasureCounts(py::module &mod) {
  using namespace cudaq;

  // TODO Bind the variants of this functions that take the register name
  // as input.
  py::class_<sample_result>(
      mod, "SampleResult",
      R"#(A data-type containing the results of a call to :func:`sample`. 
This includes all measurement counts data from both mid-circuit and 
terminal measurements.

Note:
	At this time, mid-circuit measurements are not directly supported. 
	Mid-circuit measurements may only be used if they are passed through 
	to `c_if`.

Attributes:
	register_names (List[str]): A list of the names of each measurement 
		register that are stored in `self`.)#")
      .def_property_readonly("register_names", &sample_result::register_names)
      .def(
          "dump", [](sample_result &self) { self.dump(); },
          "Print a string of the raw measurement counts data to the "
          "terminal.\n")
      .def(
          "__str__",
          [](sample_result &self) {
            std::stringstream ss;
            self.dump(ss);
            return ss.str();
          },
          "Return a string of the raw measurement counts that are stored in "
          "`self`.\n")
      .def(
          "__getitem__",
          [](sample_result &self, const std::string &bitstring) {
            auto map = self.to_map();
            auto iter = map.find(bitstring);
            if (iter == map.end())
              throw py::key_error("bitstring '" + bitstring +
                                  "' does not exist");

            return iter->second;
          },
          py::arg("bitstring"),
          R"#(Return the measurement counts for the given `bitstring`.

Args:
	bitstring (str): The binary string to return the measurement data of.

Returns:
	float: The number of times the given `bitstring` was measured 
	during the `shots_count` number of executions on the QPU.)#")
      .def(
          "__len__", [](sample_result &self) { return self.to_map().size(); },
          "Return the number of elements in `self`. Equivalent to "
          "the number of uniquely measured bitstrings.\n")
      .def(
          "__iter__",
          [](sample_result &self) {
            return py::make_key_iterator(self.begin(), self.end());
          },
          py::keep_alive<0, 1>(),
          "Iterate through the :class:`SampleResult` dictionary.\n")
      .def("expectation_z", &sample_result::exp_val_z,
           py::arg("register_name") = GlobalRegisterName,
           "Return the expectation value in the Z-basis of the :class:`Kernel` "
           "that was sampled.\n")
      .def("probability", &sample_result::probability,
           "Return the probability of observing the given bit string.\n",
           py::arg("bitstring"), py::arg("register_name") = GlobalRegisterName,
           R"#(Return the probability of measuring the given `bitstring`.

Args:
  bitstring (str): The binary string to return the measurement 
		probability of.
  register_name (Optional[str]): The optional measurement register 
		name to extract the probability from. Defaults to the '__global__' 
		register.

Returns:
  float: 
	The probability of measuring the given `bitstring`. Equivalent 
	to the proportion of the total times the bitstring was measured 
	vs. the number of experiments (`shots_count`).)#")
      .def("most_probable", &sample_result::most_probable,
           py::arg("register_name") = GlobalRegisterName,
           R"#(Return the bitstring that was measured most frequently in the 
experiment.

Args:
  register_name (Optional[str]): The optional measurement register 
		name to extract the most probable bitstring from. Defaults to the 
		'__global__' register.

Returns:
  str: The most frequently measured binary string during the experiment.)#")
      .def("count", &sample_result::count, py::arg("bitstring"),
           py::arg("register_name") = GlobalRegisterName,
           R"#(Return the number of times the given bitstring was observed.

Args:
  bitstring (str): The binary string to return the measurement counts for.
  register_name (Optional[str]): The optional measurement register name to 
		extract the probability from. Defaults to the '__global__' register.

Returns:
  int : The number of times the given bitstring was measured during the experiment.)#")
      .def("get_marginal_counts",
           static_cast<sample_result (sample_result::*)(
               const std::vector<std::size_t> &, const std::string_view)>(
               &sample_result::get_marginal),
           py::arg("marginal_indices"), py::kw_only(),
           py::arg("register_name") = GlobalRegisterName,
           R"#(Extract the measurement counts data for the provided subset of 
qubits (`marginal_indices`).

Args:
  marginal_indices (list[int]): A list of the qubit indices to extract the 
		measurement data from.
  register_name (Optional[str]): The optional measurement register name to extract 
		the counts data from. Defaults to the '__global__' register.
Returns:
  :class:`SampleResult`: 
	A new `SampleResult` dictionary containing the extracted measurement data.)#")
      .def("get_sequential_data", &sample_result::sequential_data,
           py::arg("register_name") = GlobalRegisterName,
           "Return the data from the given register (`register_name`) as it "
           "was collected sequentially. A list of measurement results, not "
           "collated into a map.\n")
      .def(
          "get_register_counts",
          [&](sample_result &self, const std::string &registerName) {
            auto cd = self.to_map(registerName);
            ExecutionResult res(cd);
            return sample_result(res);
          },
          py::arg("register_name"),
          "Extract the provided sub-register (`register_name`) as a new "
          ":class:`SampleResult`.\n")
      .def(
          "items",
          [](sample_result &self) {
            return py::make_iterator(self.begin(), self.end());
          },
          py::keep_alive<0, 1>(),
          "Return the key/value pairs in this :class:`SampleResult` "
          "dictionary.\n")
      .def(
          "values",
          [](sample_result &self) {
            return py::make_value_iterator(self.begin(), self.end());
          },
          py::keep_alive<0, 1>(),
          "Return all values (the counts) in this :class:`SampleResult` "
          "dictionary.\n")
      .def("clear", &sample_result::clear,
           "Clear out all metadata from `self`.\n");
}

} // namespace cudaq
