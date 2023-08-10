/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "LinkedLibraryHolder.h"
#include "cudaq/domains/chemistry/MoleculePackageDriver.h"
#include <pybind11/embed.h>

namespace py = pybind11;
using namespace cudaq;

namespace {

/// @brief Reference to the pybind11 scoped interpreter
thread_local static std::unique_ptr<py::scoped_interpreter> interp;

/// @brief Map an OpenFermion QubitOperator represented as a py::object
/// to a CUDA Quantum spin_op
spin_op fromOpenFermionQubitOperator(const py::object &op) {
  if (!py::hasattr(op, "terms"))
    throw std::runtime_error(
        "This is not an openfermion operator, must have 'terms' attribute.");
  std::map<std::string, std::function<spin_op(std::size_t)>> creatorMap{
      {"X", [](std::size_t i) { return spin::x(i); }},
      {"Y", [](std::size_t i) { return spin::y(i); }},
      {"Z", [](std::size_t i) { return spin::z(i); }}};
  auto terms = op.attr("terms");
  spin_op H;
  for (auto term : terms) {
    auto termTuple = term.cast<py::tuple>();
    spin_op localTerm;
    for (auto &element : termTuple) {
      auto casted = element.cast<std::pair<std::size_t, std::string>>();
      localTerm *= creatorMap[casted.second](casted.first);
    }
    H += terms[term].cast<double>() * localTerm;
  }
  H -= spin::i(H.num_qubits() - 1);
  return H;
}

/// @brief Implement the CUDA Quantum MoleculePackageDriver interface
/// with support for generating molecular hamiltonians via PySCF. We
/// acheive this via Pybind11's embedded interpreter capabilities.
class PySCFPackageDriver : public MoleculePackageDriver {
protected:
  /// @brief The name of the chemistry python module.
  constexpr static const char ChemistryModuleName[] = "cudaq.domains.chemistry";

  /// @brief The name of the function we'll use to drive PySCF.
  constexpr static const char CreatorFunctionName[] =
      "__internal_cpp_create_molecular_hamiltonian";

public:
  /// @brief Create the molecular hamiltonian
  molecular_hamiltonian createMolecule(
      const molecular_geometry &geometry, const std::string &basis,
      int multiplicity, int charge,
      std::optional<std::size_t> nActiveElectrons = std::nullopt,
      std::optional<std::size_t> nActiveOrbitals = std::nullopt) override {
    if (!interp)
      interp = std::make_unique<py::scoped_interpreter>();

    // Convert the molecular_geometry to a list[tuple(str,tuple)]
    py::list pyGeometry(geometry.size());
    for (std::size_t counter = 0; auto &atom : geometry) {
      py::tuple coordinate(3);
      for (int i = 0; i < 3; i++)
        coordinate[i] = atom.coordinates[i];

      pyGeometry[counter++] = py::make_tuple(atom.name, coordinate);
    }

    // We don't want to modify the platform, indicate so
    cudaq::LinkedLibraryHolder::disallowTargetModification = true;

    // Import the cudaq python chemistry module
    auto cudaqModule = py::module_::import(ChemistryModuleName);

    // Reset it
    cudaq::LinkedLibraryHolder::disallowTargetModification = false;

    // Setup the active space if requested.
    py::object nElectrons = py::none();
    py::object nActive = py::none();
    if (nActiveElectrons.has_value())
      nElectrons = py::int_(nActiveElectrons.value());
    if (nActiveOrbitals.has_value())
      nActive = py::int_(nActiveOrbitals.value());

    // Run the openfermion-pyscf wrapper to create the hamiltonian + metadata
    auto hamiltonianGen = cudaqModule.attr(CreatorFunctionName);
    auto resultTuple = hamiltonianGen(pyGeometry, basis, multiplicity, charge,
                                      nElectrons, nActive)
                           .cast<py::tuple>();

    // Get the spin_op representation
    auto spinOp = fromOpenFermionQubitOperator(resultTuple[0]);

    // Get the OpenFermion molecule representation
    auto openFermionMolecule = resultTuple[1];

    // Extract the one-body integrals
    auto pyOneBody = openFermionMolecule.attr("one_body_integrals");
    auto shape = pyOneBody.attr("shape").cast<py::tuple>();
    one_body_integrals oneBody(
        {shape[0].cast<std::size_t>(), shape[1].cast<std::size_t>()});
    for (std::size_t i = 0; i < oneBody.shape[0]; i++)
      for (std::size_t j = 0; j < oneBody.shape[1]; j++)
        oneBody(i, j) =
            pyOneBody.attr("__getitem__")(py::make_tuple(i, j)).cast<double>();

    // Extract the two-body integrals
    auto pyTwoBody = openFermionMolecule.attr("two_body_integrals");
    shape = pyTwoBody.attr("shape").cast<py::tuple>();
    two_body_integals twoBody(
        {shape[0].cast<std::size_t>(), shape[1].cast<std::size_t>(),
         shape[2].cast<std::size_t>(), shape[3].cast<std::size_t>()});
    for (std::size_t i = 0; i < twoBody.shape[0]; i++)
      for (std::size_t j = 0; j < twoBody.shape[1]; j++)
        for (std::size_t k = 0; k < twoBody.shape[2]; k++)
          for (std::size_t l = 0; l < twoBody.shape[3]; l++)
            twoBody(i, j, k, l) =
                pyTwoBody.attr("__getitem__")(py::make_tuple(i, j, k, l))
                    .cast<double>();

    // return a new molecular_hamiltonian
    return molecular_hamiltonian{
        spinOp,
        std::move(oneBody),
        std::move(twoBody),
        openFermionMolecule.attr("n_electrons").cast<std::size_t>(),
        openFermionMolecule.attr("n_orbitals").cast<std::size_t>(),
        openFermionMolecule.attr("nuclear_repulsion").cast<double>(),
        openFermionMolecule.attr("hf_energy").cast<double>(),
        openFermionMolecule.attr("fci_energy").cast<double>()};
  }
};

} // namespace
CUDAQ_REGISTER_TYPE(MoleculePackageDriver, PySCFPackageDriver, pyscf)
