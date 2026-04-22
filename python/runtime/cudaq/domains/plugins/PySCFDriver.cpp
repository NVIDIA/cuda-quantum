/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/domains/chemistry/MoleculePackageDriver.h"
#include "cudaq/target_control.h"
#include <map>
#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>

using namespace cudaq;

namespace {

/// @brief Map an OpenFermion QubitOperator represented as a nanobind::object
/// to a CUDA-Q spin_op
spin_op fromOpenFermionQubitOperator(const nanobind::object &op) {
  if (!nanobind::hasattr(op, "terms"))
    throw std::runtime_error(
        "This is not an openfermion operator, must have 'terms' attribute.");
  std::map<std::string, std::function<spin_op_term(std::size_t)>> creatorMap{
      {"X", [](std::size_t i) { return spin_op::x(i); }},
      {"Y", [](std::size_t i) { return spin_op::y(i); }},
      {"Z", [](std::size_t i) { return spin_op::z(i); }}};
  auto terms = op.attr("terms");
  auto H = spin_op::empty();
  for (auto term : terms) {
    auto termTuple = nanobind::cast<nanobind::tuple>(term);
    auto localTerm = spin_op::identity();
    for (auto element : termTuple) {
      auto casted =
          nanobind::cast<std::pair<std::size_t, std::string>>(element);
      localTerm *= creatorMap[casted.second](casted.first);
    }
    H += nanobind::cast<double>(terms[term]) * localTerm;
  }
  return H;
}

/// @brief Implement the CUDA-Q MoleculePackageDriver interface
/// with support for generating molecular Hamiltonians via PySCF. We
/// achieve this via nanobind's Python API wrappers.
class PySCFPackageDriver : public MoleculePackageDriver {
protected:
  /// @brief The name of the chemistry python module.
  static constexpr const char ChemistryModuleName[] = "cudaq.domains.chemistry";

  /// @brief The name of the function we'll use to drive PySCF.
  static constexpr const char CreatorFunctionName[] =
      "__internal_cpp_create_molecular_hamiltonian";

public:
  /// @brief Create the molecular hamiltonian
  molecular_hamiltonian createMolecule(
      const molecular_geometry &geometry, const std::string &basis,
      int multiplicity, int charge,
      std::optional<std::size_t> nActiveElectrons = std::nullopt,
      std::optional<std::size_t> nActiveOrbitals = std::nullopt) override {
    if (!Py_IsInitialized())
      Py_Initialize();

    // Convert the molecular_geometry to a list[tuple(str,tuple)]
    nanobind::list pyGeometry;
    for (auto &atom : geometry) {
      nanobind::object coordinate = nanobind::steal(PyTuple_New(3));
      for (int i = 0; i < 3; i++)
        PyTuple_SET_ITEM(coordinate.ptr(), i,
                         nanobind::cast(atom.coordinates[i]).release().ptr());

      pyGeometry.append(nanobind::make_tuple(atom.name, coordinate));
    }

    // We don't want to modify the platform, indicate so
    cudaq::__internal__::disableTargetModification();

    // Import the cudaq python chemistry module
    auto cudaqModule = nanobind::module_::import_(ChemistryModuleName);

    // Reset it
    cudaq::__internal__::enableTargetModification();

    // Setup the active space if requested.
    nanobind::object nElectrons = nanobind::none();
    nanobind::object nActive = nanobind::none();
    if (nActiveElectrons.has_value())
      nElectrons = nanobind::int_(nActiveElectrons.value());
    if (nActiveOrbitals.has_value())
      nActive = nanobind::int_(nActiveOrbitals.value());

    // Run the openfermion-pyscf wrapper to create the hamiltonian + metadata
    auto hamiltonianGen = cudaqModule.attr(CreatorFunctionName);
    auto resultTuple = nanobind::cast<nanobind::tuple>(hamiltonianGen(
        pyGeometry, basis, multiplicity, charge, nElectrons, nActive));

    // Get the spin_op representation
    auto spinOp =
        fromOpenFermionQubitOperator(nanobind::borrow(resultTuple[0]));

    // Get the OpenFermion molecule representation
    auto openFermionMolecule = nanobind::borrow(resultTuple[1]);

    // Extract the one-body integrals
    auto pyOneBody = openFermionMolecule.attr("one_body_integrals");
    auto shape = nanobind::cast<nanobind::tuple>(pyOneBody.attr("shape"));
    one_body_integrals oneBody({nanobind::cast<std::size_t>(shape[0]),
                                nanobind::cast<std::size_t>(shape[1])});
    for (std::size_t i = 0; i < oneBody.shape[0]; i++)
      for (std::size_t j = 0; j < oneBody.shape[1]; j++)
        oneBody(i, j) = nanobind::cast<double>(
            pyOneBody.attr("__getitem__")(nanobind::make_tuple(i, j)));

    // Extract the two-body integrals
    auto pyTwoBody = openFermionMolecule.attr("two_body_integrals");
    shape = nanobind::cast<nanobind::tuple>(pyTwoBody.attr("shape"));
    two_body_integals twoBody({nanobind::cast<std::size_t>(shape[0]),
                               nanobind::cast<std::size_t>(shape[1]),
                               nanobind::cast<std::size_t>(shape[2]),
                               nanobind::cast<std::size_t>(shape[3])});
    for (std::size_t i = 0; i < twoBody.shape[0]; i++)
      for (std::size_t j = 0; j < twoBody.shape[1]; j++)
        for (std::size_t k = 0; k < twoBody.shape[2]; k++)
          for (std::size_t l = 0; l < twoBody.shape[3]; l++)
            twoBody(i, j, k, l) = nanobind::cast<double>(pyTwoBody.attr(
                "__getitem__")(nanobind::make_tuple(i, j, k, l)));

    // return a new molecular_hamiltonian
    return molecular_hamiltonian{
        spinOp,
        std::move(oneBody),
        std::move(twoBody),
        nanobind::cast<std::size_t>(openFermionMolecule.attr("n_electrons")),
        nanobind::cast<std::size_t>(openFermionMolecule.attr("n_orbitals")),
        nanobind::cast<double>(openFermionMolecule.attr("nuclear_repulsion")),
        nanobind::cast<double>(openFermionMolecule.attr("hf_energy")),
        nanobind::cast<double>(openFermionMolecule.attr("fci_energy"))};
  }
};

} // namespace
CUDAQ_REGISTER_TYPE(MoleculePackageDriver, PySCFPackageDriver, pyscf)
