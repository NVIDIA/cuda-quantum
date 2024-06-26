/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "tensornet_spin_op.h"
#include "tensornet_utils.h"
#include "timing_utils.h"

namespace nvqir {

TensorNetworkSpinOp::TensorNetworkSpinOp(const cudaq::spin_op &spinOp,
                                         cutensornetHandle_t handle)
    : m_cutnHandle(handle) {
  LOG_API_TIME();
  const std::vector<int64_t> qubitDims(spinOp.num_qubits(), 2);
  HANDLE_CUTN_ERROR(cutensornetCreateNetworkOperator(
      m_cutnHandle, spinOp.num_qubits(), qubitDims.data(), CUDA_C_64F,
      &m_cutnNetworkOperator));
  // Heuristic threshold to perform direct observable calculation.
  // If the number of qubits in the spin_op is small, it is more efficient to
  // directly convert the spin_op into a matrix and perform the contraction
  // <psi| H |psi> in one go rather than summing over term-by-term contractions.
  constexpr std::size_t NUM_QUBITS_THRESHOLD_DIRECT_OBS = 10;
  if (spinOp.num_qubits() < NUM_QUBITS_THRESHOLD_DIRECT_OBS) {
    const auto hamMat = spinOp.to_matrix();
    std::vector<std::complex<double>> opMat(
        hamMat.data(), hamMat.data() + hamMat.rows() * hamMat.cols());
    void *opMat_d{nullptr};
    HANDLE_CUDA_ERROR(
        cudaMalloc(&opMat_d, opMat.size() * sizeof(std::complex<double>)));
    HANDLE_CUDA_ERROR(cudaMemcpy(opMat_d, opMat.data(),
                                 opMat.size() * sizeof(std::complex<double>),
                                 cudaMemcpyHostToDevice));
    m_mat_d.emplace_back(opMat_d);
    const std::vector<int32_t> numModes = {
        static_cast<int32_t>(spinOp.num_qubits())};
    std::vector<const void *> pauliTensorData = {opMat_d};
    std::vector<int32_t> stateModes(spinOp.num_qubits());
    std::iota(stateModes.begin(), stateModes.end(), 0);
    std::vector<const int32_t *> dataStateModes = {stateModes.data()};
    HANDLE_CUTN_ERROR(cutensornetNetworkOperatorAppendProduct(
        m_cutnHandle, m_cutnNetworkOperator,
        /*coefficient*/ cuDoubleComplex(1.0, 0.0), pauliTensorData.size(),
        numModes.data(), dataStateModes.data(),
        /*tensorModeStrides*/ nullptr, pauliTensorData.data(),
        /*componentId*/ nullptr));
  } else {
    // Initialize device mem for Pauli matrices
    constexpr std::complex<double> PauliI_h[4] = {
        {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}};

    constexpr std::complex<double> PauliX_h[4]{
        {0.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}};

    constexpr std::complex<double> PauliY_h[4]{
        {0.0, 0.0}, {0.0, -1.0}, {0.0, 1.0}, {0.0, 0.0}};

    constexpr std::complex<double> PauliZ_h[4]{
        {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {-1.0, 0.0}};

    for (const auto &pauli :
         {cudaq::pauli::I, cudaq::pauli::X, cudaq::pauli::Y, cudaq::pauli::Z}) {
      void *d_mat{nullptr};
      const auto mat = [&]() {
        switch (pauli) {
        case cudaq::pauli::I:
          return PauliI_h;
        case cudaq::pauli::X:
          return PauliX_h;
        case cudaq::pauli::Y:
          return PauliY_h;
        case cudaq::pauli::Z:
          return PauliZ_h;
        }
        __builtin_unreachable();
      }();
      HANDLE_CUDA_ERROR(cudaMalloc(&d_mat, 4 * sizeof(std::complex<double>)));
      HANDLE_CUDA_ERROR(cudaMemcpy(d_mat, mat, 4 * sizeof(std::complex<double>),
                                   cudaMemcpyHostToDevice));
      m_pauli_d[pauli] = d_mat;
    }

    spinOp.for_each_term([&](cudaq::spin_op &term) {
      if (term.is_identity()) {
        // Note: we don't add I Pauli.
        m_identityCoeff = term.get_coefficient();
        return;
      }

      const cuDoubleComplex termCoeff{term.get_coefficient().real(),
                                      term.get_coefficient().imag()};
      std::vector<const void *> pauliTensorData;
      std::vector<std::vector<int32_t>> stateModes;
      term.for_each_pauli([&](cudaq::pauli p, std::size_t idx) {
        if (p != cudaq::pauli::I) {
          stateModes.emplace_back(
              std::vector<int32_t>{static_cast<int32_t>(idx)});
          pauliTensorData.emplace_back(m_pauli_d[p]);
        }
      });

      std::vector<int32_t> numModes(pauliTensorData.size(), 1);
      std::vector<const int32_t *> dataStateModes;
      for (const auto &stateMode : stateModes) {
        dataStateModes.emplace_back(stateMode.data());
      }
      HANDLE_CUTN_ERROR(cutensornetNetworkOperatorAppendProduct(
          m_cutnHandle, m_cutnNetworkOperator, termCoeff,
          pauliTensorData.size(), numModes.data(), dataStateModes.data(),
          /*tensorModeStrides*/ nullptr, pauliTensorData.data(),
          /*componentId*/ nullptr));
    });
  }
}

TensorNetworkSpinOp::~TensorNetworkSpinOp() {
  HANDLE_CUTN_ERROR(cutensornetDestroyNetworkOperator(m_cutnNetworkOperator));
  for (const auto &[pauli, dMem] : m_pauli_d)
    HANDLE_CUDA_ERROR(cudaFree(dMem));
  for (const auto &dMem : m_mat_d)
    HANDLE_CUDA_ERROR(cudaFree(dMem));
}

} // namespace nvqir
