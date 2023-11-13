/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "simulator_cutensornet.h"
#include "cudaq.h"
#include "cutensornet.h"
#include "external_plugin.h"
#include "tensornet_spin_op.h"

namespace nvqir {

SimulatorTensorNetBase::SimulatorTensorNetBase() {
  int numDevices{0};
  HANDLE_CUDA_ERROR(cudaGetDeviceCount(&numDevices));
  // we assume that the processes are mapped to nodes in contiguous chunks
  const int deviceId =
      cudaq::mpi::is_initialized() ? cudaq::mpi::rank() % numDevices : 0;
  HANDLE_CUDA_ERROR(cudaSetDevice(deviceId));
  HANDLE_CUTN_ERROR(cutensornetCreate(&m_cutnHandle));
}

static std::vector<std::complex<double>>
generateFullGateTensor(std::size_t num_control_qubits,
                       const std::vector<std::complex<double>> &target_gate) {
  const auto mat_size = target_gate.size();
  // Must be square matrix (n x n)
  assert(std::ceil(std::sqrt(mat_size)) == std::floor(std::sqrt(mat_size)) &&
         "Input matrix is not a square matrix.");
  // Dim == rows == cols
  const std::size_t target_gate_dim = std::ceil(std::sqrt(mat_size));
  // Number of qubits
  const std::size_t num_target_qubits = std::bit_width(target_gate_dim) - 1;
  assert(target_gate_dim == (1u << num_target_qubits) &&
         "Gate matrix dimension is not 2^N");
  // No control => return the input matrix
  if (num_control_qubits == 0)
    return target_gate;
  // Expand the matrix
  const std::size_t full_dim =
      (1UL << (num_control_qubits + num_target_qubits));
  std::vector<std::complex<double>> gate_tensor(full_dim * full_dim,
                                                {0.0, 0.0});
  std::size_t offset = 0;
  // Set the diagonal elements:
  for (int i = 0; i < static_cast<int>(full_dim - target_gate_dim); ++i) {
    gate_tensor[offset] = {1.0, 0.0};
    offset += (full_dim + 1);
  }
  // Set the target gate matrix:
  for (std::size_t row = 0; row < target_gate_dim; ++row) {
    for (std::size_t col = 0; col < target_gate_dim; ++col) {
      const auto org_idx = row * target_gate_dim + col;
      // The anchor point of the gate matrix inside the expanded matrix (lower
      // right, i.e., shift up and left by target_gate_dim)
      const auto block_anchor = full_dim - target_gate_dim;
      // Row and column idxs in the expanded matrix
      const auto block_row_idx = block_anchor + row;
      const auto block_col_idx = block_anchor + col;
      const auto expanded_idx = block_row_idx * full_dim + block_col_idx;
      gate_tensor[expanded_idx] = target_gate[org_idx];
    }
  }
  return gate_tensor;
}

void SimulatorTensorNetBase::applyGate(const GateApplicationTask &task) {
  const auto &controls = task.controls;
  const auto &targets = task.targets;
  // Cache name lookup key:
  // <GateName>_<num control>_<Param>
  const std::string gateKey =
      task.operationName + "_" + std::to_string(controls.size()) + "_" + [&]() {
        std::stringstream paramsSs;
        for (const auto &param : task.parameters) {
          paramsSs << param << "_";
        }
        return paramsSs.str();
      }();
  const auto iter = m_gateDeviceMemCache.find(gateKey);

  // This is the first time we see this gate, allocate device mem and cache it.
  if (iter == m_gateDeviceMemCache.end()) {
    void *dMem = allocateGateMatrix(
        generateFullGateTensor(controls.size(), task.matrix));
    m_gateDeviceMemCache[gateKey] = dMem;
  }
  std::vector<int32_t> qubits;
  for (const auto &qId : controls)
    qubits.emplace_back(qId);
  for (const auto &qId : targets)
    qubits.emplace_back(qId);
  m_state->applyGate(qubits, m_gateDeviceMemCache[gateKey]);
}

void SimulatorTensorNetBase::applyExpPauli(
    double theta, const std::vector<std::size_t> &controls,
    const std::vector<std::size_t> &qubitIds, const cudaq::spin_op &op) {
  if (op.is_identity()) {
    if (controls.empty()) {
      // exp(i*theta*Id) is noop if this is not a controlled gate.
      return;
    } else {
      // Throw an error if this exp_pauli(i*theta*Id) becomes a non-trivial gate
      // due to control qubits.
      // FIXME: revisit this once
      // https://github.com/NVIDIA/cuda-quantum/issues/483 is implemented.
      throw std::logic_error("Applying controlled global phase via exp_pauli "
                             "of identity operator is not supported");
    }
  }

  flushGateQueue();
  cudaq::info(" [{} decomposing] exp_pauli({}, {})", name(), theta,
              op.to_string(false));
  std::vector<std::size_t> qubitSupport;
  std::vector<std::function<void(bool)>> basisChange;
  op.for_each_pauli([&](cudaq::pauli type, std::size_t qubitIdx) {
    if (type != cudaq::pauli::I)
      qubitSupport.push_back(qubitIds[qubitIdx]);

    if (type == cudaq::pauli::Y)
      basisChange.emplace_back([&, qubitIdx](bool reverse) {
        rx(!reverse ? M_PI_2 : -M_PI_2, qubitIds[qubitIdx]);
      });
    else if (type == cudaq::pauli::X)
      basisChange.emplace_back([&, qubitIdx](bool) { h(qubitIds[qubitIdx]); });
  });

  if (!basisChange.empty())
    for (auto &basis : basisChange)
      basis(false);

  std::vector<std::pair<std::size_t, std::size_t>> toReverse;
  for (std::size_t i = 0; i < qubitSupport.size() - 1; i++) {
    x({qubitSupport[i]}, qubitSupport[i + 1]);
    toReverse.emplace_back(qubitSupport[i], qubitSupport[i + 1]);
  }

  // Since this is a compute-action-uncompute type circuit, we only need to
  // apply control on this rz gate.
  rz(-2.0 * theta, controls, qubitSupport.back());

  std::reverse(toReverse.begin(), toReverse.end());
  for (auto &[i, j] : toReverse)
    x({i}, j);

  if (!basisChange.empty()) {
    std::reverse(basisChange.begin(), basisChange.end());
    for (auto &basis : basisChange)
      basis(true);
  }
}

/// @brief Reset the state of a given qubit to zero
void SimulatorTensorNetBase::resetQubit(const std::size_t qubitIdx) {
  flushGateQueue();
  LOG_API_TIME();
  // Prepare the state before RDM calculation
  prepareQubitTensorState();
  const auto rdm = m_state->computeRDM({static_cast<int32_t>(qubitIdx)});
  assert(rdm.size() == 4);
  const double prob0 = rdm[0].real();
  cudaq::info("Reset qubit {} with prob(|0>) = {}", qubitIdx, prob0);
  // If this is a zero state, no need to do anything.
  if (std::abs(1.0 - prob0) < 1e-9)
    return;

  // One state => flip
  if (prob0 < 1e-9) {
    x(qubitIdx);
    return;
  }

  // Otherwise, perform projection
  // Reset == project back to 0
  const std::vector<std::complex<double>> projected0Mat{
      {1.0 / std::sqrt(prob0), 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
  const std::string projKey = std::string("Project") + "_" +
                              std::to_string(false) + "_Prob" +
                              std::to_string(prob0);
  const auto iter = m_gateDeviceMemCache.find(projKey);
  if (iter == m_gateDeviceMemCache.end()) {
    void *d_gateProj{nullptr};
    HANDLE_CUDA_ERROR(
        cudaMalloc(&d_gateProj, 4 * sizeof(std::complex<double>)));
    HANDLE_CUDA_ERROR(cudaMemcpy(d_gateProj, projected0Mat.data(),
                                 4 * sizeof(std::complex<double>),
                                 cudaMemcpyHostToDevice));
    m_gateDeviceMemCache[projKey] = d_gateProj;
  }

  m_state->applyQubitProjector(m_gateDeviceMemCache[projKey], qubitIdx);
}

/// @brief Perform a measurement on a given qubit
bool SimulatorTensorNetBase::measureQubit(const std::size_t qubitIdx) {
  LOG_API_TIME();
  // Prepare the state before RDM calculation
  prepareQubitTensorState();
  const auto rdm = m_state->computeRDM({static_cast<int32_t>(qubitIdx)});
  assert(rdm.size() == 4);
  const double prob0 = rdm[0].real();
  const double prob1 = rdm[3].real();
  assert(std::abs(1.0 - (prob0 + prob1)) < 1e-9);
  const double rand = randomValues(1, 1.0)[0];
  const bool resultBool = (rand > prob0);
  const std::vector<std::complex<double>> projected0Mat{
      {1.0 / std::sqrt(prob0), 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
  const std::vector<std::complex<double>> projected1Mat{
      {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0 / std::sqrt(prob1), 0.0}};

  const std::string projKey = std::string("Project") + "_" +
                              std::to_string(resultBool) + "_Prob" +
                              std::to_string(resultBool ? prob1 : prob0);
  const auto iter = m_gateDeviceMemCache.find(projKey);
  if (iter == m_gateDeviceMemCache.end()) {
    void *d_gateProj{nullptr};
    HANDLE_CUDA_ERROR(
        cudaMalloc(&d_gateProj, 4 * sizeof(std::complex<double>)));
    HANDLE_CUDA_ERROR(cudaMemcpy(
        d_gateProj, resultBool ? projected1Mat.data() : projected0Mat.data(),
        4 * sizeof(std::complex<double>), cudaMemcpyHostToDevice));
    m_gateDeviceMemCache[projKey] = d_gateProj;
  }
  m_state->applyQubitProjector(m_gateDeviceMemCache[projKey], qubitIdx);
  return resultBool;
}

/// @brief Sample a subset of qubits
cudaq::ExecutionResult
SimulatorTensorNetBase::sample(const std::vector<std::size_t> &measuredBits,
                               const int shots) {
  LOG_API_TIME();
  std::vector<int32_t> measuredBitIds;
  std::transform(measuredBits.begin(), measuredBits.end(),
                 std::back_inserter(measuredBitIds),
                 [](std::size_t idx) { return static_cast<int32_t>(idx); });
  if (shots < 1) {
    cudaq::spin_op::spin_op_term allZTerm(2 * m_state->getNumQubits(), 0);
    for (const auto &m : measuredBits)
      allZTerm.at(m_state->getNumQubits() + m) = 1;
    cudaq::spin_op allZ(allZTerm, 1.0);
    // Just compute the expected value on <Z...Z>
    return observe(allZ);
  }

  prepareQubitTensorState();
  const auto samples = m_state->sample(measuredBitIds, shots);
  cudaq::ExecutionResult counts(samples);
  double expVal = 0.0;
  // Compute the expectation value from the counts
  for (auto &kv : counts.counts) {
    auto par = cudaq::sample_result::has_even_parity(kv.first);
    auto p = kv.second / (double)shots;
    if (!par) {
      p = -p;
    }
    expVal += p;
  }

  counts.expectationValue = expVal;

  return counts;
}

bool SimulatorTensorNetBase::canHandleObserve() { return true; }

static nvqir::CutensornetExecutor *getPluginInstance() {
  using GetPluginFunction = nvqir::CutensornetExecutor *(*)();
  auto handle = dlopen(NULL, RTLD_LAZY);
  GetPluginFunction fcn =
      (GetPluginFunction)(intptr_t)dlsym(handle, "getCutnExecutor");
  if (!fcn) {
    cudaq::info("Externally provided cutensornet plugin not found.");
    return nullptr;
  }

  cudaq::info("Successfully loaded the cutensornet plugin.");
  return fcn();
}
/// @brief Evaluate the expectation value of a given observable
cudaq::ExecutionResult
SimulatorTensorNetBase::observe(const cudaq::spin_op &ham) {
  LOG_API_TIME();
  prepareQubitTensorState();
  auto *cutnExtension = getPluginInstance();
  if (cutnExtension) {
    const auto [terms, coeffs] = ham.get_raw_data();
    const auto termExpVals =
        cutnExtension->computeExpVals(m_cutnHandle, m_state->getInternalState(),
                                      m_state->getNumQubits(), terms);
    std::complex<double> expVal = 0.0;
    for (std::size_t i = 0; i < terms.size(); ++i) {
      expVal += (coeffs[i] * termExpVals[i]);
    }
    return cudaq::ExecutionResult({}, expVal.real());
  } else {
    TensorNetworkSpinOp spinOp(ham, m_cutnHandle);
    std::complex<double> expVal =
        m_state->computeExpVal(spinOp.getNetworkOperator());
    expVal += spinOp.getIdentityTermOffset();
    return cudaq::ExecutionResult({}, expVal.real());
  }
}

/// @brief Return the state vector data
cudaq::State SimulatorTensorNetBase::getStateData() {
  LOG_API_TIME();
  if (m_state->getNumQubits() > 64)
    throw std::runtime_error("State vector data is too large.");
  // Handle empty state (e.g., no qubit allocation)
  if (!m_state)
    return cudaq::State{{0}, {}};

  const uint64_t svDim = 1ull << m_state->getNumQubits();
  return cudaq::State{{svDim}, m_state->getStateVector()};
}

nvqir::CircuitSimulator *SimulatorTensorNetBase::clone() { return nullptr; }
void SimulatorTensorNetBase::addQubitsToState(std::size_t count) {
  LOG_API_TIME();
  if (!m_state)
    m_state = std::make_unique<TensorNetState>(count, m_cutnHandle);
  else if (gateQueue.empty())
    m_state = std::make_unique<TensorNetState>(m_state->getNumQubits() + count,
                                               m_cutnHandle);
  else
    throw std::runtime_error("Expand qubit register is not supported!");
}
void SimulatorTensorNetBase::addQubitToState() { addQubitsToState(1); }

/// @brief Destroy the entire qubit register
void SimulatorTensorNetBase::deallocateStateImpl() { m_state.reset(); }

/// @brief Reset all qubits to zero
void SimulatorTensorNetBase::setToZeroState() {
  LOG_API_TIME();
  const auto numQubits = m_state->getNumQubits();
  m_state.reset();
  // Re-create a zero state of the same size
  m_state = std::make_unique<TensorNetState>(numQubits, m_cutnHandle);
}

SimulatorTensorNetBase::~SimulatorTensorNetBase() {
  m_state.reset();
  for (const auto &[key, dMem] : m_gateDeviceMemCache)
    HANDLE_CUDA_ERROR(cudaFree(dMem));

  // Finalize the cuTensorNet library
  HANDLE_CUTN_ERROR(cutensornetDestroy(m_cutnHandle));
}
} // end namespace nvqir
