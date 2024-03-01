/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "mps_simulation_state.h"

#include <cuComplex.h>

namespace nvqir {

MPSSimulationState::MPSSimulationState(TensorNetState *inState,
                                       const std::vector<MPSTensor> & mpsTensors) :
  // FIXME flip this to false to use MPS-only overlap
  TensorNetSimulationState(inState, /*auto-gen state vector stop-gap*/ true),
  m_mpsTensors(mpsTensors)
{
}

MPSSimulationState::~MPSSimulationState()
{
  deallocate();
}

double MPSSimulationState::overlap(const cudaq::SimulationState &other) {

  if (other.getDataShape() != getDataShape())
    throw std::runtime_error("[tensornet-state] overlap error - other state "
                             "dimension is not equal to this state dimension.");

  if(m_allSet) deallocateBackendStructures();

  const auto &mpsOther = dynamic_cast<const MPSSimulationState &>(other);
  const auto &mpsOtherTensors = mpsOther.m_mpsTensors;
  const int32_t mpsNumTensors = m_mpsTensors.size();
  assert(mpsNumTensors > 0);
  auto cutnHandle = state->getInternalContext();
  //auto quantumState = state->getInternalState();

  /*
  auto getNumModes = [mpsNumTensors] (int position) {
    const int32_t numModes = (position == 0 || position == (mpsNumTensors - 1)) ? 3 : 4;
    return numModes;
  };
  */

  // Create a tensor network descriptor for the overlap
  const int32_t numTensors = mpsNumTensors * 2;
  std::vector<int32_t> numModes(numTensors);
  std::vector<std::vector<int64_t>> tensExtents(numTensors);
  std::vector<cutensornetTensorQualifiers_t> tensAttr(numTensors);
  for(int i = 0; i < mpsNumTensors; ++i) {
    numModes[i] = m_mpsTensors[i].extents.size();
    numModes[mpsNumTensors + i] = mpsOtherTensors[i].extents.size();
    tensExtents[i] = m_mpsTensors[i].extents;
    tensExtents[mpsNumTensors + i] = mpsOtherTensors[i].extents;
    tensAttr[i] = cutensornetTensorQualifiers_t{0, 0, 0};
    tensAttr[mpsNumTensors + i] = cutensornetTensorQualifiers_t{0, 0, 0};
  }
  std::vector<int64_t> outExtents(mpsNumTensors);
  std::vector<int32_t> outModes(mpsNumTensors);
  std::vector<std::vector<int32_t>> tensModes(numTensors);
  int32_t umode = 0;
  for(int i = 0; i < mpsNumTensors; ++i) {
    if(i == 0) {
      outExtents[i] = m_mpsTensors[i].extents[0];
      outModes[i] = umode;
      tensModes[i] = std::initializer_list<int32_t>{umode, umode+1};
      umode += 2;
    }else if(i == (mpsNumTensors - 1)) {
      outExtents[i] = m_mpsTensors[i].extents[1];
      outModes[i] = umode;
      tensModes[i] = std::initializer_list<int32_t>{umode-1, umode};
      umode += 1;
    }else{
      outExtents[i] = m_mpsTensors[i].extents[1];
      outModes[i] = umode;
      tensModes[i] = std::initializer_list<int32_t>{umode-1, umode, umode+1};
      umode += 2;
    }
  }
  int32_t lmode = umode;
  umode = 0;
  for(int i = 0; i < mpsNumTensors; ++i) {
    if(i == 0) {
      tensModes[mpsNumTensors + i] = std::initializer_list<int32_t>{umode, lmode};
      umode += 2;
      lmode += 1;
    }else if(i == (mpsNumTensors - 1)) {
      tensModes[mpsNumTensors + i] = std::initializer_list<int32_t>{lmode-1, umode};
      umode += 1;
    }else{
      tensModes[mpsNumTensors + i] = std::initializer_list<int32_t>{lmode-1, umode, lmode};
      umode += 2;
      lmode += 1;
    }
  }
  std::size_t overlapSize = 0;
  cutensornetComputeType_t computeType;
  cudaDataType_t dataType;
  const auto prec = getPrecision();
  if(prec == precision::fp32) {
    overlapSize = sizeof(cuFloatComplex);
    dataType = CUDA_C_32F;
    computeType = CUTENSORNET_COMPUTE_32F;
  }else if (prec == precision::fp64){
    overlapSize = sizeof(cuDoubleComplex);
    dataType = CUDA_C_64F;
    computeType = CUTENSORNET_COMPUTE_64F;
  }
  std::vector<const int64_t*> extentsIn(numTensors);
  for(int i = 0; i < numTensors; ++i) {
    extentsIn[i] = tensExtents[i].data();
  }
  std::vector<const int32_t*> modesIn(numTensors);
  for(int i = 0; i < numTensors; ++i) {
    modesIn[i] = tensModes[i].data();
  }
  HANDLE_CUTN_ERROR(cutensornetCreateNetworkDescriptor(cutnHandle,
    numTensors, numModes.data(), extentsIn.data(), NULL, modesIn.data(), tensAttr.data(),
    mpsNumTensors, outExtents.data(), NULL, outModes.data(), dataType, computeType, &m_tnDescr));

  // Determine the contraction path and create the contraction plan
  HANDLE_CUTN_ERROR(cutensornetCreateContractionOptimizerConfig(cutnHandle, &m_tnConfig));
  HANDLE_CUTN_ERROR(cutensornetCreateContractionOptimizerInfo(cutnHandle, m_tnDescr, &m_tnPath));
  assert(m_scratchPad.scratchSize > 0);
  HANDLE_CUTN_ERROR(cutensornetContractionOptimize(cutnHandle, m_tnDescr, m_tnConfig,
    m_scratchPad.scratchSize, &m_tnPath));
  cutensornetWorkspaceDescriptor_t workDesc;
  HANDLE_CUTN_ERROR(cutensornetCreateWorkspaceDescriptor(cutnHandle, &workDesc));
  int64_t requiredWorkspaceSize = 0;
  HANDLE_CUTN_ERROR(cutensornetWorkspaceComputeContractionSizes(cutnHandle,
    m_tnDescr, m_tnPath, workDesc));
  HANDLE_CUTN_ERROR(cutensornetWorkspaceGetMemorySize(cutnHandle,
    workDesc, CUTENSORNET_WORKSIZE_PREF_RECOMMENDED, CUTENSORNET_MEMSPACE_DEVICE,
    CUTENSORNET_WORKSPACE_SCRATCH, &requiredWorkspaceSize));
  assert(requiredWorkspaceSize > 0);
  assert(static_cast<std::size_t>(requiredWorkspaceSize) <= m_scratchPad.scratchSize);
  HANDLE_CUTN_ERROR(cutensornetWorkspaceSetMemory(cutnHandle,
    workDesc, CUTENSORNET_MEMSPACE_DEVICE, CUTENSORNET_WORKSPACE_SCRATCH,
    m_scratchPad.d_scratch, requiredWorkspaceSize));
  HANDLE_CUTN_ERROR(cutensornetCreateContractionPlan(cutnHandle,
    m_tnDescr, m_tnPath, workDesc, &m_tnPlan));

  // Compute the unnormalized overlap
  std::vector<const void*> rawDataIn(numTensors);
  for(int i = 0; i < mpsNumTensors; ++i) {
    rawDataIn[i] = m_mpsTensors[i].deviceData;
    rawDataIn[mpsNumTensors + i] = mpsOtherTensors[i].deviceData;
  }
  HANDLE_CUDA_ERROR(cudaMalloc(&m_dOverlap, overlapSize));
  m_allSet = true;
  HANDLE_CUTN_ERROR(cutensornetContractSlices(cutnHandle,
    m_tnPlan, rawDataIn.data(), m_dOverlap, 0, workDesc, NULL, 0x0));
  // Get the overlap value back to Host
  double overlapReal = 0.0;
  if(prec == precision::fp32) {
    cuFloatComplex overlapValue;
    HANDLE_CUDA_ERROR(cudaMemcpy(&overlapValue, m_dOverlap, overlapSize, cudaMemcpyDeviceToHost));
    overlapReal = static_cast<double>(cuCrealf(overlapValue));
  }else if(prec == precision::fp64) {
    cuDoubleComplex overlapValue;
    HANDLE_CUDA_ERROR(cudaMemcpy(&overlapValue, m_dOverlap, overlapSize, cudaMemcpyDeviceToHost));
    overlapReal = cuCreal(overlapValue);
  }

  return overlapReal;
}

void MPSSimulationState::deallocate()
{
  deallocateBackendStructures();
  for (auto &tensor : m_mpsTensors)
    HANDLE_CUDA_ERROR(cudaFree(tensor.deviceData));
  m_mpsTensors.clear();
}

void MPSSimulationState::deallocateBackendStructures()
{
  if(m_allSet) {
    HANDLE_CUDA_ERROR(cudaFree(m_dOverlap));
    HANDLE_CUTN_ERROR(cutensornetDestroyContractionPlan(m_tnPlan));
    HANDLE_CUTN_ERROR(cutensornetDestroyContractionOptimizerInfo(m_tnPath));
    HANDLE_CUTN_ERROR(cutensornetDestroyContractionOptimizerConfig(m_tnConfig));
    HANDLE_CUTN_ERROR(cutensornetDestroyNetworkDescriptor(m_tnDescr));
    m_allSet = false;
  }
}

} // namespace nvqir