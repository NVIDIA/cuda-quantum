/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "MklqMetalRuntime.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace nvqir::mklq {
namespace {

struct MetalComplexFloat {
  float real;
  float imag;
};

struct SingleQubitParams {
  std::uint64_t targetMask;
  std::uint64_t lowMask;
  std::uint32_t controlCount;
  std::uint32_t padding = 0;
};

struct TwoQubitParams {
  std::uint64_t targetMask0;
  std::uint64_t targetMask1;
  std::uint32_t firstTarget;
  std::uint32_t secondTarget;
  std::uint32_t controlCount;
  std::uint32_t padding = 0;
};

struct CollapseParams {
  std::uint64_t targetMask;
  float inverseNorm;
  std::uint32_t keepOne;
};

struct MeasurementProbabilityParams {
  std::uint64_t targetMask;
  std::uint32_t stateSize;
  std::uint32_t padding = 0;
};

struct MarginalProbabilityParams {
  std::uint32_t stateSize;
  std::uint32_t qubitCount;
  std::uint32_t groupCount;
  std::uint32_t padding = 0;
};

constexpr const char *metalKernelSource = R"metal(
#include <metal_stdlib>
using namespace metal;

struct ComplexFloat {
  float real;
  float imag;
};

struct SingleQubitParams {
  ulong targetMask;
  ulong lowMask;
  uint controlCount;
  uint padding;
};

struct TwoQubitParams {
  ulong targetMask0;
  ulong targetMask1;
  uint firstTarget;
  uint secondTarget;
  uint controlCount;
  uint padding;
};

struct CollapseParams {
  ulong targetMask;
  float inverseNorm;
  uint keepOne;
};

struct MeasurementProbabilityParams {
  ulong targetMask;
  uint stateSize;
  uint padding;
};

struct MarginalProbabilityParams {
  uint stateSize;
  uint qubitCount;
  uint groupCount;
  uint padding;
};

inline ComplexFloat cadd(ComplexFloat lhs, ComplexFloat rhs) {
  return ComplexFloat{lhs.real + rhs.real, lhs.imag + rhs.imag};
}

inline ComplexFloat cmul(ComplexFloat lhs, ComplexFloat rhs) {
  return ComplexFloat{
      lhs.real * rhs.real - lhs.imag * rhs.imag,
      lhs.real * rhs.imag + lhs.imag * rhs.real};
}

kernel void mklq_apply_single_qubit(
    device ComplexFloat *state [[buffer(0)]],
    constant ComplexFloat *matrix [[buffer(1)]],
    constant ulong *controlMasks [[buffer(2)]],
    constant SingleQubitParams &params [[buffer(3)]],
    uint pair [[thread_position_in_grid]]) {
  const ulong pairIndex = pair;
  const ulong zeroIndex =
      ((pairIndex & ~params.lowMask) << 1) | (pairIndex & params.lowMask);

  for (uint control = 0; control < params.controlCount; ++control)
    if ((zeroIndex & controlMasks[control]) == 0)
      return;

  const ulong oneIndex = zeroIndex | params.targetMask;
  const ComplexFloat zeroAmplitude = state[zeroIndex];
  const ComplexFloat oneAmplitude = state[oneIndex];

  state[zeroIndex] =
      cadd(cmul(matrix[0], zeroAmplitude), cmul(matrix[1], oneAmplitude));
  state[oneIndex] =
      cadd(cmul(matrix[2], zeroAmplitude), cmul(matrix[3], oneAmplitude));
}

inline ulong insertZeroBit(ulong value, uint bit) {
  const ulong lowMask = (ulong(1) << bit) - 1;
  return ((value & ~lowMask) << 1) | (value & lowMask);
}

kernel void mklq_apply_two_qubit(
    device ComplexFloat *state [[buffer(0)]],
    constant ComplexFloat *matrix [[buffer(1)]],
    constant ulong *controlMasks [[buffer(2)]],
    constant TwoQubitParams &params [[buffer(3)]],
    uint block [[thread_position_in_grid]]) {
  const ulong blockIndex = block;
  const ulong base = insertZeroBit(
      insertZeroBit(blockIndex, params.firstTarget), params.secondTarget);

  for (uint control = 0; control < params.controlCount; ++control)
    if ((base & controlMasks[control]) == 0)
      return;

  const ulong index0 = base;
  const ulong index1 = base | params.targetMask0;
  const ulong index2 = base | params.targetMask1;
  const ulong index3 = base | params.targetMask0 | params.targetMask1;

  const ComplexFloat amplitude0 = state[index0];
  const ComplexFloat amplitude1 = state[index1];
  const ComplexFloat amplitude2 = state[index2];
  const ComplexFloat amplitude3 = state[index3];

  state[index0] = cadd(
      cadd(cmul(matrix[0], amplitude0), cmul(matrix[1], amplitude1)),
      cadd(cmul(matrix[2], amplitude2), cmul(matrix[3], amplitude3)));
  state[index1] = cadd(
      cadd(cmul(matrix[4], amplitude0), cmul(matrix[5], amplitude1)),
      cadd(cmul(matrix[6], amplitude2), cmul(matrix[7], amplitude3)));
  state[index2] = cadd(
      cadd(cmul(matrix[8], amplitude0), cmul(matrix[9], amplitude1)),
      cadd(cmul(matrix[10], amplitude2), cmul(matrix[11], amplitude3)));
  state[index3] = cadd(
      cadd(cmul(matrix[12], amplitude0), cmul(matrix[13], amplitude1)),
      cadd(cmul(matrix[14], amplitude2), cmul(matrix[15], amplitude3)));
}

kernel void mklq_fill_probabilities(
    device const ComplexFloat *state [[buffer(0)]],
    device float *probabilities [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
  const ComplexFloat amplitude = state[index];
  probabilities[index] =
      amplitude.real * amplitude.real + amplitude.imag * amplitude.imag;
}

kernel void mklq_fill_marginal_probabilities(
    device const ComplexFloat *state [[buffer(0)]],
    constant ulong *qubitMasks [[buffer(1)]],
    device float *partialSums [[buffer(2)]],
    constant MarginalProbabilityParams &params [[buffer(3)]],
    threadgroup float *scratch [[threadgroup(0)]],
    uint localIndex [[thread_index_in_threadgroup]],
    uint3 groupPosition [[threadgroup_position_in_grid]]) {
  const uint index = groupPosition.x * 256u + localIndex;
  const uint targetOutcome = groupPosition.y;
  float value = 0.0f;

  if (index < params.stateSize) {
    uint outcome = 0u;
    for (uint bit = 0u; bit < params.qubitCount; ++bit)
      if ((ulong(index) & qubitMasks[bit]) != 0)
        outcome |= (1u << bit);

    if (outcome == targetOutcome) {
      const ComplexFloat amplitude = state[index];
      value = amplitude.real * amplitude.real +
              amplitude.imag * amplitude.imag;
    }
  }

  scratch[localIndex] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = 128u; stride > 0u; stride >>= 1u) {
    if (localIndex < stride)
      scratch[localIndex] += scratch[localIndex + stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (localIndex == 0)
    partialSums[ulong(targetOutcome) * params.groupCount + groupPosition.x] =
        scratch[0];
}

kernel void mklq_measure_qubit_probability(
    device const ComplexFloat *state [[buffer(0)]],
    device float *partialSums [[buffer(1)]],
    constant MeasurementProbabilityParams &params [[buffer(2)]],
    threadgroup float *scratch [[threadgroup(0)]],
    uint localIndex [[thread_index_in_threadgroup]],
    uint3 groupPosition [[threadgroup_position_in_grid]]) {
  const uint index = groupPosition.x * 256u + localIndex;
  float value = 0.0f;
  if (index < params.stateSize &&
      ((ulong(index) & params.targetMask) != 0)) {
    const ComplexFloat amplitude = state[index];
    value = amplitude.real * amplitude.real +
            amplitude.imag * amplitude.imag;
  }

  scratch[localIndex] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = 128u; stride > 0u; stride >>= 1u) {
    if (localIndex < stride)
      scratch[localIndex] += scratch[localIndex + stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (localIndex == 0)
    partialSums[groupPosition.x] = scratch[0];
}

kernel void mklq_collapse_qubit(
    device ComplexFloat *state [[buffer(0)]],
    constant CollapseParams &params [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
  const bool keep = ((ulong(index) & params.targetMask) != 0) ==
                    (params.keepOne != 0);
  if (keep) {
    state[index].real *= params.inverseNorm;
    state[index].imag *= params.inverseNorm;
  } else {
    state[index] = ComplexFloat{0.0f, 0.0f};
  }
}
)metal";

MetalDeviceInfo describeDevice(id<MTLDevice> device) {
  if (!device)
    return {};

  NSString *deviceName = [device name];
  MetalDeviceInfo info;
  info.available = true;
  if (deviceName)
    info.name = [deviceName UTF8String];
  info.lowPower = [device isLowPower];
  info.headless = [device isHeadless];
  info.removable = [device isRemovable];
  return info;
}

std::string describeError(NSError *error, const char *fallback) {
  if (!error)
    return fallback;
  NSString *description = [error localizedDescription];
  if (!description)
    return fallback;
  return [description UTF8String];
}

bool isPowerOfTwo(std::size_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

bool qubitWithinState(std::size_t qubit, std::size_t stateSize) {
  return qubit < std::numeric_limits<std::uint64_t>::digits &&
         (std::uint64_t{1} << qubit) < stateSize;
}

bool fitsKernelThreadIndex(std::size_t count) {
  return count <= std::numeric_limits<std::uint32_t>::max();
}

bool hasDuplicateControlOrTargetOverlap(const std::size_t *controlQubits,
                                        std::size_t controlCount,
                                        const std::size_t *targetQubits,
                                        std::size_t targetCount) {
  for (std::size_t control = 0; control < controlCount; ++control) {
    for (std::size_t previous = 0; previous < control; ++previous)
      if (controlQubits[control] == controlQubits[previous])
        return true;
    for (std::size_t target = 0; target < targetCount; ++target)
      if (controlQubits[control] == targetQubits[target])
        return true;
  }
  return false;
}

} // namespace

MetalDeviceInfo queryMetalDevice() {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    MetalDeviceInfo info = describeDevice(device);
    [device release];
    return info;
  }
}

struct MetalStateVectorExecutor::Impl {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> commandQueue = nil;
  id<MTLComputePipelineState> singleQubitPipeline = nil;
  id<MTLComputePipelineState> twoQubitPipeline = nil;
  id<MTLComputePipelineState> probabilityPipeline = nil;
  id<MTLComputePipelineState> marginalProbabilityPipeline = nil;
  id<MTLComputePipelineState> measurementProbabilityPipeline = nil;
  id<MTLComputePipelineState> collapsePipeline = nil;
  id<MTLBuffer> residentStateBuffer = nil;
  std::size_t residentStateSize = 0;
  MetalDeviceInfo info;
  std::string error;
  std::size_t singleQubitApplications = 0;
  std::size_t twoQubitApplications = 0;
  std::size_t probabilityFillApplications = 0;
  std::size_t marginalProbabilityApplications = 0;
  std::size_t measurementProbabilityApplications = 0;
  std::size_t measurementProbabilityReductionApplications = 0;
  std::size_t measurementCollapseApplications = 0;
  std::size_t residentUploads = 0;
  std::size_t residentDownloads = 0;

  Impl() {
    @autoreleasepool {
      device = MTLCreateSystemDefaultDevice();
      if (!device) {
        error = "Metal device is unavailable.";
        return;
      }

      info = describeDevice(device);
      commandQueue = [device newCommandQueue];
      if (!commandQueue) {
        error = "failed to create Metal command queue.";
        return;
      }

      NSString *source = [NSString stringWithUTF8String:metalKernelSource];
      NSError *libraryError = nil;
      id<MTLLibrary> library =
          [device newLibraryWithSource:source options:nil error:&libraryError];
      if (!library) {
        error = describeError(libraryError,
                              "failed to compile MKL-Q Metal kernels.");
        return;
      }

      id<MTLFunction> function =
          [library newFunctionWithName:@"mklq_apply_single_qubit"];
      if (!function) {
        [library release];
        error = "failed to load mklq_apply_single_qubit Metal kernel.";
        return;
      }

      NSError *pipelineError = nil;
      singleQubitPipeline =
          [device newComputePipelineStateWithFunction:function
                                                error:&pipelineError];
      [function release];

      if (!singleQubitPipeline) {
        [library release];
        error = describeError(pipelineError,
                              "failed to create Metal compute pipeline.");
        return;
      }

      function = [library newFunctionWithName:@"mklq_apply_two_qubit"];
      if (!function) {
        [library release];
        error = "failed to load mklq_apply_two_qubit Metal kernel.";
        return;
      }

      pipelineError = nil;
      twoQubitPipeline = [device newComputePipelineStateWithFunction:function
                                                               error:&pipelineError];
      [function release];

      if (!twoQubitPipeline) {
        [library release];
        error = describeError(pipelineError,
                              "failed to create Metal two-qubit compute pipeline.");
        return;
      }

      function = [library newFunctionWithName:@"mklq_fill_probabilities"];
      if (!function) {
        [library release];
        error = "failed to load mklq_fill_probabilities Metal kernel.";
        return;
      }

      pipelineError = nil;
      probabilityPipeline =
          [device newComputePipelineStateWithFunction:function
                                                error:&pipelineError];
      [function release];

      if (!probabilityPipeline) {
        [library release];
        error = describeError(
            pipelineError,
            "failed to create Metal probability-fill compute pipeline.");
        return;
      }

      function =
          [library newFunctionWithName:@"mklq_fill_marginal_probabilities"];
      if (!function) {
        [library release];
        error = "failed to load mklq_fill_marginal_probabilities Metal kernel.";
        return;
      }

      pipelineError = nil;
      marginalProbabilityPipeline =
          [device newComputePipelineStateWithFunction:function
                                                error:&pipelineError];
      [function release];

      if (!marginalProbabilityPipeline) {
        [library release];
        error = describeError(
            pipelineError,
            "failed to create Metal marginal-probability compute pipeline.");
        return;
      }

      function =
          [library newFunctionWithName:@"mklq_measure_qubit_probability"];
      if (!function) {
        [library release];
        error = "failed to load mklq_measure_qubit_probability Metal kernel.";
        return;
      }

      pipelineError = nil;
      measurementProbabilityPipeline =
          [device newComputePipelineStateWithFunction:function
                                                error:&pipelineError];
      [function release];

      if (!measurementProbabilityPipeline) {
        [library release];
        error = describeError(
            pipelineError,
            "failed to create Metal measurement probability compute pipeline.");
        return;
      }

      function = [library newFunctionWithName:@"mklq_collapse_qubit"];
      if (!function) {
        [library release];
        error = "failed to load mklq_collapse_qubit Metal kernel.";
        return;
      }

      pipelineError = nil;
      collapsePipeline =
          [device newComputePipelineStateWithFunction:function
                                                error:&pipelineError];
      [function release];
      [library release];

      if (!collapsePipeline) {
        error = describeError(
            pipelineError,
            "failed to create Metal collapse compute pipeline.");
        return;
      }
    }
  }

  ~Impl() {
    [residentStateBuffer release];
    [collapsePipeline release];
    [measurementProbabilityPipeline release];
    [marginalProbabilityPipeline release];
    [probabilityPipeline release];
    [twoQubitPipeline release];
    [singleQubitPipeline release];
    [commandQueue release];
    [device release];
  }

  bool available() const {
    return device && commandQueue && singleQubitPipeline && twoQubitPipeline &&
           probabilityPipeline && marginalProbabilityPipeline &&
           measurementProbabilityPipeline && collapsePipeline;
  }
};

MetalStateVectorExecutor::MetalStateVectorExecutor()
    : impl(std::make_unique<Impl>()) {}

MetalStateVectorExecutor::~MetalStateVectorExecutor() = default;

MetalStateVectorExecutor::MetalStateVectorExecutor(
    MetalStateVectorExecutor &&) noexcept = default;

MetalStateVectorExecutor &MetalStateVectorExecutor::operator=(
    MetalStateVectorExecutor &&) noexcept = default;

bool MetalStateVectorExecutor::available() const {
  return impl && impl->available();
}

MetalDeviceInfo MetalStateVectorExecutor::deviceInfo() const {
  return impl ? impl->info : MetalDeviceInfo{};
}

std::string MetalStateVectorExecutor::lastError() const {
  if (!impl)
    return "Metal executor is not initialized.";
  return impl->error;
}

bool MetalStateVectorExecutor::uploadState(const std::complex<double> *state,
                                           std::size_t stateSize) {
  if (!impl || !impl->available())
    return false;
  if (!state || stateSize == 0 || !isPowerOfTwo(stateSize)) {
    impl->error = "invalid Metal resident state upload input.";
    return false;
  }

  std::vector<MetalComplexFloat> gpuState(stateSize);
  for (std::size_t index = 0; index < stateSize; ++index)
    gpuState[index] = MetalComplexFloat{static_cast<float>(state[index].real()),
                                        static_cast<float>(state[index].imag())};

  @autoreleasepool {
    id<MTLBuffer> stateBuffer =
        [impl->device newBufferWithBytes:gpuState.data()
                                  length:gpuState.size() *
                                         sizeof(MetalComplexFloat)
                                 options:MTLResourceStorageModeShared];
    if (!stateBuffer) {
      impl->error = "failed to allocate Metal resident state buffer.";
      return false;
    }

    [impl->residentStateBuffer release];
    impl->residentStateBuffer = stateBuffer;
    impl->residentStateSize = stateSize;
  }

  impl->error.clear();
  ++impl->residentUploads;
  return true;
}

bool MetalStateVectorExecutor::downloadState(std::complex<double> *state,
                                             std::size_t stateSize) {
  if (!impl || !impl->available())
    return false;
  if (!state || !impl->residentStateBuffer ||
      stateSize != impl->residentStateSize) {
    impl->error = "invalid Metal resident state download input.";
    return false;
  }

  auto *residentState =
      reinterpret_cast<MetalComplexFloat *>([impl->residentStateBuffer contents]);
  for (std::size_t index = 0; index < stateSize; ++index)
    state[index] = {static_cast<double>(residentState[index].real),
                    static_cast<double>(residentState[index].imag)};

  impl->error.clear();
  ++impl->residentDownloads;
  return true;
}

void MetalStateVectorExecutor::releaseResidentState() {
  if (!impl)
    return;
  [impl->residentStateBuffer release];
  impl->residentStateBuffer = nil;
  impl->residentStateSize = 0;
}

bool MetalStateVectorExecutor::hasResidentState(std::size_t stateSize) const {
  return impl && impl->residentStateBuffer && impl->residentStateSize == stateSize;
}

bool MetalStateVectorExecutor::applyResidentSingleQubitGate(
    const std::complex<double> *matrix, const std::size_t *controlQubits,
    std::size_t controlCount, std::size_t targetQubit) {
  if (!impl || !impl->available())
    return false;
  if (!impl->residentStateBuffer || impl->residentStateSize < 2 ||
      !isPowerOfTwo(impl->residentStateSize) || !matrix) {
    impl->error = "invalid Metal resident single-qubit gate input.";
    return false;
  }
  if (!qubitWithinState(targetQubit, impl->residentStateSize)) {
    impl->error = "target qubit exceeds Metal state range.";
    return false;
  }
  if (!fitsKernelThreadIndex(impl->residentStateSize >> 1)) {
    impl->error = "state size exceeds Metal single-qubit thread index range.";
    return false;
  }
  if (controlCount > 0 && !controlQubits) {
    impl->error = "missing Metal control qubit input.";
    return false;
  }
  if (hasDuplicateControlOrTargetOverlap(controlQubits, controlCount,
                                         &targetQubit, 1)) {
    impl->error = "duplicate or overlapping Metal control qubit.";
    return false;
  }

  const auto targetMask = std::uint64_t{1} << targetQubit;
  const auto lowMask = targetMask - 1;
  std::vector<std::uint64_t> controlMasks;
  controlMasks.reserve(std::max<std::size_t>(controlCount, 1));
  for (std::size_t control = 0; control < controlCount; ++control) {
    if (!qubitWithinState(controlQubits[control], impl->residentStateSize)) {
      impl->error = "control qubit exceeds Metal state range.";
      return false;
    }
    controlMasks.push_back(std::uint64_t{1} << controlQubits[control]);
  }
  if (controlMasks.empty())
    controlMasks.push_back(0);

  std::array<MetalComplexFloat, 4> gpuMatrix;
  for (std::size_t index = 0; index < gpuMatrix.size(); ++index)
    gpuMatrix[index] =
        MetalComplexFloat{static_cast<float>(matrix[index].real()),
                          static_cast<float>(matrix[index].imag())};

  SingleQubitParams params{targetMask, lowMask,
                           static_cast<std::uint32_t>(controlCount), 0};

  @autoreleasepool {
    id<MTLBuffer> matrixBuffer =
        [impl->device newBufferWithBytes:gpuMatrix.data()
                                  length:gpuMatrix.size() *
                                         sizeof(MetalComplexFloat)
                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> controlsBuffer =
        [impl->device newBufferWithBytes:controlMasks.data()
                                  length:controlMasks.size() *
                                         sizeof(std::uint64_t)
                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> paramsBuffer =
        [impl->device newBufferWithBytes:&params
                                  length:sizeof(SingleQubitParams)
                                 options:MTLResourceStorageModeShared];

    if (!matrixBuffer || !controlsBuffer || !paramsBuffer) {
      [matrixBuffer release];
      [controlsBuffer release];
      [paramsBuffer release];
      impl->error = "failed to allocate Metal resident gate buffers.";
      return false;
    }

    id<MTLCommandBuffer> commandBuffer = [impl->commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    if (!commandBuffer || !encoder) {
      [matrixBuffer release];
      [controlsBuffer release];
      [paramsBuffer release];
      impl->error = "failed to create Metal resident command encoder.";
      return false;
    }

    [encoder setComputePipelineState:impl->singleQubitPipeline];
    [encoder setBuffer:impl->residentStateBuffer offset:0 atIndex:0];
    [encoder setBuffer:matrixBuffer offset:0 atIndex:1];
    [encoder setBuffer:controlsBuffer offset:0 atIndex:2];
    [encoder setBuffer:paramsBuffer offset:0 atIndex:3];

    const auto pairCount = impl->residentStateSize >> 1;
    const auto pipelineWidth =
        [impl->singleQubitPipeline maxTotalThreadsPerThreadgroup];
    const auto threadsPerThreadgroup =
        std::max<NSUInteger>(1, std::min<NSUInteger>(pipelineWidth, pairCount));
    [encoder dispatchThreads:MTLSizeMake(pairCount, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(threadsPerThreadgroup, 1, 1)];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    const auto status = [commandBuffer status];
    if (status != MTLCommandBufferStatusCompleted) {
      impl->error =
          describeError([commandBuffer error],
                        "Metal resident command buffer failed.");
      [matrixBuffer release];
      [controlsBuffer release];
      [paramsBuffer release];
      return false;
    }

    [matrixBuffer release];
    [controlsBuffer release];
    [paramsBuffer release];
  }

  impl->error.clear();
  ++impl->singleQubitApplications;
  return true;
}

bool MetalStateVectorExecutor::applySingleQubitGate(
    std::complex<double> *state, std::size_t stateSize,
    const std::complex<double> *matrix, const std::size_t *controlQubits,
    std::size_t controlCount, std::size_t targetQubit) {
  if (!impl || !impl->available())
    return false;
  if (!state || !matrix || stateSize < 2 || !isPowerOfTwo(stateSize)) {
    impl->error = "invalid Metal single-qubit gate input.";
    return false;
  }
  if (!qubitWithinState(targetQubit, stateSize)) {
    impl->error = "target qubit exceeds Metal state range.";
    return false;
  }
  if (!fitsKernelThreadIndex(stateSize >> 1)) {
    impl->error = "state size exceeds Metal single-qubit thread index range.";
    return false;
  }
  if (controlCount > 0 && !controlQubits) {
    impl->error = "missing Metal control qubit input.";
    return false;
  }
  if (hasDuplicateControlOrTargetOverlap(controlQubits, controlCount,
                                         &targetQubit, 1)) {
    impl->error = "duplicate or overlapping Metal control qubit.";
    return false;
  }

  const auto targetMask = std::uint64_t{1} << targetQubit;
  const auto lowMask = targetMask - 1;
  std::vector<std::uint64_t> controlMasks;
  controlMasks.reserve(std::max<std::size_t>(controlCount, 1));
  for (std::size_t control = 0; control < controlCount; ++control) {
    if (!qubitWithinState(controlQubits[control], stateSize)) {
      impl->error = "control qubit exceeds Metal state range.";
      return false;
    }
    controlMasks.push_back(std::uint64_t{1} << controlQubits[control]);
  }
  if (controlMasks.empty())
    controlMasks.push_back(0);
  releaseResidentState();

  std::vector<MetalComplexFloat> gpuState(stateSize);
  for (std::size_t index = 0; index < stateSize; ++index)
    gpuState[index] = MetalComplexFloat{static_cast<float>(state[index].real()),
                                        static_cast<float>(state[index].imag())};

  std::array<MetalComplexFloat, 4> gpuMatrix;
  for (std::size_t index = 0; index < gpuMatrix.size(); ++index)
    gpuMatrix[index] =
        MetalComplexFloat{static_cast<float>(matrix[index].real()),
                          static_cast<float>(matrix[index].imag())};

  SingleQubitParams params{targetMask, lowMask,
                           static_cast<std::uint32_t>(controlCount), 0};

  @autoreleasepool {
    id<MTLBuffer> stateBuffer =
        [impl->device newBufferWithBytes:gpuState.data()
                                  length:gpuState.size() *
                                         sizeof(MetalComplexFloat)
                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> matrixBuffer =
        [impl->device newBufferWithBytes:gpuMatrix.data()
                                  length:gpuMatrix.size() *
                                         sizeof(MetalComplexFloat)
                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> controlsBuffer =
        [impl->device newBufferWithBytes:controlMasks.data()
                                  length:controlMasks.size() *
                                         sizeof(std::uint64_t)
                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> paramsBuffer =
        [impl->device newBufferWithBytes:&params
                                  length:sizeof(SingleQubitParams)
                                 options:MTLResourceStorageModeShared];

    if (!stateBuffer || !matrixBuffer || !controlsBuffer || !paramsBuffer) {
      [stateBuffer release];
      [matrixBuffer release];
      [controlsBuffer release];
      [paramsBuffer release];
      impl->error = "failed to allocate Metal gate buffers.";
      return false;
    }

    id<MTLCommandBuffer> commandBuffer = [impl->commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    if (!commandBuffer || !encoder) {
      [stateBuffer release];
      [matrixBuffer release];
      [controlsBuffer release];
      [paramsBuffer release];
      impl->error = "failed to create Metal command encoder.";
      return false;
    }

    [encoder setComputePipelineState:impl->singleQubitPipeline];
    [encoder setBuffer:stateBuffer offset:0 atIndex:0];
    [encoder setBuffer:matrixBuffer offset:0 atIndex:1];
    [encoder setBuffer:controlsBuffer offset:0 atIndex:2];
    [encoder setBuffer:paramsBuffer offset:0 atIndex:3];

    const auto pairCount = stateSize >> 1;
    const auto pipelineWidth =
        [impl->singleQubitPipeline maxTotalThreadsPerThreadgroup];
    const auto threadsPerThreadgroup =
        std::max<NSUInteger>(1, std::min<NSUInteger>(pipelineWidth, pairCount));
    [encoder dispatchThreads:MTLSizeMake(pairCount, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(threadsPerThreadgroup, 1, 1)];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    const auto status = [commandBuffer status];
    if (status != MTLCommandBufferStatusCompleted) {
      impl->error =
          describeError([commandBuffer error], "Metal command buffer failed.");
      [stateBuffer release];
      [matrixBuffer release];
      [controlsBuffer release];
      [paramsBuffer release];
      return false;
    }

    auto *updatedState =
        reinterpret_cast<MetalComplexFloat *>([stateBuffer contents]);
    for (std::size_t index = 0; index < stateSize; ++index)
      state[index] = {static_cast<double>(updatedState[index].real),
                      static_cast<double>(updatedState[index].imag)};

    [stateBuffer release];
    [matrixBuffer release];
    [controlsBuffer release];
    [paramsBuffer release];
  }

  impl->error.clear();
  ++impl->singleQubitApplications;
  return true;
}

bool MetalStateVectorExecutor::applyTwoQubitGate(
    std::complex<double> *state, std::size_t stateSize,
    const std::complex<double> *matrix, const std::size_t *controlQubits,
    std::size_t controlCount, const std::size_t *targetQubits) {
  if (!impl || !impl->available())
    return false;
  if (!state || !matrix || !targetQubits || stateSize < 4 ||
      !isPowerOfTwo(stateSize)) {
    impl->error = "invalid Metal two-qubit gate input.";
    return false;
  }
  if (targetQubits[0] == targetQubits[1]) {
    impl->error = "duplicate Metal two-qubit gate target.";
    return false;
  }
  if (!qubitWithinState(targetQubits[0], stateSize) ||
      !qubitWithinState(targetQubits[1], stateSize)) {
    impl->error = "target qubit exceeds Metal state range.";
    return false;
  }
  if (!fitsKernelThreadIndex(stateSize >> 2)) {
    impl->error = "state size exceeds Metal two-qubit thread index range.";
    return false;
  }
  if (controlCount > 0 && !controlQubits) {
    impl->error = "missing Metal control qubit input.";
    return false;
  }
  if (hasDuplicateControlOrTargetOverlap(controlQubits, controlCount,
                                         targetQubits, 2)) {
    impl->error = "duplicate or overlapping Metal control qubit.";
    return false;
  }

  std::vector<std::uint64_t> controlMasks;
  controlMasks.reserve(std::max<std::size_t>(controlCount, 1));
  for (std::size_t control = 0; control < controlCount; ++control) {
    if (!qubitWithinState(controlQubits[control], stateSize)) {
      impl->error = "control qubit exceeds Metal state range.";
      return false;
    }
    controlMasks.push_back(std::uint64_t{1} << controlQubits[control]);
  }
  if (controlMasks.empty())
    controlMasks.push_back(0);
  releaseResidentState();

  std::vector<MetalComplexFloat> gpuState(stateSize);
  for (std::size_t index = 0; index < stateSize; ++index)
    gpuState[index] = MetalComplexFloat{static_cast<float>(state[index].real()),
                                        static_cast<float>(state[index].imag())};

  std::array<MetalComplexFloat, 16> gpuMatrix;
  for (std::size_t index = 0; index < gpuMatrix.size(); ++index)
    gpuMatrix[index] =
        MetalComplexFloat{static_cast<float>(matrix[index].real()),
                          static_cast<float>(matrix[index].imag())};

  const auto firstTarget =
      static_cast<std::uint32_t>(std::min(targetQubits[0], targetQubits[1]));
  const auto secondTarget =
      static_cast<std::uint32_t>(std::max(targetQubits[0], targetQubits[1]));
  TwoQubitParams params{std::uint64_t{1} << targetQubits[0],
                        std::uint64_t{1} << targetQubits[1], firstTarget,
                        secondTarget,
                        static_cast<std::uint32_t>(controlCount), 0};

  @autoreleasepool {
    id<MTLBuffer> stateBuffer =
        [impl->device newBufferWithBytes:gpuState.data()
                                  length:gpuState.size() *
                                         sizeof(MetalComplexFloat)
                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> matrixBuffer =
        [impl->device newBufferWithBytes:gpuMatrix.data()
                                  length:gpuMatrix.size() *
                                         sizeof(MetalComplexFloat)
                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> controlsBuffer =
        [impl->device newBufferWithBytes:controlMasks.data()
                                  length:controlMasks.size() *
                                         sizeof(std::uint64_t)
                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> paramsBuffer =
        [impl->device newBufferWithBytes:&params
                                  length:sizeof(TwoQubitParams)
                                 options:MTLResourceStorageModeShared];

    if (!stateBuffer || !matrixBuffer || !controlsBuffer || !paramsBuffer) {
      [stateBuffer release];
      [matrixBuffer release];
      [controlsBuffer release];
      [paramsBuffer release];
      impl->error = "failed to allocate Metal two-qubit gate buffers.";
      return false;
    }

    id<MTLCommandBuffer> commandBuffer = [impl->commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    if (!commandBuffer || !encoder) {
      [stateBuffer release];
      [matrixBuffer release];
      [controlsBuffer release];
      [paramsBuffer release];
      impl->error = "failed to create Metal two-qubit command encoder.";
      return false;
    }

    [encoder setComputePipelineState:impl->twoQubitPipeline];
    [encoder setBuffer:stateBuffer offset:0 atIndex:0];
    [encoder setBuffer:matrixBuffer offset:0 atIndex:1];
    [encoder setBuffer:controlsBuffer offset:0 atIndex:2];
    [encoder setBuffer:paramsBuffer offset:0 atIndex:3];

    const auto blockCount = stateSize >> 2;
    const auto pipelineWidth =
        [impl->twoQubitPipeline maxTotalThreadsPerThreadgroup];
    const auto threadsPerThreadgroup =
        std::max<NSUInteger>(1, std::min<NSUInteger>(pipelineWidth, blockCount));
    [encoder dispatchThreads:MTLSizeMake(blockCount, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(threadsPerThreadgroup, 1, 1)];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    const auto status = [commandBuffer status];
    if (status != MTLCommandBufferStatusCompleted) {
      impl->error =
          describeError([commandBuffer error], "Metal two-qubit command buffer failed.");
      [stateBuffer release];
      [matrixBuffer release];
      [controlsBuffer release];
      [paramsBuffer release];
      return false;
    }

    auto *updatedState =
        reinterpret_cast<MetalComplexFloat *>([stateBuffer contents]);
    for (std::size_t index = 0; index < stateSize; ++index)
      state[index] = {static_cast<double>(updatedState[index].real),
                      static_cast<double>(updatedState[index].imag)};

    [stateBuffer release];
    [matrixBuffer release];
    [controlsBuffer release];
    [paramsBuffer release];
  }

  impl->error.clear();
  ++impl->twoQubitApplications;
  return true;
}

bool MetalStateVectorExecutor::applyResidentTwoQubitGate(
    const std::complex<double> *matrix, const std::size_t *controlQubits,
    std::size_t controlCount, const std::size_t *targetQubits) {
  if (!impl || !impl->available())
    return false;
  if (!impl->residentStateBuffer || impl->residentStateSize < 4 ||
      !isPowerOfTwo(impl->residentStateSize) || !matrix || !targetQubits) {
    impl->error = "invalid Metal resident two-qubit gate input.";
    return false;
  }
  if (targetQubits[0] == targetQubits[1]) {
    impl->error = "duplicate Metal two-qubit gate target.";
    return false;
  }
  if (!qubitWithinState(targetQubits[0], impl->residentStateSize) ||
      !qubitWithinState(targetQubits[1], impl->residentStateSize)) {
    impl->error = "target qubit exceeds Metal state range.";
    return false;
  }
  if (!fitsKernelThreadIndex(impl->residentStateSize >> 2)) {
    impl->error = "state size exceeds Metal two-qubit thread index range.";
    return false;
  }
  if (controlCount > 0 && !controlQubits) {
    impl->error = "missing Metal control qubit input.";
    return false;
  }
  if (hasDuplicateControlOrTargetOverlap(controlQubits, controlCount,
                                         targetQubits, 2)) {
    impl->error = "duplicate or overlapping Metal control qubit.";
    return false;
  }

  std::vector<std::uint64_t> controlMasks;
  controlMasks.reserve(std::max<std::size_t>(controlCount, 1));
  for (std::size_t control = 0; control < controlCount; ++control) {
    if (!qubitWithinState(controlQubits[control], impl->residentStateSize)) {
      impl->error = "control qubit exceeds Metal state range.";
      return false;
    }
    controlMasks.push_back(std::uint64_t{1} << controlQubits[control]);
  }
  if (controlMasks.empty())
    controlMasks.push_back(0);

  std::array<MetalComplexFloat, 16> gpuMatrix;
  for (std::size_t index = 0; index < gpuMatrix.size(); ++index)
    gpuMatrix[index] =
        MetalComplexFloat{static_cast<float>(matrix[index].real()),
                          static_cast<float>(matrix[index].imag())};

  const auto firstTarget =
      static_cast<std::uint32_t>(std::min(targetQubits[0], targetQubits[1]));
  const auto secondTarget =
      static_cast<std::uint32_t>(std::max(targetQubits[0], targetQubits[1]));
  TwoQubitParams params{std::uint64_t{1} << targetQubits[0],
                        std::uint64_t{1} << targetQubits[1], firstTarget,
                        secondTarget,
                        static_cast<std::uint32_t>(controlCount), 0};

  @autoreleasepool {
    id<MTLBuffer> matrixBuffer =
        [impl->device newBufferWithBytes:gpuMatrix.data()
                                  length:gpuMatrix.size() *
                                         sizeof(MetalComplexFloat)
                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> controlsBuffer =
        [impl->device newBufferWithBytes:controlMasks.data()
                                  length:controlMasks.size() *
                                         sizeof(std::uint64_t)
                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> paramsBuffer =
        [impl->device newBufferWithBytes:&params
                                  length:sizeof(TwoQubitParams)
                                 options:MTLResourceStorageModeShared];

    if (!matrixBuffer || !controlsBuffer || !paramsBuffer) {
      [matrixBuffer release];
      [controlsBuffer release];
      [paramsBuffer release];
      impl->error = "failed to allocate Metal resident two-qubit buffers.";
      return false;
    }

    id<MTLCommandBuffer> commandBuffer = [impl->commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    if (!commandBuffer || !encoder) {
      [matrixBuffer release];
      [controlsBuffer release];
      [paramsBuffer release];
      impl->error = "failed to create Metal resident two-qubit command encoder.";
      return false;
    }

    [encoder setComputePipelineState:impl->twoQubitPipeline];
    [encoder setBuffer:impl->residentStateBuffer offset:0 atIndex:0];
    [encoder setBuffer:matrixBuffer offset:0 atIndex:1];
    [encoder setBuffer:controlsBuffer offset:0 atIndex:2];
    [encoder setBuffer:paramsBuffer offset:0 atIndex:3];

    const auto blockCount = impl->residentStateSize >> 2;
    const auto pipelineWidth =
        [impl->twoQubitPipeline maxTotalThreadsPerThreadgroup];
    const auto threadsPerThreadgroup =
        std::max<NSUInteger>(1, std::min<NSUInteger>(pipelineWidth, blockCount));
    [encoder dispatchThreads:MTLSizeMake(blockCount, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(threadsPerThreadgroup, 1, 1)];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    const auto status = [commandBuffer status];
    if (status != MTLCommandBufferStatusCompleted) {
      impl->error =
          describeError([commandBuffer error],
                        "Metal resident two-qubit command buffer failed.");
      [matrixBuffer release];
      [controlsBuffer release];
      [paramsBuffer release];
      return false;
    }

    [matrixBuffer release];
    [controlsBuffer release];
    [paramsBuffer release];
  }

  impl->error.clear();
  ++impl->twoQubitApplications;
  return true;
}

bool MetalStateVectorExecutor::fillFullRegisterProbabilities(
    const std::complex<double> *state, std::size_t stateSize,
    double *probabilities, std::size_t probabilityCount) {
  if (!impl || !impl->available())
    return false;
  if (!state || !probabilities || stateSize == 0 ||
      probabilityCount != stateSize || !isPowerOfTwo(stateSize)) {
    impl->error = "invalid Metal probability-fill input.";
    return false;
  }
  if (stateSize > std::numeric_limits<std::uint32_t>::max()) {
    impl->error = "state size exceeds Metal probability-fill index range.";
    return false;
  }

  std::vector<MetalComplexFloat> gpuState(stateSize);
  for (std::size_t index = 0; index < stateSize; ++index)
    gpuState[index] = MetalComplexFloat{static_cast<float>(state[index].real()),
                                        static_cast<float>(state[index].imag())};
  std::vector<float> gpuProbabilities(stateSize, 0.0f);

  @autoreleasepool {
    id<MTLBuffer> stateBuffer =
        [impl->device newBufferWithBytes:gpuState.data()
                                  length:gpuState.size() *
                                         sizeof(MetalComplexFloat)
                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> probabilitiesBuffer =
        [impl->device newBufferWithBytes:gpuProbabilities.data()
                                  length:gpuProbabilities.size() *
                                         sizeof(float)
                                 options:MTLResourceStorageModeShared];

    if (!stateBuffer || !probabilitiesBuffer) {
      [stateBuffer release];
      [probabilitiesBuffer release];
      impl->error = "failed to allocate Metal probability-fill buffers.";
      return false;
    }

    id<MTLCommandBuffer> commandBuffer = [impl->commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    if (!commandBuffer || !encoder) {
      [stateBuffer release];
      [probabilitiesBuffer release];
      impl->error = "failed to create Metal probability-fill command encoder.";
      return false;
    }

    [encoder setComputePipelineState:impl->probabilityPipeline];
    [encoder setBuffer:stateBuffer offset:0 atIndex:0];
    [encoder setBuffer:probabilitiesBuffer offset:0 atIndex:1];

    const auto pipelineWidth =
        [impl->probabilityPipeline maxTotalThreadsPerThreadgroup];
    const auto threadsPerThreadgroup =
        std::max<NSUInteger>(1, std::min<NSUInteger>(pipelineWidth, stateSize));
    [encoder dispatchThreads:MTLSizeMake(stateSize, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(threadsPerThreadgroup, 1, 1)];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    const auto status = [commandBuffer status];
    if (status != MTLCommandBufferStatusCompleted) {
      impl->error = describeError([commandBuffer error],
                                  "Metal probability-fill command buffer failed.");
      [stateBuffer release];
      [probabilitiesBuffer release];
      return false;
    }

    auto *updatedProbabilities =
        reinterpret_cast<float *>([probabilitiesBuffer contents]);
    for (std::size_t index = 0; index < stateSize; ++index)
      probabilities[index] = static_cast<double>(updatedProbabilities[index]);

    [stateBuffer release];
    [probabilitiesBuffer release];
  }

  impl->error.clear();
  ++impl->probabilityFillApplications;
  return true;
}

bool MetalStateVectorExecutor::fillResidentFullRegisterProbabilities(
    double *probabilities, std::size_t probabilityCount) {
  if (!impl || !impl->available())
    return false;
  if (!probabilities || !impl->residentStateBuffer ||
      probabilityCount != impl->residentStateSize ||
      !isPowerOfTwo(impl->residentStateSize)) {
    impl->error = "invalid Metal resident probability-fill input.";
    return false;
  }
  if (impl->residentStateSize > std::numeric_limits<std::uint32_t>::max()) {
    impl->error = "state size exceeds Metal probability-fill index range.";
    return false;
  }

  std::vector<float> gpuProbabilities(impl->residentStateSize, 0.0f);

  @autoreleasepool {
    id<MTLBuffer> probabilitiesBuffer =
        [impl->device newBufferWithBytes:gpuProbabilities.data()
                                  length:gpuProbabilities.size() *
                                         sizeof(float)
                                 options:MTLResourceStorageModeShared];

    if (!probabilitiesBuffer) {
      impl->error =
          "failed to allocate Metal resident probability-fill buffer.";
      return false;
    }

    id<MTLCommandBuffer> commandBuffer = [impl->commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    if (!commandBuffer || !encoder) {
      [probabilitiesBuffer release];
      impl->error =
          "failed to create Metal resident probability-fill command encoder.";
      return false;
    }

    [encoder setComputePipelineState:impl->probabilityPipeline];
    [encoder setBuffer:impl->residentStateBuffer offset:0 atIndex:0];
    [encoder setBuffer:probabilitiesBuffer offset:0 atIndex:1];

    const auto pipelineWidth =
        [impl->probabilityPipeline maxTotalThreadsPerThreadgroup];
    const auto threadsPerThreadgroup = std::max<NSUInteger>(
        1, std::min<NSUInteger>(pipelineWidth, impl->residentStateSize));
    [encoder dispatchThreads:MTLSizeMake(impl->residentStateSize, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(threadsPerThreadgroup, 1, 1)];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    const auto status = [commandBuffer status];
    if (status != MTLCommandBufferStatusCompleted) {
      impl->error = describeError(
          [commandBuffer error],
          "Metal resident probability-fill command buffer failed.");
      [probabilitiesBuffer release];
      return false;
    }

    auto *updatedProbabilities =
        reinterpret_cast<float *>([probabilitiesBuffer contents]);
    for (std::size_t index = 0; index < impl->residentStateSize; ++index)
      probabilities[index] = static_cast<double>(updatedProbabilities[index]);

    [probabilitiesBuffer release];
  }

  impl->error.clear();
  ++impl->probabilityFillApplications;
  return true;
}

bool MetalStateVectorExecutor::fillResidentMarginalProbabilities(
    const std::size_t *qubits, std::size_t qubitCount, double *probabilities,
    std::size_t probabilityCount) {
  if (!impl || !impl->available())
    return false;
  if ((!qubits && qubitCount != 0) || !probabilities ||
      !impl->residentStateBuffer || !isPowerOfTwo(impl->residentStateSize)) {
    if (impl)
      impl->error = "invalid Metal resident marginal probability input.";
    return false;
  }
  if (qubitCount >= std::numeric_limits<std::uint32_t>::digits) {
    impl->error =
        "Metal resident marginal probability qubit count exceeds output range.";
    return false;
  }
  const auto expectedProbabilityCount = std::size_t{1} << qubitCount;
  if (probabilityCount != expectedProbabilityCount) {
    impl->error =
        "Metal resident marginal probability buffer has incorrect size.";
    return false;
  }
  if (!fitsKernelThreadIndex(impl->residentStateSize)) {
    impl->error =
        "state size exceeds Metal marginal probability thread index range.";
    return false;
  }

  std::vector<std::uint64_t> qubitMasks(std::max<std::size_t>(qubitCount, 1),
                                        0);
  for (std::size_t i = 0; i < qubitCount; ++i) {
    if (!qubitWithinState(qubits[i], impl->residentStateSize)) {
      impl->error = "marginal probability qubit exceeds Metal state range.";
      return false;
    }
    for (std::size_t previous = 0; previous < i; ++previous) {
      if (qubits[i] == qubits[previous]) {
        impl->error = "duplicate marginal probability qubit.";
        return false;
      }
    }
    qubitMasks[i] = std::uint64_t{1} << qubits[i];
  }

  if ([impl->marginalProbabilityPipeline maxTotalThreadsPerThreadgroup] <
      marginalProbabilityThreadsPerThreadgroup) {
    impl->error =
        "Metal marginal probability pipeline does not support 256-thread "
        "reduction groups.";
    return false;
  }

  const auto groupCount =
      (impl->residentStateSize + marginalProbabilityThreadsPerThreadgroup - 1) /
      marginalProbabilityThreadsPerThreadgroup;
  if (groupCount >
      std::numeric_limits<std::uint32_t>::max()) {
    impl->error = "Metal marginal probability group count exceeds range.";
    return false;
  }
  if (probabilityCount >
      std::numeric_limits<std::size_t>::max() / groupCount) {
    impl->error = "Metal marginal probability partial-sum size overflow.";
    return false;
  }
  const auto partialSumCount = probabilityCount * groupCount;
  if (partialSumCount >
      std::numeric_limits<std::size_t>::max() / sizeof(float)) {
    impl->error = "Metal marginal probability partial-sum byte size overflow.";
    return false;
  }

  const MarginalProbabilityParams params{
      static_cast<std::uint32_t>(impl->residentStateSize),
      static_cast<std::uint32_t>(qubitCount),
      static_cast<std::uint32_t>(groupCount), 0};

  @autoreleasepool {
    id<MTLBuffer> masksBuffer =
        [impl->device newBufferWithBytes:qubitMasks.data()
                                  length:qubitMasks.size() *
                                         sizeof(std::uint64_t)
                                 options:MTLResourceStorageModeShared];
    if (!masksBuffer) {
      impl->error = "failed to allocate Metal marginal qubit-mask buffer.";
      return false;
    }

    id<MTLBuffer> partialSumsBuffer =
        [impl->device newBufferWithLength:partialSumCount * sizeof(float)
                                  options:MTLResourceStorageModeShared];
    if (!partialSumsBuffer) {
      [masksBuffer release];
      impl->error =
          "failed to allocate Metal marginal probability partial-sums buffer.";
      return false;
    }

    id<MTLBuffer> paramsBuffer =
        [impl->device newBufferWithBytes:&params
                                  length:sizeof(MarginalProbabilityParams)
                                 options:MTLResourceStorageModeShared];
    if (!paramsBuffer) {
      [partialSumsBuffer release];
      [masksBuffer release];
      impl->error =
          "failed to allocate Metal marginal probability parameters buffer.";
      return false;
    }

    id<MTLCommandBuffer> commandBuffer = [impl->commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    if (!commandBuffer || !encoder) {
      [paramsBuffer release];
      [partialSumsBuffer release];
      [masksBuffer release];
      impl->error =
          "failed to create Metal marginal probability command encoder.";
      return false;
    }

    [encoder setComputePipelineState:impl->marginalProbabilityPipeline];
    [encoder setBuffer:impl->residentStateBuffer offset:0 atIndex:0];
    [encoder setBuffer:masksBuffer offset:0 atIndex:1];
    [encoder setBuffer:partialSumsBuffer offset:0 atIndex:2];
    [encoder setBuffer:paramsBuffer offset:0 atIndex:3];
    [encoder setThreadgroupMemoryLength:marginalProbabilityThreadsPerThreadgroup *
                                        sizeof(float)
                                atIndex:0];
    [encoder dispatchThreadgroups:MTLSizeMake(groupCount, probabilityCount, 1)
            threadsPerThreadgroup:MTLSizeMake(
                                      marginalProbabilityThreadsPerThreadgroup,
                                      1, 1)];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    const auto status = [commandBuffer status];
    if (status != MTLCommandBufferStatusCompleted) {
      impl->error = describeError(
          [commandBuffer error],
          "Metal marginal probability command buffer failed.");
      [paramsBuffer release];
      [partialSumsBuffer release];
      [masksBuffer release];
      return false;
    }

    const auto *partialSums =
        reinterpret_cast<const float *>([partialSumsBuffer contents]);
    for (std::size_t outcome = 0; outcome < probabilityCount; ++outcome) {
      double sum = 0.0;
      const auto offset = outcome * groupCount;
      for (std::size_t group = 0; group < groupCount; ++group)
        sum += static_cast<double>(partialSums[offset + group]);
      probabilities[outcome] = sum;
    }

    [paramsBuffer release];
    [partialSumsBuffer release];
    [masksBuffer release];
  }

  impl->error.clear();
  ++impl->marginalProbabilityApplications;
  return true;
}

bool MetalStateVectorExecutor::computeResidentQubitProbability(
    std::size_t qubit, double *probabilityOne) {
  if (!impl || !impl->available())
    return false;
  if (!probabilityOne || !impl->residentStateBuffer ||
      !isPowerOfTwo(impl->residentStateSize)) {
    if (impl)
      impl->error = "invalid Metal resident measurement probability input.";
    return false;
  }
  if (!qubitWithinState(qubit, impl->residentStateSize)) {
    impl->error = "measurement qubit exceeds Metal state range.";
    return false;
  }
  if (!fitsKernelThreadIndex(impl->residentStateSize)) {
    impl->error =
        "state size exceeds Metal measurement probability thread index range.";
    return false;
  }

  constexpr NSUInteger measurementThreadsPerThreadgroup = 256;
  if ([impl->measurementProbabilityPipeline maxTotalThreadsPerThreadgroup] <
      measurementThreadsPerThreadgroup) {
    impl->error =
        "Metal measurement probability pipeline does not support 256-thread "
        "reduction groups.";
    return false;
  }

  const auto groupCount =
      (impl->residentStateSize + measurementThreadsPerThreadgroup - 1) /
      measurementThreadsPerThreadgroup;
  const MeasurementProbabilityParams params{
      std::uint64_t{1} << qubit,
      static_cast<std::uint32_t>(impl->residentStateSize), 0};

  @autoreleasepool {
    id<MTLBuffer> partialSumsBuffer =
        [impl->device newBufferWithLength:groupCount * sizeof(float)
                                  options:MTLResourceStorageModeShared];
    if (!partialSumsBuffer) {
      impl->error =
          "failed to allocate Metal resident measurement partial-sums buffer.";
      return false;
    }

    id<MTLBuffer> paramsBuffer =
        [impl->device newBufferWithBytes:&params
                                  length:sizeof(MeasurementProbabilityParams)
                                 options:MTLResourceStorageModeShared];
    if (!paramsBuffer) {
      [partialSumsBuffer release];
      impl->error =
          "failed to allocate Metal resident measurement parameters buffer.";
      return false;
    }

    id<MTLCommandBuffer> commandBuffer = [impl->commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    if (!commandBuffer || !encoder) {
      [paramsBuffer release];
      [partialSumsBuffer release];
      impl->error =
          "failed to create Metal resident measurement probability encoder.";
      return false;
    }

    [encoder setComputePipelineState:impl->measurementProbabilityPipeline];
    [encoder setBuffer:impl->residentStateBuffer offset:0 atIndex:0];
    [encoder setBuffer:partialSumsBuffer offset:0 atIndex:1];
    [encoder setBuffer:paramsBuffer offset:0 atIndex:2];
    [encoder setThreadgroupMemoryLength:measurementThreadsPerThreadgroup *
                                        sizeof(float)
                                atIndex:0];
    [encoder dispatchThreadgroups:MTLSizeMake(groupCount, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(measurementThreadsPerThreadgroup,
                                              1, 1)];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    const auto status = [commandBuffer status];
    if (status != MTLCommandBufferStatusCompleted) {
      impl->error = describeError(
          [commandBuffer error],
          "Metal resident measurement probability command buffer failed.");
      [paramsBuffer release];
      [partialSumsBuffer release];
      return false;
    }

    double sum = 0.0;
    const auto *partialSums =
        reinterpret_cast<const float *>([partialSumsBuffer contents]);
    for (std::size_t group = 0; group < groupCount; ++group)
      sum += static_cast<double>(partialSums[group]);
    *probabilityOne = sum;

    [paramsBuffer release];
    [partialSumsBuffer release];
  }

  impl->error.clear();
  ++impl->measurementProbabilityApplications;
  ++impl->measurementProbabilityReductionApplications;
  return true;
}

bool MetalStateVectorExecutor::collapseResidentQubit(
    std::size_t qubit, bool result, double branchProbability) {
  if (!impl || !impl->available())
    return false;
  if (!impl->residentStateBuffer || !isPowerOfTwo(impl->residentStateSize) ||
      branchProbability <= 0.0) {
    impl->error = "invalid Metal resident collapse input.";
    return false;
  }
  if (!qubitWithinState(qubit, impl->residentStateSize)) {
    impl->error = "collapse qubit exceeds Metal state range.";
    return false;
  }
  if (!fitsKernelThreadIndex(impl->residentStateSize)) {
    impl->error = "state size exceeds Metal collapse thread index range.";
    return false;
  }

  const auto inverseNorm =
      static_cast<float>(1.0 / std::sqrt(branchProbability));
  CollapseParams params{std::uint64_t{1} << qubit, inverseNorm,
                        result ? std::uint32_t{1} : std::uint32_t{0}};

  @autoreleasepool {
    id<MTLBuffer> paramsBuffer =
        [impl->device newBufferWithBytes:&params
                                  length:sizeof(CollapseParams)
                                 options:MTLResourceStorageModeShared];
    if (!paramsBuffer) {
      impl->error = "failed to allocate Metal resident collapse params.";
      return false;
    }

    id<MTLCommandBuffer> commandBuffer = [impl->commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    if (!commandBuffer || !encoder) {
      [paramsBuffer release];
      impl->error = "failed to create Metal resident collapse encoder.";
      return false;
    }

    [encoder setComputePipelineState:impl->collapsePipeline];
    [encoder setBuffer:impl->residentStateBuffer offset:0 atIndex:0];
    [encoder setBuffer:paramsBuffer offset:0 atIndex:1];

    const auto pipelineWidth =
        [impl->collapsePipeline maxTotalThreadsPerThreadgroup];
    const auto threadsPerThreadgroup = std::max<NSUInteger>(
        1, std::min<NSUInteger>(pipelineWidth, impl->residentStateSize));
    [encoder dispatchThreads:MTLSizeMake(impl->residentStateSize, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(threadsPerThreadgroup, 1, 1)];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    const auto status = [commandBuffer status];
    if (status != MTLCommandBufferStatusCompleted) {
      impl->error = describeError(
          [commandBuffer error],
          "Metal resident collapse command buffer failed.");
      [paramsBuffer release];
      return false;
    }

    [paramsBuffer release];
  }

  impl->error.clear();
  ++impl->measurementCollapseApplications;
  return true;
}

std::size_t MetalStateVectorExecutor::singleQubitGateApplications() const {
  return impl ? impl->singleQubitApplications : 0;
}

std::size_t MetalStateVectorExecutor::twoQubitGateApplications() const {
  return impl ? impl->twoQubitApplications : 0;
}

std::size_t MetalStateVectorExecutor::probabilityFillApplications() const {
  return impl ? impl->probabilityFillApplications : 0;
}

std::size_t MetalStateVectorExecutor::marginalProbabilityApplications() const {
  return impl ? impl->marginalProbabilityApplications : 0;
}

std::size_t MetalStateVectorExecutor::measurementProbabilityApplications()
    const {
  return impl ? impl->measurementProbabilityApplications : 0;
}

std::size_t
MetalStateVectorExecutor::measurementProbabilityReductionApplications() const {
  return impl ? impl->measurementProbabilityReductionApplications : 0;
}

std::size_t MetalStateVectorExecutor::measurementCollapseApplications() const {
  return impl ? impl->measurementCollapseApplications : 0;
}

std::size_t MetalStateVectorExecutor::residentStateUploads() const {
  return impl ? impl->residentUploads : 0;
}

std::size_t MetalStateVectorExecutor::residentStateDownloads() const {
  return impl ? impl->residentDownloads : 0;
}

} // namespace nvqir::mklq
