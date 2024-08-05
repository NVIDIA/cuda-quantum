/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma nv_diag_suppress = unsigned_compare_with_zero
#pragma nv_diag_suppress = unrecognized_gcc_pragma

#include "cuComplex.h"
#include "device_launch_parameters.h"
#include "CuStateVecCircuitSimulator.h"
#include <thrust/complex.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/inner_product.h>

namespace nvqir {

// kronprod functions adapted from
// https://github.com/DmitryLyakh/TAL_SH/blob/3cefc2133a68b67c515f4b68a0ed9e3c66e4b4b2/tensor_algebra_gpu_nvidia.cu#L745

#define THRDS_ARRAY_PRODUCT 256

#pragma push
#pragma nv_diag_suppress 177
__device__ __host__ cuDoubleComplex operator*(cuDoubleComplex a,
                                              cuDoubleComplex b) {
  return cuCmul(a, b);
}
__device__ __host__ cuDoubleComplex operator+(cuDoubleComplex a,
                                              cuDoubleComplex b) {
  return cuCadd(a, b);
}
__device__ __host__ cuFloatComplex operator*(cuFloatComplex a,
                                             cuFloatComplex b) {
  return cuCmulf(a, b);
}
__device__ __host__ cuFloatComplex operator+(cuFloatComplex a,
                                             cuFloatComplex b) {
  return cuCaddf(a, b);
}

template <typename CudaDataType>
__global__ void cudaKronprod(size_t tsize1, const CudaDataType *arr1, 
                             size_t tsize2, const CudaDataType *arr2, 
                             CudaDataType *arr0) {
  __shared__ CudaDataType lbuf[THRDS_ARRAY_PRODUCT + 1], rbuf[THRDS_ARRAY_PRODUCT];
  size_t _ib, _in, _jb, _jn, _tx, _jc, _ja;

  _tx = (size_t)threadIdx.x;
  for (_jb = blockIdx.y * THRDS_ARRAY_PRODUCT; _jb < tsize2;
       _jb += gridDim.y * THRDS_ARRAY_PRODUCT) {
    if (_jb + THRDS_ARRAY_PRODUCT > tsize2) {
      _jn = tsize2 - _jb;
    } else {
      _jn = THRDS_ARRAY_PRODUCT;
    }

    if (_tx < _jn)
      rbuf[_tx] = arr2[_jb + _tx];

    for (_ib = blockIdx.x * THRDS_ARRAY_PRODUCT; _ib < tsize1;
         _ib += gridDim.x * THRDS_ARRAY_PRODUCT) {
      if (_ib + THRDS_ARRAY_PRODUCT > tsize1) {
        _in = tsize1 - _ib;
      } else {
        _in = THRDS_ARRAY_PRODUCT;
      }

      if (_tx < _in)
        lbuf[_tx] = arr1[_ib + _tx];

      __syncthreads();
      for (_jc = 0; _jc < _jn; _jc++) {
        if (_tx < _in) {
          _ja = (_jb + _jc) * tsize1 + (_ib + _tx);
          arr0[_ja] = arr0[_ja] + lbuf[_tx] * rbuf[_jc];
        }
      }
      __syncthreads();
    }
  }
  return;
}
#pragma pop

template <typename CudaDataType>
void kronprod(uint32_t n_blocks, int32_t threads_per_block,
              size_t tsize1, const void *arr1,
              size_t tsize2, const void *arr2, 
              void *arr0) {
  cudaKronprod<<<n_blocks, threads_per_block>>>(
    tsize1, reinterpret_cast<const CudaDataType *>(arr1), 
    (1UL << tsize2), reinterpret_cast<const CudaDataType *>(arr2),
    reinterpret_cast<CudaDataType *>(arr0));
}

template void
kronprod<cuFloatComplex>(uint32_t n_blocks, int32_t threads_per_block,
                         size_t tsize1, const void *arr1, 
                         size_t tsize2, const void *arr2, 
                         void *arr0);

template void
kronprod<cuDoubleComplex>(uint32_t n_blocks, int32_t threads_per_block,
                          size_t tsize1, const void *arr1, 
                          size_t tsize2, const void *arr2, 
                          void *arr0);

/// @brief Kernel to set the first N elements of the state vector sv equal to
/// the elements provided by the vector sv2. N is the number of elements to set.
/// Size of sv must be greater than size of sv2.
template <typename CudaDataType>
__global__ void cudaSetFirstNElements(CudaDataType *sv, const CudaDataType *__restrict__ sv2, int64_t N) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < N) {
    sv[i].x = sv2[i].x;
    sv[i].y = sv2[i].y;
  } else {
    sv[i].x = 0.0;
    sv[i].y = 0.0;
  }
}

template <typename CudaDataType>
void setFirstNElements(uint32_t n_blocks, 
                       int32_t threads_per_block, 
                       void *newDeviceStateVector, 
                       void *deviceStateVector,
                       std::size_t previousStateDimension) {
  cudaSetFirstNElements<<<n_blocks, threads_per_block>>>(
    reinterpret_cast<CudaDataType *>(newDeviceStateVector),
    reinterpret_cast<CudaDataType *>(deviceStateVector),
    previousStateDimension);
}

template void
setFirstNElements<cuFloatComplex>(uint32_t n_blocks, 
                       int32_t threads_per_block, 
                       void *newDeviceStateVector, 
                       void *deviceStateVector,
                       std::size_t previousStateDimension);

template void
setFirstNElements<cuDoubleComplex>(uint32_t n_blocks, 
                       int32_t threads_per_block, 
                       void *newDeviceStateVector, 
                       void *deviceStateVector,
                       std::size_t previousStateDimension);

/// @brief Initialize the device state vector to the |0...0> state
template <typename CudaDataType>
__global__ void cudaInitializeDeviceStateVector(CudaDataType *sv, int64_t dim) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i == 0) {
    sv[i].x = 1.0;
    sv[i].y = 0.0;
  } else if (i < dim) {
    sv[i].x = 0.0;
    sv[i].y = 0.0;
  }
}

template <typename CudaDataType>
void initializeDeviceStateVector(uint32_t n_blocks, 
                                 int32_t threads_per_block, 
                                 void *deviceStateVector,
                                 std::size_t stateDimension) {
  cudaInitializeDeviceStateVector<<<n_blocks, threads_per_block>>>(
    reinterpret_cast<CudaDataType *>(deviceStateVector), stateDimension);
}

template void
initializeDeviceStateVector<cuFloatComplex>(uint32_t n_blocks, 
                                 int32_t threads_per_block, 
                                 void *deviceStateVector,
                                 std::size_t stateDimension);

template void
initializeDeviceStateVector<cuDoubleComplex>(uint32_t n_blocks, 
                                 int32_t threads_per_block, 
                                 void *deviceStateVector,
                                 std::size_t stateDimension);

/// @brief Custom functor for the thrust inner product.
template <typename ScalarType>
struct AdotConjB
    : public thrust::binary_function<thrust::complex<ScalarType>, 
                                     thrust::complex<ScalarType>,
                                     thrust::complex<ScalarType>> {
  __host__ __device__ thrust::complex<ScalarType>
  operator()(thrust::complex<ScalarType> a, thrust::complex<ScalarType> b) {
    return a * thrust::conj(b);
  };
};

template struct complexValue<double>;
template struct complexValue<float>;

template <typename ScalarType>
complexValue<ScalarType> innerProduct(
  void *devicePtr, void *otherPtr, std::size_t size, bool createDeviceAlloc) {

  auto *castedDevicePtr =
      reinterpret_cast<thrust::complex<ScalarType> *>(devicePtr);
  thrust::device_ptr<thrust::complex<ScalarType>> thrustDevPtrABegin(
      castedDevicePtr);
  thrust::device_ptr<thrust::complex<ScalarType>> thrustDevPtrAEnd(
      castedDevicePtr + size);

  thrust::device_ptr<thrust::complex<ScalarType>> thrustDevPtrBBegin;
  if (createDeviceAlloc) {
    // otherPtr is not a device pointer...
    // FIXME: WE NEED TO PROPERLY CONVERT HERE - 
    // PASS A BUFFER RATHER THAN REINTERPRETE_CAST AND HOPE FOR THE BEST...
    auto *castedOtherPtr = reinterpret_cast<std::complex<ScalarType> *>(otherPtr);
    std::vector<std::complex<ScalarType>> dataAsVec(castedOtherPtr,
                                                    castedOtherPtr + size);
    thrust::device_vector<thrust::complex<ScalarType>> otherDevPtr(dataAsVec);
    thrustDevPtrBBegin = otherDevPtr.data();
  } else {
    // other is a device pointer
    auto *castedOtherPtr = reinterpret_cast<thrust::complex<ScalarType> *>(otherPtr);
    thrustDevPtrBBegin = thrust::device_ptr<thrust::complex<ScalarType>>(castedOtherPtr);
  }

  thrust::complex<ScalarType> result = thrust::inner_product(
    thrustDevPtrABegin, thrustDevPtrAEnd, thrustDevPtrBBegin,
    thrust::complex<ScalarType>(0.0),
    thrust::plus<thrust::complex<ScalarType>>(),
    AdotConjB<ScalarType>());

  complexValue<ScalarType> complex;
  complex.real = result.real();
  complex.imaginary = result.imag();
  return complex;
}


template complexValue<double> 
innerProduct(void *devicePtr, void *otherPtr, std::size_t size, bool createDeviceAlloc);

template complexValue<float> 
innerProduct(void *devicePtr, void *otherPtr, std::size_t size, bool createDeviceAlloc);

}
