/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/*! \file mpi_comm_impl.cpp
    \brief Implementation of CUDAQ MPI interface wrapper based on mpi4py

    If a natively-built MPI wrapper plugin is not available, CUDA Quantum also
   provides a plugin implementation based on Python's mpi4py lib via embedded
   interpreter. Rationale: this plugin targets CUDA Quantum wheel distribution:
   we ship pre-built wheel binary and leverage mpi4py build-from-source
   distribution mechanism to get MPI support for Python users.
*/

#include "distributed_capi.h"
#include <complex>
#include <iostream>
#include <mutex>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <stdexcept>

namespace py = pybind11;

namespace {

/// @brief Reference to the pybind11 scoped interpreter
thread_local static std::unique_ptr<py::scoped_interpreter> interp;

/// @brief Did we find mpi4py during library load?
static bool mpi4pyFound = false;

/// @brief True if this plugin did the MPI_Init call
bool initCalledByThis = false;

/// @brief Convert supported data type enum to the corresponding mpi4py's MPI
/// type
py::object convertType(DataType dataType) {
  auto mpiMod = py::module::import("mpi4py.MPI");
  switch (dataType) {
  case INT_8:
    return mpiMod.attr("INT8_T");
  case INT_16:
    return mpiMod.attr("INT16_T");
  case INT_32:
    return mpiMod.attr("INT32_T");
  case INT_64:
    return mpiMod.attr("INT64_T");
  case FLOAT_32:
    return mpiMod.attr("FLOAT");
  case FLOAT_64:
    return mpiMod.attr("DOUBLE");
  case FLOAT_COMPLEX:
    return mpiMod.attr("C_FLOAT_COMPLEX");
  case DOUBLE_COMPLEX:
    return mpiMod.attr("C_DOUBLE_COMPLEX");
  }
  __builtin_unreachable();
}

/// @brief Convert supported data type enum to the corresponding mpi4py's MPI
/// type
py::object convertTypeMinLoc(DataType dataType) {
  auto mpiMod = py::module::import("mpi4py.MPI");
  switch (dataType) {
  case FLOAT_32:
    return mpiMod.attr("FLOAT_INT");
  case FLOAT_64:
    return mpiMod.attr("DOUBLE_INT");
  default:
    throw std::runtime_error("Unsupported MINLOC data type");
  }
  __builtin_unreachable();
}

/// @brief Get size (in bytes) for a data type
std::size_t getDataSize(DataType dataType) {
  switch (dataType) {
  case INT_8:
    return sizeof(int8_t);
  case INT_16:
    return sizeof(int16_t);
  case INT_32:
    return sizeof(int32_t);
  case INT_64:
    return sizeof(int64_t);
  case FLOAT_32:
    return sizeof(float);
  case FLOAT_64:
    return sizeof(double);
  case FLOAT_COMPLEX:
    return sizeof(std::complex<float>);
  case DOUBLE_COMPLEX:
    return sizeof(std::complex<double>);
  }
  __builtin_unreachable();
}

/// @brief Convert supported op type enum to the corresponding mpi4py's MPI type
py::object convertType(ReduceOp opType) {
  auto mpiMod = py::module::import("mpi4py.MPI");
  switch (opType) {
  case SUM:
    return mpiMod.attr("SUM");
  case PROD:
    return mpiMod.attr("PROD");
  case MIN:
    return mpiMod.attr("MIN");
  case MIN_LOC:
    return mpiMod.attr("MINLOC");
  }
  __builtin_unreachable();
}

/// @brief Unpack the type-erased communicator object into a mpi4py's Comm
/// object
py::object unpackMpiCommunicator(const cudaqDistributedCommunicator_t *comm) {
  try {
    auto mpiMod = py::module::import("mpi4py.MPI");
    auto mpi_obj = mpiMod.attr("Intracomm")();
    auto address = mpiMod.attr("_addressof")(mpi_obj).cast<int64_t>();
    void **pointer = reinterpret_cast<void **>(address);
    pointer[0] = *reinterpret_cast<void **>(comm->commPtr);
    return mpi_obj;
  } catch (std::exception &e) {
    std::cerr << "[mpi4py] Caught exception \"" << e.what() << "\"\n";
    throw std::runtime_error(
        "Invalid distributed communicator encountered in CUDAQ mpi4py plugin.");
  }
}

/// @brief Tracking in-flight non-blocking send and receive requests.
struct PendingRequest {
  py::object requests[2];
  int nActiveRequests;
  PendingRequest() : nActiveRequests(0){};
  static std::mutex g_mutex;
  static std::unordered_map<const cudaqDistributedCommunicator_t *,
                            PendingRequest>
      g_requests;
};
std::mutex PendingRequest::g_mutex;
std::unordered_map<const cudaqDistributedCommunicator_t *, PendingRequest>
    PendingRequest::g_requests;
} // namespace
extern "C" {
/// @brief Wrapper of MPI_Init
static int mpi_initialize(int32_t *argc, char ***argv) {
  try {
    auto mpiMod = py::module::import("mpi4py.MPI");
    if (mpiMod.attr("Is_initialized")().cast<bool>())
      return 0;

    mpiMod.attr("Init")();
    initCalledByThis = true;
    return 0;
  } catch (std::exception &e) {
    std::cerr << "[mpi4py] Caught exception \"" << e.what() << "\"\n";
    return 1;
  }
}

/// @brief Wrapper of MPI_Finalize
static int mpi_finalize() {
  PendingRequest::g_requests.clear();
  if (!initCalledByThis)
    return 0;
  try {
    auto mpiMod = py::module::import("mpi4py.MPI");
    if (mpiMod.attr("Is_finalized")().cast<bool>())
      return 0;

    mpiMod.attr("Finalize")();
    return 0;
  } catch (std::exception &e) {
    std::cerr << "[mpi4py] Caught exception \"" << e.what() << "\"\n";
    return 1;
  }
}

/// @brief Wrapper of MPI_Initialized
static int mpi_initialized(int32_t *flag) {
  try {
    auto mpiMod = py::module::import("mpi4py.MPI");
    if (mpiMod.attr("Is_initialized")().cast<bool>())
      *flag = 1;
    else
      *flag = 0;

    return 0;
  } catch (std::exception &e) {
    std::cerr << "[mpi4py] Caught exception \"" << e.what() << "\"\n";
    return 1;
  }
}

/// @brief Wrapper of MPI_Finalized
static int mpi_finalized(int32_t *flag) {
  try {
    auto mpiMod = py::module::import("mpi4py.MPI");
    if (mpiMod.attr("Is_finalized")().cast<bool>())
      *flag = 1;
    else
      *flag = 0;

    return 0;
  } catch (std::exception &e) {
    std::cerr << "[mpi4py] Caught exception \"" << e.what() << "\"\n";
    return 1;
  }
}

/// @brief Wrapper of MPI_Comm_size
static int mpi_getNumRanks(const cudaqDistributedCommunicator_t *comm,
                           int32_t *size) {
  try {
    auto pyComm = unpackMpiCommunicator(comm);

    *size = pyComm.attr("Get_size")().cast<int>();
    return 0;
  } catch (std::exception &e) {
    std::cerr << "[mpi4py] Caught exception \"" << e.what() << "\"\n";
    return 1;
  }
}

/// @brief Wrapper of MPI_Comm_rank
static int mpi_getProcRank(const cudaqDistributedCommunicator_t *comm,
                           int32_t *rank) {
  try {
    auto pyComm = unpackMpiCommunicator(comm);

    *rank = pyComm.attr("Get_rank")().cast<int>();
    return 0;
  } catch (std::exception &e) {
    std::cerr << "[mpi4py] Caught exception \"" << e.what() << "\"\n";
    return 1;
  }
}

/// @brief Returns the size of the local subgroup of processes sharing node
/// memory
static int mpi_getCommSizeShared(const cudaqDistributedCommunicator_t *comm,
                                 int32_t *numRanks) {
  *numRanks = 0;
  try {
    auto mpiMod = py::module::import("mpi4py.MPI");
    auto COMM_TYPE_SHARED = mpiMod.attr("COMM_TYPE_SHARED");
    auto MPI_Info = mpiMod.attr("Info");
    auto info = MPI_Info.attr("Create")();
    info.attr("Set")("mpi_hw_resource_type", "mpi_shared_memory");
    auto pyComm = unpackMpiCommunicator(comm);
    int procRank = pyComm.attr("Get_rank")().cast<int>();
    auto localComm =
        pyComm.attr("Split_type")(COMM_TYPE_SHARED, procRank, info);

    int nranks = localComm.attr("Get_size")().cast<int>();
    *numRanks = nranks;
    localComm.attr("Free")();

    return 0;
  } catch (std::exception &e) {
    std::cerr << "[mpi4py] Caught exception \"" << e.what() << "\"\n";
    return 1;
  }
}

/// @brief Wrapper of MPI_Barrier
static int mpi_Barrier(const cudaqDistributedCommunicator_t *comm) {
  try {
    auto pyComm = unpackMpiCommunicator(comm);

    pyComm.attr("Barrier")();
    return 0;
  } catch (std::exception &e) {
    std::cerr << "[mpi4py] Caught exception \"" << e.what() << "\"\n";
    return 1;
  }
}

/// @brief Helper to pack native data as mpi4py's memory/buffer object
static py::object packData(const void *buffer, int32_t count, DataType dataType,
                           bool readOnly = false) {
  auto mpiMod = py::module::import("mpi4py.MPI");
  auto mpiBuffer = mpiMod.attr("memory");
  const auto bBytes = getDataSize(dataType) * count;
  auto pyBuffer =
      mpiBuffer.attr("fromaddress")((int64_t)buffer, bBytes, readOnly);
  return pyBuffer;
}

/// @brief Wrapper of MPI_Bcast
static int mpi_Bcast(const cudaqDistributedCommunicator_t *comm, void *buffer,
                     int32_t count, DataType dataType, int32_t rootRank) {
  try {
    auto pyComm = unpackMpiCommunicator(comm);

    pyComm.attr("Bcast")(packData(buffer, count, dataType), rootRank);
    return 0;
  } catch (std::exception &e) {
    std::cerr << "[mpi4py] Caught exception \"" << e.what() << "\"\n";
    return 1;
  }
}

/// @brief Wrapper of MPI_Allreduce
static int mpi_Allreduce(const cudaqDistributedCommunicator_t *comm,
                         const void *sendBuffer, void *recvBuffer,
                         int32_t count, DataType dataType, ReduceOp opType) {
  try {
    auto pyComm = unpackMpiCommunicator(comm);
    auto pyMpiType = (opType == MIN_LOC) ? convertTypeMinLoc(dataType)
                                         : convertType(dataType);
    py::tuple sendBuf = py::make_tuple(
        packData(sendBuffer, count, dataType, true), count, pyMpiType);
    py::tuple recvBuf =
        py::make_tuple(packData(recvBuffer, count, dataType), count, pyMpiType);
    pyComm.attr("Allreduce")(sendBuf, recvBuf, convertType(opType));
    return 0;
  } catch (std::exception &e) {
    std::cerr << "[mpi4py] Caught exception \"" << e.what() << "\"\n";
    return 1;
  }
}

/// @brief Wrapper of MPI_Allreduce with MPI_IN_PLACE
static int mpi_AllreduceInplace(const cudaqDistributedCommunicator_t *comm,
                                void *recvBuffer, int32_t count,
                                DataType dataType, ReduceOp opType) {
  try {
    auto mpiMod = py::module::import("mpi4py.MPI");
    auto inplaceAddr = mpiMod.attr("IN_PLACE");
    py::tuple recvBuf = py::make_tuple(packData(recvBuffer, count, dataType),
                                       count, convertType(dataType));
    auto pyComm = unpackMpiCommunicator(comm);

    pyComm.attr("Allreduce")(inplaceAddr, recvBuf, convertType(opType));
    return 0;
  } catch (std::exception &e) {
    std::cerr << "[mpi4py] Caught exception \"" << e.what() << "\"\n";
    return 1;
  }
}

/// @brief Wrapper of MPI_Allgather
static int mpi_Allgather(const cudaqDistributedCommunicator_t *comm,
                         const void *sendBuffer, void *recvBuffer,
                         int32_t count, DataType dataType) {
  try {
    auto pyComm = unpackMpiCommunicator(comm);
    const auto size = pyComm.attr("Get_size")().cast<int>();
    pyComm.attr("Allgather")(packData(sendBuffer, count, dataType, true),
                             packData(recvBuffer, size * count, dataType));
    return 0;
  } catch (std::exception &e) {
    std::cerr << "[mpi4py] Caught exception \"" << e.what() << "\"\n";
    return 1;
  }
}

/// @brief Wrapper of MPI_Allgatherv
static int mpi_AllgatherV(const cudaqDistributedCommunicator_t *comm,
                          const void *sendBuf, int sendCount, void *recvBuf,
                          const int *recvCounts, const int *displs,
                          DataType dataType) {
  try {
    auto pyComm = unpackMpiCommunicator(comm);
    const auto size = pyComm.attr("Get_size")().cast<int>();
    py::array_t<int> pyRecvCounts(size);
    py::array_t<int> pyDispls(size);
    auto pyRecvCountsData = pyRecvCounts.mutable_unchecked<1>();
    auto pyDisplsData = pyDispls.mutable_unchecked<1>();
    for (int i = 0; i < size; ++i) {
      pyRecvCountsData(i) = recvCounts[i];
      pyDisplsData(i) = displs[i];
    }
    pyComm.attr("Allgatherv")(
        packData(sendBuf, sendCount, dataType, true),
        py::make_tuple(packData(recvBuf, size * sendCount, dataType),
                       pyRecvCounts, pyDispls, convertType(dataType)));
    return 0;
  } catch (std::exception &e) {
    std::cerr << "[mpi4py] Caught exception \"" << e.what() << "\"\n";
    return 1;
  }
  return 0;
}

/// @brief Wrapper of MPI_Isend and track pending requests for synchronization
static int mpi_SendAsync(const cudaqDistributedCommunicator_t *comm,
                         const void *buf, int count, DataType dataType,
                         int peer, int32_t tag) {
  std::scoped_lock<std::mutex> lock(PendingRequest::g_mutex);
  if (PendingRequest::g_requests[comm].nActiveRequests == 2)
    return -1;
  try {
    auto pyComm = unpackMpiCommunicator(comm);
    auto request =
        pyComm.attr("Isend")(packData(buf, count, dataType, true), peer, tag);
    PendingRequest::g_requests[comm]
        .requests[PendingRequest::g_requests[comm].nActiveRequests] = request;
    ++PendingRequest::g_requests[comm].nActiveRequests;
    return 0;
  } catch (std::exception &e) {
    std::cerr << "[mpi4py] Caught exception \"" << e.what() << "\"\n";
    return 1;
  }
  return 0;
}

/// @brief Wrapper of MPI_Irecv and track pending requests for synchronization
static int mpi_RecvAsync(const cudaqDistributedCommunicator_t *comm, void *buf,
                         int count, DataType dataType, int peer, int32_t tag) {
  std::scoped_lock<std::mutex> lock(PendingRequest::g_mutex);
  if (PendingRequest::g_requests[comm].nActiveRequests == 2)
    return -1;
  try {
    auto pyComm = unpackMpiCommunicator(comm);
    auto request =
        pyComm.attr("Irecv")(packData(buf, count, dataType), peer, tag);
    PendingRequest::g_requests[comm]
        .requests[PendingRequest::g_requests[comm].nActiveRequests] = request;
    ++PendingRequest::g_requests[comm].nActiveRequests;
    return 0;
  } catch (std::exception &e) {
    std::cerr << "[mpi4py] Caught exception \"" << e.what() << "\"\n";
    return 1;
  }
  return 0;
}

/// @brief Combined MPI_Isend and MPI_Irecv requests
static int mpi_SendRecvAsync(const cudaqDistributedCommunicator_t *comm,
                             const void *sendbuf, void *recvbuf, int count,
                             DataType dataType, int peer, int32_t tag) {
  std::scoped_lock<std::mutex> lock(PendingRequest::g_mutex);
  if (PendingRequest::g_requests[comm].nActiveRequests != 0)
    return -1;
  try {
    auto pyComm = unpackMpiCommunicator(comm);
    auto sendRequest = pyComm.attr("Isend")(
        packData(sendbuf, count, dataType, true), peer, tag);
    auto recvRequest =
        pyComm.attr("Irecv")(packData(recvbuf, count, dataType), peer, tag);
    PendingRequest::g_requests[comm].requests[0] = sendRequest;
    PendingRequest::g_requests[comm].requests[1] = recvRequest;
    PendingRequest::g_requests[comm].nActiveRequests = 2;
    return 0;
  } catch (std::exception &e) {
    std::cerr << "[mpi4py] Caught exception \"" << e.what() << "\"\n";
    return 1;
  }
  return 0;
}

/// @brief Wait for in-flight MPI_Isend and MPI_Irecv to complete
static int mpi_Synchronize(const cudaqDistributedCommunicator_t *comm) {
  std::scoped_lock<std::mutex> lock(PendingRequest::g_mutex);
  try {
    auto pyComm = unpackMpiCommunicator(comm);
    for (int i = 0; i < PendingRequest::g_requests[comm].nActiveRequests; ++i) {
      PendingRequest::g_requests[comm].requests[i].attr("Wait")();
    }
    PendingRequest::g_requests[comm].nActiveRequests = 0;
    return 0;
  } catch (std::exception &e) {
    std::cerr << "[mpi4py] Caught exception \"" << e.what() << "\"\n";
    return 1;
  }
  return 0;
}

/// @brief Wrapper of MPI_Abort
static int mpi_Abort(const cudaqDistributedCommunicator_t *comm,
                     int errorCode) {
  try {
    auto pyComm = unpackMpiCommunicator(comm);
    pyComm.attr("Abort")(errorCode);
    return 0;
  } catch (std::exception &e) {
    std::cerr << "[mpi4py] Caught exception \"" << e.what() << "\"\n";
    return 1;
  }
}

/// @brief Helper to convert mpi4py's Comm handle to void*
static void *voidPtrCast(py::handle src) {
  PyObject *source = src.ptr();
  PyObject *tmp = PyNumber_Long(source);
  if (!tmp)
    return nullptr;
  void *casted = PyLong_AsVoidPtr(tmp);
  Py_DECREF(tmp);
  return casted;
}

/// @brief Wrapper of MPI_Comm_dup
static int mpi_CommDup(const cudaqDistributedCommunicator_t *comm,
                       cudaqDistributedCommunicator_t **newDupComm) {
  try {
    auto mpiMod = py::module::import("mpi4py.MPI");
    auto pyComm = unpackMpiCommunicator(comm);
    // Use std::deque to make sure pointers to elements are valid.
    static std::deque<std::pair<cudaqDistributedCommunicator_t, void *>>
        dup_comms;

    const auto dup = pyComm.attr("Dup")();
    dup_comms.emplace_back(std::pair<cudaqDistributedCommunicator_t, void *>());
    auto &[newComm, commPtr] = dup_comms.back();
    commPtr = voidPtrCast(mpiMod.attr("_handleof")(dup));
    newComm.commPtr = &commPtr;
    newComm.commSize = mpiMod.attr("_sizeof")(dup).cast<std::size_t>();
    *newDupComm = &newComm;
    return 0;
  } catch (std::exception &e) {
    std::cerr << "[mpi4py] Caught exception \"" << e.what() << "\"\n";
    return 1;
  }
}

/// @brief Wrapper of MPI_Comm_split
static int mpi_CommSplit(const cudaqDistributedCommunicator_t *comm,
                         int32_t color, int32_t key,
                         cudaqDistributedCommunicator_t **newSplitComm) {
  try {
    auto mpiMod = py::module::import("mpi4py.MPI");
    auto pyComm = unpackMpiCommunicator(comm);
    // Use std::deque to make sure pointers to elements are valid.
    static std::deque<std::pair<cudaqDistributedCommunicator_t, void *>>
        split_comms;

    const auto split = pyComm.attr("Split")(color, key);
    split_comms.emplace_back(
        std::pair<cudaqDistributedCommunicator_t, void *>());
    auto &[newComm, commPtr] = split_comms.back();
    commPtr = voidPtrCast(mpiMod.attr("_handleof")(split));
    newComm.commPtr = &commPtr;
    newComm.commSize = mpiMod.attr("_sizeof")(split).cast<std::size_t>();
    *newSplitComm = &newComm;
    return 0;
  } catch (std::exception &e) {
    std::cerr << "[mpi4py] Caught exception \"" << e.what() << "\"\n";
    return 1;
  }
}

/// @brief Return the underlying MPI_Comm as a type-erased object
cudaqDistributedCommunicator_t *getMpiCommunicator() {
  static cudaqDistributedCommunicator_t commWorld;
  try {
    auto mpiMod = py::module::import("mpi4py.MPI");
    auto pyCommWorld = mpiMod.attr("COMM_WORLD");
    static auto commPtr =
        (void *)(mpiMod.attr("_handleof")(pyCommWorld).cast<int64_t>());
    commWorld.commPtr = &commPtr;
    commWorld.commSize =
        mpiMod.attr("_sizeof")(pyCommWorld).cast<std::size_t>();
  } catch (std::exception &) {
    commWorld.commPtr = nullptr;
    commWorld.commSize = 0;
  }
  return &commWorld;
}

/// @brief Return the MPI shim interface (as a function table)
cudaqDistributedInterface_t *getDistributedInterface() {
  static cudaqDistributedInterface_t cudaqDistributedInterface{
      CUDAQ_DISTRIBUTED_INTERFACE_VERSION,
      mpi_initialize,
      mpi_finalize,
      mpi_initialized,
      mpi_finalized,
      mpi_getNumRanks,
      mpi_getProcRank,
      mpi_getCommSizeShared,
      mpi_Barrier,
      mpi_Bcast,
      mpi_Allreduce,
      mpi_AllreduceInplace,
      mpi_Allgather,
      mpi_AllgatherV,
      mpi_SendAsync,
      mpi_RecvAsync,
      mpi_SendRecvAsync,
      mpi_Synchronize,
      mpi_Abort,
      mpi_CommDup,
      mpi_CommSplit};
  return &cudaqDistributedInterface;
}
}

__attribute__((constructor)) void dllMain() {
  if (Py_IsInitialized() == 0) {
    // Create a scoped interpreter if none exists.
    // Note: if this was invoked from Python, an interpreter is already active.
    interp = std::make_unique<py::scoped_interpreter>();
  }
  try {
    py::module::import("mpi4py");
  } catch (std::exception &e) {
    // mpi4py not installed
    mpi4pyFound = false;
    return;
  }
  mpi4pyFound = true;
  // Disable auto init
  // https://mpi4py.readthedocs.io/en/stable/mpi4py.html#mpi4py.mpi4py.rc.initialize
  auto mpiRcMod = py::module::import("mpi4py.rc");
  mpiRcMod.attr("initialize") = false;
}