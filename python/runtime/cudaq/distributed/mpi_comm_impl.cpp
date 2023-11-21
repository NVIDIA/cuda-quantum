/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "distributed_capi.h"
#include <complex>
#include <iostream>
#include <pybind11/embed.h>
#include <stdexcept>

namespace py = pybind11;

namespace {

/// @brief Reference to the pybind11 scoped interpreter
thread_local static std::unique_ptr<py::scoped_interpreter> interp;

static bool mpi4pyFound = false;

bool initCalledByThis = false;
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
} // namespace
extern "C" {

int mpi_initialize(int32_t *argc, char ***argv) {
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

int mpi_finalize() {
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

int mpi_initialized(int32_t *flag) {
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

int mpi_finalized(int32_t *flag) {
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

int mpi_getNumRanks(const cudaqDistributedCommunicator_t *comm, int32_t *size) {
  auto pyComm = unpackMpiCommunicator(comm);
  try {
    *size = pyComm.attr("Get_size")().cast<int>();
    return 0;
  } catch (std::exception &e) {
    std::cerr << "[mpi4py] Caught exception \"" << e.what() << "\"\n";
    return 1;
  }
}

int mpi_getProcRank(const cudaqDistributedCommunicator_t *comm, int32_t *rank) {
  auto pyComm = unpackMpiCommunicator(comm);
  try {
    *rank = pyComm.attr("Get_rank")().cast<int>();
    return 0;
  } catch (std::exception &e) {
    std::cerr << "[mpi4py] Caught exception \"" << e.what() << "\"\n";
    return 1;
  }
}

int mpi_getCommSizeShared(const cudaqDistributedCommunicator_t *comm,
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

int mpi_Barrier(const cudaqDistributedCommunicator_t *comm) {
  auto pyComm = unpackMpiCommunicator(comm);
  try {
    pyComm.attr("Barrier")();
    return 0;
  } catch (std::exception &e) {
    std::cerr << "[mpi4py] Caught exception \"" << e.what() << "\"\n";
    return 1;
  }
}

static py::object packData(const void *buffer, int32_t count, DataType dataType,
                           bool readOnly = false) {
  auto mpiMod = py::module::import("mpi4py.MPI");
  auto mpiBuffer = mpiMod.attr("memory");
  const auto bBytes = getDataSize(dataType) * count;
  auto pyBuffer =
      mpiBuffer.attr("fromaddress")((int64_t)buffer, bBytes, readOnly);
  return pyBuffer;
}

int mpi_Bcast(const cudaqDistributedCommunicator_t *comm, void *buffer,
              int32_t count, DataType dataType, int32_t rootRank) {
  auto pyComm = unpackMpiCommunicator(comm);
  try {
    pyComm.attr("Bcast")(packData(buffer, count, dataType), rootRank);
    return 0;
  } catch (std::exception &e) {
    std::cerr << "[mpi4py] Caught exception \"" << e.what() << "\"\n";
    return 1;
  }
}

int mpi_Allreduce(const cudaqDistributedCommunicator_t *comm,
                  const void *sendBuffer, void *recvBuffer, int32_t count,
                  DataType dataType, ReduceOp opType) {
  py::tuple sendBuf =
      py::make_tuple(packData(sendBuffer, count, dataType, true), count,
                     convertType(dataType));
  
  auto pyComm = unpackMpiCommunicator(comm);
  try {
    if (opType == MIN_LOC) {
      py::tuple recvBuf = py::make_tuple(packData(recvBuffer, 1, INT_64), 1,
                                         convertTypeMinLoc(dataType));
      pyComm.attr("Allreduce")(sendBuf, recvBuf, convertType(opType));
    } else {
      py::tuple recvBuf = py::make_tuple(packData(recvBuffer, count, dataType),
                                         count, convertType(dataType));
      pyComm.attr("Allreduce")(sendBuf, recvBuf, convertType(opType));
    }
    return 0;
  } catch (std::exception &e) {
    std::cerr << "[mpi4py] Caught exception \"" << e.what() << "\"\n";
    return 1;
  }
}

int mpi_AllreduceInplace(const cudaqDistributedCommunicator_t *comm,
                         void *recvBuffer, int32_t count, DataType dataType,
                         ReduceOp opType) {
  auto mpiMod = py::module::import("mpi4py.MPI");
  auto inplaceAddr = mpiMod.attr("IN_PLACE");
  py::tuple recvBuf = py::make_tuple(packData(recvBuffer, count, dataType),
                                     count, convertType(dataType));
  auto pyComm = unpackMpiCommunicator(comm);
  try {
    pyComm.attr("Allreduce")(inplaceAddr, recvBuf, convertType(opType));
    return 0;
  } catch (std::exception &e) {
    std::cerr << "[mpi4py] Caught exception \"" << e.what() << "\"\n";
    return 1;
  }
}

int mpi_Allgather(const cudaqDistributedCommunicator_t *comm,
                  const void *sendBuffer, void *recvBuffer, int32_t count,
                  DataType dataType) {
  auto pyComm = unpackMpiCommunicator(comm);
  const auto size = pyComm.attr("Get_size")().cast<int>();
  try {
    pyComm.attr("Allgather")(packData(sendBuffer, count, dataType, true),
                             packData(recvBuffer, size * count, dataType));
    return 0;
  } catch (std::exception &e) {
    std::cerr << "[mpi4py] Caught exception \"" << e.what() << "\"\n";
    return 1;
  }
}

int mpi_CommDup(const cudaqDistributedCommunicator_t *comm,
                cudaqDistributedCommunicator_t **newDupComm) {
  auto mpiMod = py::module::import("mpi4py.MPI");
  auto pyComm = unpackMpiCommunicator(comm);
  // Use std::deque to make sure pointers to elements are valid.
  static std::deque<std::pair<cudaqDistributedCommunicator_t, void *>>
      dup_comms;
  try {
    const auto dup = pyComm.attr("Dup")();
    dup_comms.emplace_back(std::pair<cudaqDistributedCommunicator_t, void *>());
    auto &[newComm, commPtr] = dup_comms.back();
    commPtr = (void *)(mpiMod.attr("_handleof")(dup).cast<int64_t>());
    newComm.commPtr = &commPtr;
    newComm.commSize = mpiMod.attr("_sizeof")(dup).cast<std::size_t>();
    *newDupComm = &newComm;
    return 0;
  } catch (std::exception &e) {
    std::cerr << "[mpi4py] Caught exception \"" << e.what() << "\"\n";
    return 1;
  }
}

int mpi_CommSplit(const cudaqDistributedCommunicator_t *comm, int32_t color,
                  int32_t key, cudaqDistributedCommunicator_t **newSplitComm) {
  auto mpiMod = py::module::import("mpi4py.MPI");
  auto pyComm = unpackMpiCommunicator(comm);
  // Use std::deque to make sure pointers to elements are valid.
  static std::deque<std::pair<cudaqDistributedCommunicator_t, void *>>
      split_comms;
  try {
    const auto split = pyComm.attr("Split")(color, key);
    split_comms.emplace_back(
        std::pair<cudaqDistributedCommunicator_t, void *>());
    auto &[newComm, commPtr] = split_comms.back();
    commPtr = (void *)(mpiMod.attr("_handleof")(split).cast<int64_t>());
    newComm.commPtr = &commPtr;
    newComm.commSize = mpiMod.attr("_sizeof")(split).cast<std::size_t>();
    *newSplitComm = &newComm;
    return 0;
  } catch (std::exception &e) {
    std::cerr << "[mpi4py] Caught exception \"" << e.what() << "\"\n";
    return 1;
  }
}

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
      mpi_CommDup,
      mpi_CommSplit};
  return &cudaqDistributedInterface;
}
}

__attribute__((constructor)) void dllMain() {
  interp = std::make_unique<py::scoped_interpreter>();
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