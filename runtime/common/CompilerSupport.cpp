/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Stateless host helpers emitted by kernel lowering. Build these into both
// libcudaq (for existing native clients) and its LLVM-free kernel facade.
#include "common/ThunkInterface.h"
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <vector>

namespace cudaq::support {
extern "C" {
void __nvqpp_initializer_list_to_vector_bool(std::vector<bool> &result,
                                             char *initList, std::size_t size) {
  // result is a sret return value. Make sure it is default initialized. Takes
  // advantage of default empty vector being all 0s.
  std::memset(static_cast<void *>(&result), 0, sizeof(result));
  // Allocate space.
  result.reserve(size);
  // Copy in the initialization list data.
  char *p = initList;
  for (std::size_t i = 0; i < size; ++i, ++p)
    result.push_back(static_cast<bool>(*p));
  // Free the initialization list, which was heap allocated.
  free(initList);
}

/// Construct a block of 0 and 1 bytes that corresponds to the `vector<bool>`
/// values. This gets rid of the bit packing implementation of the
/// `std::vector<bool>` overload. The conversion turns the `std::vector<bool>`
/// into a mock vector structure that looks like `std::vector<char>`. The
/// calling routine must cleanup the buffer allocated by this code.
/// This helper routine may only be called on the host side.
void __nvqpp_vector_bool_to_initializer_list(
    void *outData, const std::vector<bool> &inVec,
    std::vector<char *> **allocations) {
  // The MockVector must be allocated by the caller.
  struct MockVector {
    char *start;
    char *end;
    char *end2;
  };
  MockVector *mockVec = reinterpret_cast<MockVector *>(outData);
  auto outSize = inVec.size();
  // The buffer allocated here must be freed by the caller.
  if (!*allocations)
    *allocations = new std::vector<char *>;
  char *newData = static_cast<char *>(malloc(outSize));
  (*allocations)->push_back(newData);
  mockVec->start = newData;
  mockVec->end2 = mockVec->end = newData + outSize;
  for (unsigned i = 0; i < outSize; ++i)
    newData[i] = static_cast<char>(inVec[i]);
}

/// Destroy the host `std::vector<bool>` that \p vec refers to, releasing its
/// storage. The vector is left in an unspecified state and must not be used
/// again. The compiler cannot know the layout of the host's specialization of
/// `std::vector<bool>`, so the destruction is done here.
/// This helper routine may only be called on the host side.
void __nvqpp_vector_bool_destroy(std::vector<bool> &vec) { vec.~vector(); }

/// Release storage that was obtained with `operator new`, such as the storage
/// of a host `std::vector<T>`. The compiler calls this wrapper instead of the
/// mangled name of `operator delete`, which varies by platform and standard
/// library.
/// This helper routine may only be called on the host side.
void __nvqpp_hostDeallocate(void *ptr) { ::operator delete(ptr); }

/// This helper routine deletes the vector that tracks all the temporaries that
/// were created as well as the temporaries themselves.
/// This routine may only be called on the host side.
void __nvqpp_vector_bool_free_temporary_initlists(
    std::vector<char *> *allocations) {
  for (auto *p : *allocations)
    free(p);
  delete allocations;
}

/// Quasi-portable string helpers for Python (non-C++ frontends).  These library
/// helper functions allow non-C++ front-ends to remain portable with the core
/// layer. As these helpers ought to be built along with the bindings, there
/// should not be a compatibility issue.
const char *__nvqpp_getStringData(const std::string &s) { return s.data(); }
std::uint64_t __nvqpp_getStringSize(const std::string &s) { return s.size(); }

/// Dispatch hook for the generalized, distributed-memory reference
/// `device_call` lowering. This reference implementation assumes the `device`
/// and host share the same process and address space: the compiler-generated
/// marshal code already passes the unmarshal function pointer directly (no
/// registry lookup by name is needed here), so dispatch is just an indirect
/// call through it with the shared communication buffer. \p deviceId, \p name,
/// \p numBlocks, and \p numThreads are unused by this same-process
/// reference implementation; a distributed target's runtime would replace this
/// hook to route the call to the appropriate remote device instead.
cudaq::KernelThunkResultType
__nvqpp__device_callback_run(std::int64_t deviceId, const char *name,
                             void *unmarshalFunc, void *buffer,
                             std::int64_t bufferSize, std::int64_t returnOffset,
                             std::int64_t numBlocks, std::int64_t numThreads) {
  auto thunk = reinterpret_cast<cudaq::KernelThunkType>(unmarshalFunc);
  return thunk(buffer, /*isRemote=*/false);
}
}
} // namespace cudaq::support
