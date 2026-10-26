/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/SampleResult.h"
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

namespace cudaq {

/// The QuantumTask is ultimately what gets added
/// to the execution queue. It is meant to wrap any
/// Sampling or Observe task with an appropriate std::promise
/// instance being provided and set.
using QuantumTask = std::function<void()>;

/// The QuantumExecutionQueue provides a queue running on a
/// separate thread from the main CUDA-Q host thread that clients
/// can submit execution tasks to, and these tasks will be executed
/// (asynchronously from the calling thread) in the order they are submitted.
class QuantumExecutionQueue {
public:
  /// The Constructor
  QuantumExecutionQueue();
  /// The Destructor
  ~QuantumExecutionQueue();

  /// Enqueue a Sampling task. Throws once the queue has been shut down.
  void enqueue(QuantumTask &task);

  /// Stop accepting tasks, discard any still queued, and join the thread,
  /// waiting for the task currently executing.
  void shutdown();

  /// Get id of the thread this queue executes on.
  std::thread::id getExecutionThreadId() const;

protected:
  /// The mutex, used for locking when adding to the queue
  std::mutex lock;

  /// The thread this queue executes on
  std::thread thread;

  /// The execution queue
  std::queue<QuantumTask> queue;

  /// The condition variable used for notifying listeners
  std::condition_variable cv;

  /// Should we quit this thread?
  bool quit = false;

  /// Ensures concurrent shutdown callers wait for completion.
  std::once_flag shutdownFlag;

  /// Main execution thread, loops until destruction,
  /// continuously pops tasks off the queue and executes them
  void handler(void);
};
} // namespace cudaq
