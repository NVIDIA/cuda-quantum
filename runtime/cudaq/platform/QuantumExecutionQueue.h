/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/MeasureCounts.h"
#include <condition_variable>
#include <functional>
#include <future>
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
/// separate thread from the main CUDA Quantum host thread that clients
/// can submit execution tasks to, and these tasks will be executed
/// (asynchronously from the calling thread) in the order they are submitted.
class QuantumExecutionQueue {
public:
  /// The Constructor
  QuantumExecutionQueue();
  /// The Destructor
  ~QuantumExecutionQueue();

  /// Enqueue a Sampling task.
  void enqueue(QuantumTask &task);

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

  /// Main execution thread, loops until destruction,
  /// continuously pops tasks off the queue and executes them
  void handler(void);
};
} // namespace cudaq
