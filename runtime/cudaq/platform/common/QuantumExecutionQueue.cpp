/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/platform/QuantumExecutionQueue.h"

namespace cudaq {

QuantumExecutionQueue::QuantumExecutionQueue() : lock() {
  thread = std::thread(&QuantumExecutionQueue::handler, this);
}

QuantumExecutionQueue::~QuantumExecutionQueue() {
  std::unique_lock<std::mutex> l(lock);
  quit = true;
  cv.notify_all();
  l.unlock();
  if (thread.joinable()) {
    thread.join();
  }
}

void QuantumExecutionQueue::enqueue(QuantumTask &t) {
  std::unique_lock<std::mutex> l(lock);
  queue.push(t);
  cv.notify_one();
  return;
}

void QuantumExecutionQueue::handler(void) {
  std::unique_lock<std::mutex> l(lock);

  do {
    // Wait until we have data or a quit signal
    cv.wait(l, [this] { return (queue.size() || quit); });

    // after wait, we own the lock
    if (!quit && queue.size()) {

      auto op = std::move(queue.front());
      queue.pop();

      // unlock now that we're done messing with the queue
      l.unlock();

      op();
      l.lock();
    }
  } while (!quit);
}

} // namespace cudaq
