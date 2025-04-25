/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022-2025 NVIDIA Corporation & Affiliates.                    *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "Observer.h"

#include <vector> 

namespace cudaq {
std::vector<GlobalStateObserver *> observers;

void registerObserver(GlobalStateObserver *obs) { observers.push_back(obs); }

void notifyAll(const observer_data &data) {
  for (auto &obs : observers)
    obs->oneWayNotify(data);
}

observer_data notifyWithResponse(const observer_data &data) {
  for (auto &obs : observers)
    if (auto [success, response] = obs->notifyWithResponse(data); success)
      return response;
  return {};
}

} // namespace cudaq
