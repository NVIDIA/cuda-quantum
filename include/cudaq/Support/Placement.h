/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Support/Device.h"

namespace cudaq {

class Placement {
public:
  struct VirtualQ : Handle {
    using Handle::Handle;
  };
  using DeviceQ = Device::Qubit;

  Placement(unsigned numVr, unsigned numPhy)
      : vrToPhy(numVr), phyToVr(numPhy) {}

  unsigned getNumVirtualQ() const { return vrToPhy.size(); }

  unsigned getNumDeviceQ() const { return phyToVr.size(); }

  VirtualQ getVr(DeviceQ phy) const {
    assert(phy.isValid() && "Invalid physical qubit");
    return phyToVr[phy.index];
  }

  DeviceQ getPhy(VirtualQ vr) const {
    assert(vr.isValid() && "Invalid virtual qubit");
    return vrToPhy[vr.index];
  }

  void map(VirtualQ vr, DeviceQ phy) {
    assert(vr.isValid() || phy.isValid());
    if (vr.isValid())
      vrToPhy[vr.index] = phy;
    if (phy.isValid())
      phyToVr[phy.index] = vr;
  }

  void swap(DeviceQ phy0, DeviceQ phy1) {
    assert(phy0.isValid() && "Invalid physical qubit");
    assert(phy1.isValid() && "Invalid physical qubit");
    VirtualQ vr0 = phyToVr[phy0.index];
    VirtualQ vr1 = phyToVr[phy1.index];
    if (vr0.isValid())
      vrToPhy[vr0.index] = phy1;
    if (vr1.isValid())
      vrToPhy[vr1.index] = phy0;
    std::swap(phyToVr[phy0.index], phyToVr[phy1.index]);
  }

private:
  mlir::SmallVector<DeviceQ> vrToPhy;
  mlir::SmallVector<VirtualQ> phyToVr;
};

} // namespace cudaq
