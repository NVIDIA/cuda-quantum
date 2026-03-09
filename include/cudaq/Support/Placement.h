/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Support/Device.h"

namespace cudaq {

/// The `Placement` class contains a mapping between "virtual" and "physical"
/// qubits. The concept of "virtual" qubits corresponds to the original qubit
/// identifier at the input to the program, and the concept of "physical" qubits
/// corresponds to the placement of a virtual qubit onto a `Device` node (aka
/// `Device qubit`).
class Placement {
public:
  struct VirtualQ : Handle {
    using Handle::Handle;
  };
  using DeviceQ = Device::Qubit;

  /// Construct placement object with \p numVr virtual qubits and \p numPhy
  /// physical qubits
  Placement(unsigned numVr, unsigned numPhy)
      : vrToPhy(numVr), phyToVr(numPhy) {}

  /// Returns the number of virtual qubits
  unsigned getNumVirtualQubits() const { return vrToPhy.size(); }

  /// Returns the number of physical qubits on the device
  unsigned getNumDeviceQubits() const { return phyToVr.size(); }

  /// Returns the virtual qubit placed on physical qubit \p phy
  VirtualQ getVr(DeviceQ phy) const {
    assert(phy.isValid() && "Invalid physical qubit");
    return phyToVr[phy.index];
  }

  /// Returns the physical qubit on which virtual qubit \p vr is placed
  DeviceQ getPhy(VirtualQ vr) const {
    assert(vr.isValid() && "Invalid virtual qubit");
    return vrToPhy[vr.index];
  }

  /// Assign virtual qubit \p vr to be placed on physical qubit \p phy
  void map(VirtualQ vr, DeviceQ phy) {
    assert(vr.isValid() || phy.isValid());
    if (vr.isValid())
      vrToPhy[vr.index] = phy;
    if (phy.isValid())
      phyToVr[phy.index] = vr;
  }

  /// Swap the virtual qubits that are physically assigned to \p phy0 and \p
  /// phy1
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

  void dump(llvm::raw_ostream &os = llvm::errs()) const {
    os << "Device qubits:\n";
    for (unsigned i = 0; i < getNumDeviceQubits(); ++i) {
      os << "Q" << i << " -> " << getVr(DeviceQ(i)) << "\n";
    }
    os << "Virtual qubits:\n";
    for (unsigned i = 0; i < getNumVirtualQubits(); ++i) {
      os << "Q" << i << " -> " << getPhy(VirtualQ(i)) << "\n";
    }
  }

private:
  mlir::SmallVector<DeviceQ> vrToPhy;
  mlir::SmallVector<VirtualQ> phyToVr;
};

} // namespace cudaq
