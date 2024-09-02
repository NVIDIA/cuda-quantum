/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Dialect/CC/CCOps.h"

#include <complex>
#include <vector>

using namespace mlir;

namespace cudaq {

std::vector<std::complex<double>>
readGlobalConstantArray(cudaq::cc::GlobalOp &global) {
  std::vector<std::complex<double>> result{};

  auto attr = global.getValue();
  auto elementsAttr = cast<mlir::ElementsAttr>(attr.value());
  auto eleTy = elementsAttr.getElementType();
  auto values = elementsAttr.getValues<mlir::Attribute>();

  for (auto it = values.begin(); it != values.end(); ++it) {
    auto valAttr = *it;

    auto v = [&]() -> std::complex<double> {
      if (isa<FloatType>(eleTy))
        return {cast<FloatAttr>(valAttr).getValue().convertToDouble(),
                static_cast<double>(0.0)};
      if (isa<IntegerType>(eleTy))
        return {static_cast<double>(cast<IntegerAttr>(valAttr).getInt()),
                static_cast<double>(0.0)};
      assert(isa<ComplexType>(eleTy));
      auto arrayAttr = cast<mlir::ArrayAttr>(valAttr);
      auto real = cast<FloatAttr>(arrayAttr[0]).getValue().convertToDouble();
      auto imag = cast<FloatAttr>(arrayAttr[1]).getValue().convertToDouble();
      return {real, imag};
    }();

    result.push_back(v);
  }
  return result;
}

} // namespace cudaq
