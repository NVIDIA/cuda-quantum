/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ %s -o %t && %t

#include <tuple>
#include <utility>
#include <vector>

int main() {
  // Check that tuples are laid out "backwards" from the recursive definition.
  std::tuple<char, double, int, std::vector<int>, short, float> t = {
      'a', 2.0, 4, {10, 20, 30}, 82, 62.2f};

  if ((void *)&std::get<0>(t) <= (void *)&std::get<1>(t) ||
      (void *)&std::get<1>(t) <= (void *)&std::get<2>(t) ||
      (void *)&std::get<2>(t) <= (void *)&std::get<3>(t) ||
      (void *)&std::get<3>(t) <= (void *)&std::get<4>(t) ||
      (void *)&std::get<4>(t) <= (void *)&std::get<5>(t))
    return 1;

  // Meanwhile check that pairs are laid out "forwards" like a struct.
  std::pair<char, double> p = {'a', 2.0};
  if (((char *)&std::get<0>(p)) + sizeof(double) != (char *)&std::get<1>(p))
    return 2;

  return 0;
}
