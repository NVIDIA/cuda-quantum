/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// The fixup-linkage tool is used to rewrite the LLVM IR produced by clang for
/// the classical compute code such that it can be linked correctly with the
/// LLVM IR that is generated for the quantum code. This avoids linker errors
/// such as "duplicate symbol definition".

#include <fstream>
#include <iostream>
#include <regex>

int main(int argc, char *argv[]) {
  if (argc != 4) {
    std::cerr << "Usage:\n\tfixup-linkage <Quake-file> <LLVM-file> <output>\n";
    return 1;
  }

  // 1. Look for all the mangled kernel names. These will be found in the
  // mangled_name_map in the quake file. Add these names to `funcs`.
  std::ifstream modFile(argv[1]);
  std::string line;
  std::vector<std::string> funcs;
  {
    std::regex mapRegex{"quake\\.mangled_name_map[^\"]*"};
    std::regex stringRegex{"\"(.*?)\""};
    while (std::getline(modFile, line) && funcs.empty()) {
      auto funcsBegin =
          std::sregex_iterator(line.begin(), line.end(), mapRegex);
      auto rgxEnd = std::sregex_iterator();
      if (funcsBegin == rgxEnd)
        continue;
      auto names = line.substr(funcsBegin->str().size() - 1);
      auto namesBegin =
          std::sregex_iterator(names.begin(), names.end(), stringRegex);
      for (std::sregex_iterator i = namesBegin; i != rgxEnd; ++i) {
        auto s = i->str();
        funcs.push_back(s.substr(1, s.size() - 2));
      }
    }
    modFile.close();
    if (funcs.empty()) {
      std::cerr << "No mangled name map in the quake file.\n";
      return 1;
    }
  }

  // 2. Scan the LLVM file looking for the mangled kernel names. Where these
  // kernels are defined, they have their linkage modified to `linkonce_odr` if
  // that is not already the linkage. This change will prevent the duplicate
  // symbols defined error from the linker.
  std::ifstream llFile(argv[2]);
  std::ofstream outFile(argv[3]);
  std::regex filterRegex("^define (dso_local|internal) ");
  auto rgxEnd = std::sregex_iterator();
  while (std::getline(llFile, line)) {
    auto iter = std::sregex_iterator(line.begin(), line.end(), filterRegex);
    if (iter == rgxEnd) {
      outFile << line << std::endl;
      continue;
    }
    bool replaced = false;
    for (auto fn : funcs) {
      auto pos = line.find(fn);
      if (pos == std::string::npos)
        continue;
      auto ms = (*iter)[1].str();
      pos = (ms == "internal") ? sizeof("define internal")
                               : sizeof("define dso_local");
      outFile << "define linkonce_odr dso_preemptable " << line.substr(pos)
              << std::endl;
      replaced = true;
      break;
    }
    if (!replaced)
      outFile << line << std::endl;
  }
  llFile.close();
  outFile.close();
  return 0;
}
