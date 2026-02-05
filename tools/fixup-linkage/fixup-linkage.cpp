/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// The fixup-linkage tool is used to rewrite the LLVM IR produced by clang for
/// the classical compute code such that it can be linked correctly with the
/// LLVM IR that is generated for the quantum code. This avoids linker errors
/// such as "duplicate symbol definition".

#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <fstream>
#include <iostream>
#include <regex>
#include <set>

int main(int argc, char *argv[]) {
  if (argc != 4) {
    std::cerr << "Usage:\n\tfixup-linkage <Quake-file> <LLVM-file> <output>\n";
    return 1;
  }

  // 1. Look for all the mangled kernel names. These will be found in the
  // mangled_name_map in the quake file. Add these names to `funcs`.
  std::ifstream modFile(argv[1]);
  std::string line;
  std::set<std::string> funcs;
  {
    std::regex mapRegex{"quake\\.mangled_name_map = [{]"};
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
        funcs.insert(s.substr(1, s.size() - 2));
      }
    }
    modFile.close();
    if (funcs.empty()) {
      std::cerr << "No mangled name map in the quake file.\n";
      return 1;
    }
  }

  // 2. Parse the LLVM IR file using LLVM APIs.
  llvm::LLVMContext context;
  llvm::SMDiagnostic err;
  std::unique_ptr<llvm::Module> module =
      llvm::parseIRFile(argv[2], err, context);

  if (!module) {
    err.print(argv[0], llvm::errs());
    return 1;
  }

  // 3. For each kernel function, change its linkage to linkonce_odr if needed.
  for (llvm::Function &func : *module) {
    if (func.isDeclaration())
      continue;

    // Check if this function is one of our kernels.
    std::string funcName = func.getName().str();
    if (!funcs.contains(funcName))
      continue;

    // Skip if already linkonce_odr or weak linkage.
    auto linkage = func.getLinkage();
    if (linkage == llvm::GlobalValue::LinkOnceODRLinkage ||
        linkage == llvm::GlobalValue::WeakAnyLinkage ||
        linkage == llvm::GlobalValue::WeakODRLinkage)
      continue;

    // Change to linkonce_odr with dso_preemptable.
    func.setLinkage(llvm::GlobalValue::LinkOnceODRLinkage);
    func.setDSOLocal(false);
  }

  // 4. Write the modified module to the output file.
  std::error_code ec;
  llvm::raw_fd_ostream outFile(argv[3], ec, llvm::sys::fs::OF_Text);
  if (ec) {
    std::cerr << "Error opening output file: " << ec.message() << "\n";
    return 1;
  }

  module->print(outFile, nullptr);
  outFile.close();

  return 0;
}
