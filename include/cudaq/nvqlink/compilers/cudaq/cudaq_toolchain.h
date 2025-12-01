/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include <dlfcn.h>
#include <memory>
#include <sstream>
#include <vector>

namespace cudaq::nvqlink {

struct toolchain_options {
  std::string tempDir = "/tmp";
  std::string optimizationLevel = "-O3";
  std::vector<std::string> cudaqOptPasses = {"--canonicalize"};
  std::string qirTarget = "qir"; // or "qir-base"
  bool keepTempFiles = true;
  std::string targetTriple = "x86_64-unknown-linux-gnu";
};

// Base class for all command line tools
class command_line_tool {
public:
  struct tool_result {
    std::string outputFile;
    std::string content;
    int exitCode = 0;
    std::string errorMessage;

    explicit operator bool() const { return exitCode == 0; }
  };

  command_line_tool(const std::string &toolName) : toolName_(toolName) {}
  virtual ~command_line_tool() = default;

  virtual tool_result execute(const std::string &inputFile) = 0;
  virtual tool_result executeWithContent(const std::string &content) = 0;

protected:
  std::string toolName_;
  std::string tempDir_ = "/tmp";
  std::vector<std::string> tempFiles_;

  std::string executeCommand(const std::string &command);
  std::string generateUniqueId();
  std::string createTempFile(const std::string &content,
                             const std::string &suffix);
  void addTempFile(const std::string &file) { tempFiles_.push_back(file); }

public:
  void setTempDir(const std::string &tempDir) { tempDir_ = tempDir; }
  const std::vector<std::string> &getTempFiles() const { return tempFiles_; }
};
} // namespace cudaq::nvqlink

namespace cudaq {
using namespace nvqlink;

// cudaq-opt tool
class opt : public command_line_tool {
private:
  std::vector<std::string> passes_;

public:
  opt(const std::vector<std::string> &passes = {"--canonicalize"})
      : command_line_tool("cudaq-opt"), passes_(passes) {}

  tool_result execute(const std::string &inputFile) override;
  tool_result executeWithContent(const std::string &content) override;

  opt &addPass(const std::string &pass) {
    passes_.push_back(pass);
    return *this;
  }
};

// cudaq-translate tool
class translate : public command_line_tool {
private:
  std::string target_;

public:
  translate(const std::string &target = "qir")
      : command_line_tool("cudaq-translate"), target_(target) {}

  tool_result execute(const std::string &inputFile) override;
  tool_result executeWithContent(const std::string &content) override;

  translate &setTarget(const std::string &target) {
    target_ = target;
    return *this;
  }
};
} // namespace cudaq
using namespace cudaq::nvqlink;

namespace llvm {
// LLC tool
class llc : public command_line_tool {
private:
  std::string targetTriple_;

public:
  llc(const std::string &targetTriple = "x86_64-unknown-linux-gnu")
      : command_line_tool("llc"), targetTriple_(targetTriple) {}

  tool_result execute(const std::string &inputFile) override;
  tool_result executeWithContent(const std::string &content) override;

  llc &setTargetTriple(const std::string &targetTriple) {
    targetTriple_ = targetTriple;
    return *this;
  }
};
} // namespace llvm

namespace clang {
// Clang tool
class linker : public command_line_tool {
private:
  std::string cudaqLibraryPath;
  std::string optimizationLevel_;
  bool shared_;
  bool pic_;

public:
  linker(const std::string &cudaq, const std::string &optimizationLevel = "-O3",
         bool shared = true, bool pic = true)
      : command_line_tool("clang"), cudaqLibraryPath(cudaq),
        optimizationLevel_(optimizationLevel), shared_(shared), pic_(pic) {}

  tool_result execute(const std::string &inputFile) override;
  tool_result executeWithContent(const std::string &content) override;
  tool_result linkMultiple(const std::vector<std::string> &objectFiles);

  linker &setOptimizationLevel(const std::string &level) {
    optimizationLevel_ = level;
    return *this;
  }
};
} // namespace clang

using namespace cudaq::nvqlink;
// Pipe operator overloads for chaining tools
command_line_tool::tool_result
operator|(const command_line_tool::tool_result &input, cudaq::opt &tool);
command_line_tool::tool_result
operator|(const command_line_tool::tool_result &input,
          const cudaq::translate &tool);
command_line_tool::tool_result
operator|(const command_line_tool::tool_result &input, const llvm::llc &tool);
command_line_tool::tool_result
operator|(const command_line_tool::tool_result &input, clang::linker &tool);

namespace cudaq {
// Helper class to start a pipeline with content
class pipeline {
private:
  std::string content_;
  std::string tempDir_;

public:
  pipeline(const std::string &content, const std::string &tempDir = "/tmp")
      : content_(content), tempDir_(tempDir) {}

  command_line_tool::tool_result operator|(const cudaq::opt &tool);
};
} // namespace cudaq
