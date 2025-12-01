/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "cudaq/nvqlink/compilers/cudaq/cudaq_toolchain.h"
#include "../../utils/logger.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <regex>
#include <sstream>
#include <sys/wait.h>
#include <unistd.h>

using namespace cudaq::nvqlink;

// command_line_tool base implementation
std::string command_line_tool::executeCommand(const std::string &command) {
  std::array<char, 128> buffer;
  std::string result;

  Logger::log("[exec cmd] {}", command);
  FILE *pipe = popen((command + " 2>&1").c_str(), "r");
  if (!pipe) {
    throw std::runtime_error("Failed to execute command: " + command);
  }

  while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
    result += buffer.data();
  }

  int exitCode = pclose(pipe);
  if (exitCode != 0) {
    throw std::runtime_error("Command failed with exit code " +
                             std::to_string(exitCode) + ": " + command +
                             "\nOutput: " + result);
  }

  return result;
}

std::string command_line_tool::generateUniqueId() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(1000000, 9999999);
  return std::to_string(dis(gen));
}

std::string command_line_tool::createTempFile(const std::string &content,
                                              const std::string &suffix) {
  std::string filename =
      tempDir_ + "/" + toolName_ + "_" + generateUniqueId() + suffix;
  std::ofstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to create temporary file: " + filename);
  }
  file << content;
  file.close();
  addTempFile(filename);
  return filename;
}

// cudaq::opt implementation
command_line_tool::tool_result
cudaq::opt::execute(const std::string &inputFile) {
  std::string outputFile =
      tempDir_ + "/optimized_" + generateUniqueId() + ".qke";
  addTempFile(outputFile);

  std::stringstream cmd;
  cmd << toolName_;
  for (const auto &pass : passes_) {
    cmd << " " << pass;
  }
  cmd << " " << inputFile << " -o " << outputFile;

  tool_result result;
  try {
    executeCommand(cmd.str());
    if (std::filesystem::exists(outputFile)) {
      result.outputFile = outputFile;
      result.exitCode = 0;

      // Read content for potential piping
      std::ifstream file(outputFile);
      if (file.is_open()) {
        std::ostringstream content;
        content << file.rdbuf();
        result.content = content.str();
      }
    } else {
      result.exitCode = 1;
      result.errorMessage = "Output file not generated";
    }
  } catch (const std::exception &e) {
    result.exitCode = 1;
    result.errorMessage = e.what();
  }

  if (result.content.empty())
    throw std::runtime_error("cudaq::opt error - output is empty.");
  return result;
}

command_line_tool::tool_result
cudaq::opt::executeWithContent(const std::string &content) {
  std::string inputFile = createTempFile(content, ".qke");
  return execute(inputFile);
}

// cudaq::translate implementation
command_line_tool::tool_result
cudaq::translate::execute(const std::string &inputFile) {
  std::string outputFile =
      tempDir_ + "/translated_" + generateUniqueId() + ".ll";
  addTempFile(outputFile);

  std::stringstream cmd;
  cmd << toolName_ << " --convert-to=" << target_ << " " << inputFile << " -o "
      << outputFile;

  tool_result result;
  try {
    executeCommand(cmd.str());
    if (std::filesystem::exists(outputFile)) {
      result.outputFile = outputFile;
      result.exitCode = 0;

      // Read content for potential piping
      std::ifstream file(outputFile);
      if (file.is_open()) {
        std::ostringstream content;
        content << file.rdbuf();
        result.content = content.str();
      }
    } else {
      result.exitCode = 1;
      result.errorMessage = "Output file not generated";
    }
  } catch (const std::exception &e) {
    result.exitCode = 1;
    result.errorMessage = e.what();
  }

  return result;
}

command_line_tool::tool_result
cudaq::translate::executeWithContent(const std::string &content) {
  std::string inputFile = createTempFile(content, ".qke");
  return execute(inputFile);
}

// llvm::llc implementation
command_line_tool::tool_result
llvm::llc::execute(const std::string &inputFile) {
  std::string outputFile = tempDir_ + "/compiled_" + generateUniqueId() + ".o";
  addTempFile(outputFile);

  std::stringstream cmd;
  cmd << toolName_ << " -filetype=obj --relocation-model=pic";
  if (!targetTriple_.empty()) {
    cmd << " -mtriple=" << targetTriple_;
  }
  cmd << " " << inputFile << " -o " << outputFile;

  tool_result result;
  try {
    executeCommand(cmd.str());
    if (std::filesystem::exists(outputFile)) {
      result.outputFile = outputFile;
      result.exitCode = 0;
      // Object files are binary, so we don't read content
    } else {
      result.exitCode = 1;
      result.errorMessage = "Output file not generated";
    }
  } catch (const std::exception &e) {
    result.exitCode = 1;
    result.errorMessage = e.what();
  }

  return result;
}

command_line_tool::tool_result
llvm::llc::executeWithContent(const std::string &content) {
  std::string inputFile = createTempFile(content, ".ll");
  return execute(inputFile);
}

// clang::linker implementation
command_line_tool::tool_result
clang::linker::execute(const std::string &inputFile) {
  std::string outputFile = tempDir_ + "/linked_" + generateUniqueId() + ".so";

  std::stringstream cmd;
  cmd << toolName_;
  if (shared_)
    cmd << " -shared";
  if (pic_)
    cmd << " -fPIC";
  cmd << " " << optimizationLevel_ << " " << inputFile << " -o " << outputFile;

  tool_result result;
  try {
    executeCommand(cmd.str());
    if (std::filesystem::exists(outputFile)) {
      result.outputFile = outputFile;
      result.exitCode = 0;
    } else {
      result.exitCode = 1;
      result.errorMessage = "Output file not generated";
    }
  } catch (const std::exception &e) {
    result.exitCode = 1;
    result.errorMessage = e.what();
  }

  return result;
}

command_line_tool::tool_result
clang::linker::executeWithContent(const std::string &content) {
  std::string inputFile = createTempFile(content, ".o");
  return execute(inputFile);
}

command_line_tool::tool_result
clang::linker::linkMultiple(const std::vector<std::string> &objectFiles) {
  std::string outputFile = tempDir_ + "/linked_" + generateUniqueId() + ".so";

  std::stringstream cmd;
  cmd << toolName_;
  if (shared_)
    cmd << " -shared";
  if (pic_)
    cmd << " -fPIC";
  cmd << " " << optimizationLevel_ << "  -L " << cudaqLibraryPath
      << " -lcudaq -Wl,-rpath," << cudaqLibraryPath;

  for (const auto &objFile : objectFiles) {
    cmd << " " << objFile;
  }
  cmd << " -o " << outputFile;

  tool_result result;
  try {
    executeCommand(cmd.str());
    if (std::filesystem::exists(outputFile)) {
      result.outputFile = outputFile;
      result.exitCode = 0;
    } else {
      result.exitCode = 1;
      result.errorMessage = "Output file not generated";
    }
  } catch (const std::exception &e) {
    result.exitCode = 1;
    result.errorMessage = e.what();
  }

  return result;
}

// Pipe operator implementations
command_line_tool::tool_result
operator|(const command_line_tool::tool_result &input, cudaq::opt &tool) {
  if (!input) {
    return input; // Propagate error
  }

  if (!input.content.empty()) {
    return tool.executeWithContent(input.content);
  } else if (!input.outputFile.empty()) {
    return tool.execute(input.outputFile);
  }

  command_line_tool::tool_result error;
  error.exitCode = 1;
  error.errorMessage = "No valid input for cudaq::opt";
  return error;
}

command_line_tool::tool_result
operator|(const command_line_tool::tool_result &input,
          const cudaq::translate &tool) {
  if (!input) {
    return input; // Propagate error
  }

  if (!input.content.empty()) {
    return const_cast<cudaq::translate &>(tool).executeWithContent(
        input.content);
  } else if (!input.outputFile.empty()) {
    return const_cast<cudaq::translate &>(tool).execute(input.outputFile);
  }

  command_line_tool::tool_result error;
  error.exitCode = 1;
  error.errorMessage = "No valid input for cudaq::translate";
  return error;
}

command_line_tool::tool_result
operator|(const command_line_tool::tool_result &input, const llvm::llc &tool) {
  if (!input) {
    return input; // Propagate error
  }

  if (!input.content.empty()) {
    return const_cast<llvm::llc &>(tool).executeWithContent(input.content);
  } else if (!input.outputFile.empty()) {
    return const_cast<llvm::llc &>(tool).execute(input.outputFile);
  }

  command_line_tool::tool_result error;
  error.exitCode = 1;
  error.errorMessage = "No valid input for llvm::llc";
  return error;
}

command_line_tool::tool_result
operator|(const command_line_tool::tool_result &input, clang::linker &tool) {
  if (!input) {
    return input; // Propagate error
  }

  if (!input.outputFile.empty()) {
    return tool.execute(input.outputFile);
  }

  command_line_tool::tool_result error;
  error.exitCode = 1;
  error.errorMessage = "No valid input for clang::linker";
  return error;
}

// cudaq::pipeline implementation
command_line_tool::tool_result
cudaq::pipeline::operator|(const cudaq::opt &tool) {
  auto &cc = const_cast<cudaq::opt &>(tool);
  cc.setTempDir(tempDir_);
  return cc.executeWithContent(content_);
}
