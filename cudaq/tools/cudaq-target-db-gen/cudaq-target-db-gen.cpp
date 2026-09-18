/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Build-time tool that reads every in-tree target's `.yml` file (via the
// exact same parser used at runtime, `cudaq::config::loadTargetConfig`) and
// emits a single generated C++ translation unit containing a literal,
// statically-allocated table of the parsed `cudaq::config::TargetConfig`
// structs. This lets `lookupBuiltinTarget` (TargetDatabase.h) resolve every
// in-tree target with zero YAML parsing and zero disk I/O at runtime.
//
// Usage:
//   cudaq-target-db-gen -o <output.cpp> <name>=<path/to/name.yml> ...

#include "cudaq/Target/TargetConfig.h"
#include "cudaq/Target/TargetDatabase.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include <sstream>

// Freezes this tool's own CUDAQ_TARGET_DB_ABI_SYMBOL_NAME (from whichever
// TargetDatabase.h this tool itself was built against) into a literal
// string, emitted verbatim as the ABI marker symbol name in generated
// output - see TargetDatabase.h for why this must be a frozen value, not a
// macro invocation re-expanded when the generated file is later compiled.
#define CUDAQ_STRINGIFY_IMPL(x) #x
#define CUDAQ_STRINGIFY(x) CUDAQ_STRINGIFY_IMPL(x)

using namespace llvm;

static cl::list<std::string> inputs(cl::Positional,
                                    cl::desc("<name>=<path/to/name.yml> ..."));

static cl::opt<std::string> outputFilename("o",
                                           cl::desc("Specify output filename"),
                                           cl::value_desc("filename"));

static cl::opt<bool> pluginMode(
    "plugin",
    cl::desc("Emit a single-target plugin translation unit (compiled into a "
             "shared library exporting cudaq::config::kTargetPluginSymbolName) "
             "instead of the in-tree, multi-target database table. Exactly one "
             "<name>=<path> input is required."));

namespace {

std::string cxxStringLiteral(std::string_view s) {
  std::string out = "\"";
  for (char c : s) {
    switch (c) {
    case '"':
      out += "\\\"";
      break;
    case '\\':
      out += "\\\\";
      break;
    case '\n':
      out += "\\n";
      break;
    default:
      out += c;
    }
  }
  out += "\"";
  return out;
}

std::string cxxStringVector(const std::vector<std::string> &values) {
  std::string out = "{";
  for (const auto &v : values)
    out += cxxStringLiteral(v) + ", ";
  out += "}";
  return out;
}

std::string cxxOptionalBool(const std::optional<bool> &value) {
  if (!value.has_value())
    return "std::nullopt";
  return value.value() ? "std::optional<bool>(true)"
                       : "std::optional<bool>(false)";
}

std::string cxxArgumentType(cudaq::config::ArgumentType type) {
  using cudaq::config::ArgumentType;
  switch (type) {
  case ArgumentType::String:
    return "cudaq::config::ArgumentType::String";
  case ArgumentType::Int:
    return "cudaq::config::ArgumentType::Int";
  case ArgumentType::UUID:
    return "cudaq::config::ArgumentType::UUID";
  case ArgumentType::FeatureFlag:
    return "cudaq::config::ArgumentType::FeatureFlag";
  case ArgumentType::MachineConfig:
    return "cudaq::config::ArgumentType::MachineConfig";
  }
  return "cudaq::config::ArgumentType::String";
}

std::string
cxxArchitectureSettings(const cudaq::config::TargetArchitectureSettings &s) {
  std::ostringstream os;
  os << "cudaq::config::TargetArchitectureSettings{.CodegenEmission = "
     << cxxStringLiteral(s.CodegenEmission) << "}";
  return os.str();
}

std::string cxxMachineConfigs(
    const std::vector<cudaq::config::MachineArchitectureConfig> &configs) {
  std::ostringstream os;
  os << "{";
  for (const auto &c : configs) {
    os << "cudaq::config::MachineArchitectureConfig{"
       << ".Name = " << cxxStringLiteral(c.Name) << ", "
       << ".MachineNames = " << cxxStringVector(c.MachineNames) << ", "
       << ".MachinePattern = " << cxxStringLiteral(c.MachinePattern) << ", "
       << ".Configuration = " << cxxArchitectureSettings(c.Configuration)
       << "}, ";
  }
  os << "}";
  return os.str();
}

std::string
cxxTargetArguments(const std::vector<cudaq::config::TargetArgument> &args) {
  std::ostringstream os;
  os << "{";
  for (const auto &a : args) {
    os << "cudaq::config::TargetArgument{"
       << ".KeyName = " << cxxStringLiteral(a.KeyName) << ", "
       << ".IsRequired = " << (a.IsRequired ? "true" : "false") << ", "
       << ".PlatformArgKey = " << cxxStringLiteral(a.PlatformArgKey) << ", "
       << ".HelpString = " << cxxStringLiteral(a.HelpString) << ", "
       << ".Type = " << cxxArgumentType(a.Type) << ", "
       << ".MachineConfigs = " << cxxMachineConfigs(a.MachineConfigs) << "}, ";
  }
  os << "}";
  return os.str();
}

std::string cxxConditionalBuildConfigs(
    const std::vector<cudaq::config::ConditionalBuildConfig> &configs) {
  std::ostringstream os;
  os << "{";
  for (const auto &c : configs) {
    os << "cudaq::config::ConditionalBuildConfig{"
       << ".Condition = " << cxxStringLiteral(c.Condition) << ", "
       << ".CompileFlag = " << cxxStringLiteral(c.CompileFlag) << ", "
       << ".LinkFlag = " << cxxStringLiteral(c.LinkFlag) << "}, ";
  }
  os << "}";
  return os.str();
}

std::string
cxxSimulationBackend(const cudaq::config::SimulationBackendSetting &s) {
  return "cudaq::config::SimulationBackendSetting{.values = " +
         cxxStringVector(s.values) + "}";
}

std::string
cxxBackendConfigEntry(const cudaq::config::BackendEndConfigEntry &c) {
  std::ostringstream os;
  os << "cudaq::config::BackendEndConfigEntry{"
     << ".GenTargetBackend = " << cxxOptionalBool(c.GenTargetBackend) << ", "
     << ".LibraryMode = " << cxxOptionalBool(c.LibraryMode) << ", "
     << ".SupportResourceCounts = " << cxxOptionalBool(c.SupportResourceCounts)
     << ", "
     << ".JITHighLevelPipeline = " << cxxStringLiteral(c.JITHighLevelPipeline)
     << ", "
     << ".JITMidLevelPipeline = " << cxxStringLiteral(c.JITMidLevelPipeline)
     << ", "
     << ".JITLowLevelPipeline = " << cxxStringLiteral(c.JITLowLevelPipeline)
     << ", "
     << ".TargetPassPipeline = " << cxxStringLiteral(c.TargetPassPipeline)
     << ", "
     << ".CodegenEmission = " << cxxStringLiteral(c.CodegenEmission) << ", "
     << ".PostCodeGenPasses = " << cxxStringLiteral(c.PostCodeGenPasses) << ", "
     << ".PlatformLibrary = " << cxxStringLiteral(c.PlatformLibrary) << ", "
     << ".LibraryModeExecutionManager = "
     << cxxStringLiteral(c.LibraryModeExecutionManager) << ", "
     << ".PlatformQpu = " << cxxStringLiteral(c.PlatformQpu) << ", "
     << ".PreprocessorDefines = " << cxxStringVector(c.PreprocessorDefines)
     << ", "
     << ".CompilerFlags = " << cxxStringVector(c.CompilerFlags) << ", "
     << ".LinkLibs = " << cxxStringVector(c.LinkLibs) << ", "
     << ".PluginLibraries = " << cxxStringVector(c.PluginLibraries) << ", "
     << ".LinkerFlags = " << cxxStringVector(c.LinkerFlags) << ", "
     << ".SimulationBackend = " << cxxSimulationBackend(c.SimulationBackend)
     << ", "
     << ".ConditionalBuildConfigs = "
     << cxxConditionalBuildConfigs(c.ConditionalBuildConfigs) << "}";
  return os.str();
}

std::string cxxOptionalBackendConfig(
    const std::optional<cudaq::config::BackendEndConfigEntry> &c) {
  if (!c.has_value())
    return "std::nullopt";
  return "std::optional<cudaq::config::BackendEndConfigEntry>(" +
         cxxBackendConfigEntry(c.value()) + ")";
}

std::string
cxxConfigMap(const std::vector<cudaq::config::BackendFeatureMap> &map) {
  std::ostringstream os;
  os << "{";
  for (const auto &m : map) {
    os << "cudaq::config::BackendFeatureMap{"
       << ".Name = " << cxxStringLiteral(m.Name) << ", "
       << ".Flags = static_cast<cudaq::config::TargetFeatureFlag>("
       << static_cast<unsigned>(m.Flags) << "u), "
       << ".Default = " << cxxOptionalBool(m.Default) << ", "
       << ".Config = " << cxxBackendConfigEntry(m.Config) << "}, ";
  }
  os << "}";
  return os.str();
}

std::string cxxTargetConfig(const cudaq::config::TargetConfig &config) {
  std::ostringstream os;
  os << "cudaq::config::TargetConfig{"
     << ".CudaqVersion = " << cxxStringLiteral(config.CudaqVersion) << ", "
     << ".PluginLibraries = " << cxxStringVector(config.PluginLibraries) << ", "
     << ".Name = " << cxxStringLiteral(config.Name) << ", "
     << ".Description = " << cxxStringLiteral(config.Description) << ", "
     << ".WarningMsg = " << cxxStringLiteral(config.WarningMsg) << ", "
     << ".TargetArguments = " << cxxTargetArguments(config.TargetArguments)
     << ", "
     << ".GpuRequired = " << (config.GpuRequired ? "true" : "false") << ", "
     << ".BackendConfig = " << cxxOptionalBackendConfig(config.BackendConfig)
     << ", "
     << ".ConfigMap = " << cxxConfigMap(config.ConfigMap) << "}";
  return os.str();
}

std::string sanitizeIdentifier(std::string_view name) {
  std::string out;
  for (char c : name)
    out += (std::isalnum(static_cast<unsigned char>(c)) ? c : '_');
  return out;
}

} // namespace

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, "CUDA-Q Target Database Generator\n");

  struct Entry {
    std::string name;
    cudaq::config::TargetConfig config;
  };
  std::vector<Entry> entries;
  for (const auto &input : inputs) {
    auto eq = input.find('=');
    if (eq == std::string::npos) {
      errs() << "Malformed input '" << input << "', expected <name>=<path>\n";
      return 1;
    }
    std::string name = input.substr(0, eq);
    std::string path = input.substr(eq + 1);
    entries.push_back({name, cudaq::config::loadTargetConfig(path)});
  }

  if (pluginMode && entries.size() != 1) {
    errs() << "--plugin requires exactly one <name>=<path> input, got "
           << entries.size() << "\n";
    return 1;
  }

  std::error_code ec;
  ToolOutputFile out(outputFilename, ec, sys::fs::OF_None);
  if (ec) {
    errs() << "Failed to open output file '" << outputFilename << "'\n";
    return ec.value();
  }

  raw_ostream &os = out.os();
  if (pluginMode) {
    const auto &entry = entries.front();
    os << "// Generated by cudaq-target-db-gen --plugin. Do not edit.\n"
       << "#include \"cudaq/Target/TargetPluginLibrary.h\"\n\n"
       << "namespace {\n"
       << "const cudaq::config::TargetConfig kPluginTarget_"
       << sanitizeIdentifier(entry.name) << " = "
       << cxxTargetConfig(entry.config) << ";\n"
       << "} // namespace\n\n"
       << "// Symbol name must match cudaq::config::kTargetPluginSymbolName.\n"
       << "extern \"C\" const cudaq::config::TargetConfig *"
       << "cudaq_target_config_v1() {\n"
       << "  return &kPluginTarget_" << sanitizeIdentifier(entry.name) << ";\n"
       << "}\n";
    out.keep();
    return 0;
  }

  os << "// Generated by cudaq-target-db-gen. Do not edit.\n"
     << "#include \"cudaq/Target/TargetDatabase.h\"\n\n"
     // ABI marker (see TargetDatabase.h): frozen at generation time to
     // whatever CUDAQ_TARGET_DB_ABI_VERSION *this* cudaq-target-db-gen
     // binary was built against. A stale generated file left over from
     // before a schema-changing rebuild will still carry the *old* name
     // here, so TargetDatabase.cpp's reference to the *current* name fails
     // to link instead of silently reading mismatched data.
     << "extern \"C\" void " << CUDAQ_STRINGIFY(CUDAQ_TARGET_DB_ABI_SYMBOL_NAME)
     << "() {}\n\n"
     << "namespace cudaq::config {\n"
     << "namespace {\n";
  for (const auto &entry : entries)
    os << "const TargetConfig kBuiltinTarget_" << sanitizeIdentifier(entry.name)
       << " = " << cxxTargetConfig(entry.config) << ";\n";
  os << "} // namespace\n\n"
     << "namespace detail {\n"
     << "const std::vector<std::pair<std::string_view, const TargetConfig "
        "*>> &builtinTargetTable() {\n"
     << "  static const std::vector<std::pair<std::string_view, const "
        "TargetConfig *>> table = {\n";
  for (const auto &entry : entries)
    os << "    {" << cxxStringLiteral(entry.name) << ", &kBuiltinTarget_"
       << sanitizeIdentifier(entry.name) << "},\n";
  os << "  };\n"
     << "  return table;\n"
     << "}\n"
     << "} // namespace detail\n"
     << "} // namespace cudaq::config\n";

  out.keep();
  return 0;
}
