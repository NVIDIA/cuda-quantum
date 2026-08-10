/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "QDMIQPU.h"

#include "common/ExecutionContext.h"
#include "common/KernelExecution.h"
#include "common/RuntimeTarget.h"
#include "cudaq_internal/compiler/Compiler.h"
#include "nlohmann/json.hpp"
#include "qdmi/Client.hpp"
#include "cudaq/platform.h"
#include "cudaq/platform/qpu_utils.h"
#include "cudaq/runtime/logger/logger.h"

#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <future>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <ranges>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace {

using BackendConfig = std::map<std::string, std::string>;
using JobParameters = std::array<std::optional<qdmi::CustomJobParameter>, 5>;

enum class ProgramEncoding : std::uint8_t { text, qirText, qirModule };

struct ProgramFormat {
  QDMI_Program_Format qdmi;
  std::string_view name;
  std::string_view codegen;
  ProgramEncoding encoding;
};

struct BasisMetadata {
  std::string value;
  std::set<std::string> excludedOperations;
};

// This order is the public automatic-selection policy. Prefer richer QIR
// profiles and binary transport, then the more capable OpenQASM version, and
// keep the provider-specific IQM JSON format as the last resort.
constexpr std::array programFormats{
    ProgramFormat{.qdmi = QDMI_PROGRAM_FORMAT_QIRADAPTIVEMODULE,
                  .name = "qir-adaptive-module",
                  .codegen = "qir-adaptive",
                  .encoding = ProgramEncoding::qirModule},
    ProgramFormat{.qdmi = QDMI_PROGRAM_FORMAT_QIRADAPTIVESTRING,
                  .name = "qir-adaptive-string",
                  .codegen = "qir-adaptive",
                  .encoding = ProgramEncoding::qirText},
    ProgramFormat{.qdmi = QDMI_PROGRAM_FORMAT_QIRBASEMODULE,
                  .name = "qir-base-module",
                  .codegen = "qir-base",
                  .encoding = ProgramEncoding::qirModule},
    ProgramFormat{.qdmi = QDMI_PROGRAM_FORMAT_QIRBASESTRING,
                  .name = "qir-base-string",
                  .codegen = "qir-base",
                  .encoding = ProgramEncoding::qirText},
    ProgramFormat{.qdmi = QDMI_PROGRAM_FORMAT_QASM3,
                  .name = "qasm3",
                  .codegen = "qasm2",
                  .encoding = ProgramEncoding::text},
    ProgramFormat{.qdmi = QDMI_PROGRAM_FORMAT_QASM2,
                  .name = "qasm2",
                  .codegen = "qasm2",
                  .encoding = ProgramEncoding::text},
    ProgramFormat{.qdmi = QDMI_PROGRAM_FORMAT_IQMJSON,
                  .name = "iqm-json",
                  .codegen = "iqm",
                  .encoding = ProgramEncoding::text},
};

constexpr std::string_view iqmPipeline =
    "func.func(erase-implicit-output),lower-to-cfg,iqm-gate-set-mapping,"
    "func.func(add-dealloc,"
    "combine-quantum-alloc,canonicalize,factor-quantum-alloc,dqe,memtoreg),"
    "add-wireset,func.func(assign-wire-indices),qubit-mapping{device="
    "%qdmi_connectivity%},func.func(delay-measurements,regtomem),"
    "func.func(cse,normalize-phase-placement,lower-phase,canonicalize),"
    "iqm-gate-set-mapping";

std::optional<std::string> getValue(const BackendConfig &config,
                                    const std::string_view key) {
  if (const auto iterator = config.find(std::string(key));
      iterator != config.end() && !iterator->second.empty())
    return iterator->second;
  return std::nullopt;
}

qdmi::DeviceSessionConfig
makeSessionConfig(const BackendConfig &backendConfig) {
  qdmi::DeviceSessionConfig config;
  config.custom1 = getValue(backendConfig, "qdmi_session_custom1");
  config.custom2 = getValue(backendConfig, "qdmi_session_custom2");
  config.custom3 = getValue(backendConfig, "qdmi_session_custom3");
  config.custom4 = getValue(backendConfig, "qdmi_session_custom4");
  config.custom5 = getValue(backendConfig, "qdmi_session_custom5");
  return config;
}

JobParameters getJobParameters(const BackendConfig &config) {
  JobParameters parameters;
  for (std::size_t index = 0; index < parameters.size(); ++index) {
    if (const auto value =
            getValue(config, "qdmi_job_custom" + std::to_string(index + 1)))
      parameters[index] = qdmi::CustomJobParameter{*value};
  }
  return parameters;
}

ProgramFormat
selectProgramFormat(const std::vector<QDMI_Program_Format> &supported,
                    const std::optional<std::string> &requested) {
  const auto isSupported = [&](const ProgramFormat &format) {
    return std::ranges::find(supported, format.qdmi) != supported.end();
  };

  if (requested && *requested != "auto") {
    const auto candidate =
        std::ranges::find(programFormats, *requested, &ProgramFormat::name);
    if (candidate == programFormats.end())
      throw std::runtime_error(
          "Unknown QDMI program format '" + *requested +
          "'. Expected auto, qir-adaptive-module, qir-adaptive-string, "
          "qir-base-module, qir-base-string, qasm3, qasm2, or iqm-json.");
    if (!isSupported(*candidate))
      throw std::runtime_error("QDMI device does not support requested "
                               "program format '" +
                               *requested + "'.");
    return *candidate;
  }

  if (const auto candidate = std::ranges::find_if(programFormats, isSupported);
      candidate != programFormats.end())
    return *candidate;

  throw std::runtime_error(
      "QDMI device supports none of CUDA-Q's transport formats.");
}

std::string normalizedOperationName(const qdmi::Operation &operation) {
  auto name = operation.getName();
  std::ranges::transform(name, name.begin(), [](const unsigned char value) {
    return static_cast<char>(std::tolower(value));
  });
  return name;
}

BasisMetadata queryBasis(const qdmi::Device &device) {
  static const std::map<std::string, std::string, std::less<>> aliases{
      {"cnot", "x"},        {"cx", "x"},        {"ccnot", "x"},
      {"ccx", "x"},         {"cz", "z"},        {"cy", "y"},
      {"ch", "h"},          {"cs", "s"},        {"csdg", "s<adj>"},
      {"ct", "t"},          {"ctdg", "t<adj>"}, {"crx", "rx"},
      {"cry", "ry"},        {"crz", "rz"},      {"cp", "r1"},
      {"cu1", "r1"},        {"cu2", "u2"},      {"cu3", "u3"},
      {"p", "r1"},          {"u", "u3"},        {"u1", "r1"},
      {"prx", "phased_rx"}, {"r", "phased_rx"}, {"sdg", "s<adj>"},
      {"tdg", "t<adj>"},
  };
  static const std::set<std::string, std::less<>> cudaqOperations{
      "h",    "phased_rx", "r1",     "rx", "ry", "rz", "s", "s<adj>",
      "swap", "t",         "t<adj>", "u2", "u3", "x",  "y", "z",
  };

  std::set<std::string> basis;
  std::set<std::string> excludedOperations;
  for (const auto &operation : device.getOperations()) {
    auto name = normalizedOperationName(operation);
    const auto qubits = operation.getQubitsNum();
    if (name == "measure" || name == "reset" || name == "barrier")
      continue;

    if (!qubits || *qubits == 0 || *qubits > 3) {
      excludedOperations.emplace(std::move(name));
      continue;
    }

    if (const auto alias = aliases.find(name); alias != aliases.end())
      name = alias->second;
    if (!cudaqOperations.contains(name)) {
      excludedOperations.emplace(std::move(name));
      continue;
    }
    if (name == "swap") {
      basis.emplace(*qubits == 3 ? "swap(1)" : "swap");
      continue;
    }
    basis.emplace(std::move(name) +
                  (*qubits == 1 ? std::string{}
                                : "(" + std::to_string(*qubits - 1) + ")"));
  }

  if (basis.empty())
    throw std::runtime_error(
        "QDMI device advertises no gate operation that CUDA-Q can use as a "
        "compiler basis.");

  std::string serialized;
  for (const auto &gate : basis) {
    if (!serialized.empty())
      serialized += ',';
    serialized += gate;
  }
  return {.value = std::move(serialized),
          .excludedOperations = std::move(excludedOperations)};
}

std::optional<std::vector<std::pair<std::size_t, std::size_t>>>
queryConnectivity(const qdmi::Device &device) {
  const auto couplingMap = device.getCouplingMap();
  if (!couplingMap)
    return std::nullopt;

  const auto sites = device.getRegularSites();
  std::map<std::size_t, std::size_t> positions;
  for (std::size_t position = 0; const auto &site : sites)
    positions.emplace(site.getIndex(), position++);

  std::set<std::pair<std::size_t, std::size_t>> edges;
  for (const auto &[source, target] : *couplingMap) {
    const auto sourcePosition = positions.find(source.getIndex());
    const auto targetPosition = positions.find(target.getIndex());
    if (sourcePosition == positions.end() ||
        targetPosition == positions.end() ||
        sourcePosition->second == targetPosition->second)
      continue;
    edges.emplace(std::min(sourcePosition->second, targetPosition->second),
                  std::max(sourcePosition->second, targetPosition->second));
  }
  return std::vector(edges.begin(), edges.end());
}

class TemporaryFile {
public:
  static TemporaryFile create(const std::string_view filename) {
    std::mt19937_64 generator(
        static_cast<std::mt19937_64::result_type>(
            std::chrono::steady_clock::now().time_since_epoch().count()) ^
        std::random_device{}());
    const auto temporaryDirectory = std::filesystem::temp_directory_path();
    for (std::size_t attempt = 0; attempt < 32; ++attempt) {
      auto directory =
          temporaryDirectory / ("cudaq-qdmi-" + std::to_string(generator()));
      std::error_code error;
      if (std::filesystem::create_directory(directory, error))
        return TemporaryFile(std::move(directory), filename);
      if (error && error != std::errc::file_exists)
        throw std::filesystem::filesystem_error(
            "Could not create a QDMI temporary directory", directory, error);
    }
    throw std::runtime_error("Could not create a QDMI temporary directory.");
  }

  TemporaryFile(const TemporaryFile &) = delete;
  TemporaryFile &operator=(const TemporaryFile &) = delete;
  TemporaryFile(TemporaryFile &&other) noexcept
      : directory_(std::exchange(other.directory_, {})),
        path_(std::exchange(other.path_, {})) {}
  TemporaryFile &operator=(TemporaryFile &&other) noexcept {
    if (this == &other)
      return *this;
    remove();
    directory_ = std::exchange(other.directory_, {});
    path_ = std::exchange(other.path_, {});
    return *this;
  }
  ~TemporaryFile() { remove(); }

  [[nodiscard]] const std::filesystem::path &path() const { return path_; }

private:
  TemporaryFile(std::filesystem::path directory,
                const std::string_view filename)
      : directory_(std::move(directory)), path_(directory_ / filename) {}

  void remove() noexcept {
    std::error_code error;
    if (!path_.empty())
      std::filesystem::remove(path_, error);
    error.clear();
    if (!directory_.empty())
      std::filesystem::remove(directory_, error);
  }

  std::filesystem::path directory_;
  std::filesystem::path path_;
};

/*
 * CUDA-Q's qubit-mapping pass currently reads a connectivity file. Use a
 * private temporary directory so another process cannot replace the file
 * between path selection and file creation.
 */
std::optional<TemporaryFile> writeConnectivity(
    const std::optional<std::vector<std::pair<std::size_t, std::size_t>>>
        &connectivity,
    const std::size_t qubitCount) {
  if (!connectivity)
    return std::nullopt;

  auto file = TemporaryFile::create("connectivity.txt");
  std::ofstream output(file.path(), std::ios::out | std::ios::trunc);
  if (!output)
    throw std::runtime_error("Could not create QDMI connectivity file '" +
                             file.path().string() + "'.");

  std::map<std::size_t, std::set<std::size_t>> adjacency;
  for (const auto &[source, target] : *connectivity) {
    adjacency[source].emplace(target);
    adjacency[target].emplace(source);
  }
  output << "Number of nodes: " << qubitCount << '\n';
  for (const auto &[qubit, neighbors] : adjacency) {
    output << qubit << " --> {";
    for (std::size_t index = 0; const auto neighbor : neighbors)
      output << (index++ == 0 ? "" : ", ") << neighbor;
    output << "}\n";
  }
  output.flush();
  if (!output)
    throw std::runtime_error("Could not write QDMI connectivity file.");
  return file;
}

std::string serializeConnectivity(const std::optional<TemporaryFile> &file) {
  return file ? "file(" + file->path().string() + ")" : "bypass";
}

std::optional<std::string> makeIQMQubitMapping(const qdmi::Device &device) {
  const auto sites = device.getRegularSites();
  std::string mapping;
  for (std::size_t logical = 0; const auto &site : sites) {
    const auto name = site.getName();
    if (!name || name->empty())
      return std::nullopt;
    if (name->find_first_of(":,") != std::string::npos)
      throw std::runtime_error(
          "QDMI site names used for IQM JSON may not contain ':' or ','.");
    if (!mapping.empty())
      mapping += ',';
    mapping += "QB" + std::to_string(++logical) + ':' + *name;
  }
  return mapping.empty() ? std::nullopt
                         : std::optional<std::string>{std::move(mapping)};
}

std::string decodeQirBitcode(const std::string &encodedBitcode) {
  try {
    return cudaq::detail::decodeBase64(encodedBitcode);
  } catch (const std::exception &error) {
    throw std::runtime_error("Could not decode CUDA-Q QIR bitcode: " +
                             std::string(error.what()));
  }
}

std::string qirBitcodeToText(const std::string &encodedBitcode) {
  const auto bitcode = decodeQirBitcode(encodedBitcode);
  llvm::LLVMContext context;
  const auto buffer = llvm::MemoryBuffer::getMemBufferCopy(
      llvm::StringRef(bitcode.data(), bitcode.size()));
  auto module = llvm::parseBitcodeFile(buffer->getMemBufferRef(), context);
  if (!module)
    throw std::runtime_error("Could not parse CUDA-Q QIR bitcode: " +
                             llvm::toString(module.takeError()));

  std::string text;
  llvm::raw_string_ostream output(text);
  (*module)->print(output, nullptr);
  output.flush();
  return text;
}

std::string materializeTextProgram(const std::string &code,
                                   const ProgramFormat &format) {
  return format.encoding == ProgramEncoding::qirText ? qirBitcodeToText(code)
                                                     : code;
}

void dumpProgram(const std::span<const std::byte> program) {
  const auto *path = std::getenv("CUDAQ_QDMI_DUMP_PROGRAM");
  if (!path)
    return;
  std::ofstream output(path, std::ios::binary);
  output.write(reinterpret_cast<const char *>(program.data()),
               static_cast<std::streamsize>(program.size()));
  if (!output)
    throw std::runtime_error("Could not dump the QDMI program.");
}

void dumpProgram(const std::string &program) {
  dumpProgram(std::as_bytes(std::span(program)));
}

cudaq::CountsDictionary
toCountsDictionary(const std::map<std::string, std::size_t> &counts) {
  cudaq::CountsDictionary result;
  result.reserve(counts.size());
  for (const auto &[bits, count] : counts)
    result[bits] = count;
  return result;
}

std::map<std::string, std::size_t>
countsFromShots(const std::vector<std::string> &shots) {
  std::map<std::string, std::size_t> counts;
  for (const auto &shot : shots)
    ++counts[shot];
  return counts;
}

std::vector<std::size_t>
getMeasuredQubits(const cudaq::cudaq_json &outputNames) {
  const auto &json = outputNames.get();
  if (json.is_null() || json.empty())
    return {};
  if (!json.is_array() || json.size() != 1 || !json.front().is_array())
    throw std::runtime_error("Malformed CUDA-Q output_names metadata.");

  std::vector<std::pair<std::size_t, std::size_t>> indexedQubits;
  indexedQubits.reserve(json.front().size());
  for (const auto &entry : json.front()) {
    if (!entry.is_array() || entry.size() != 2 || !entry[1].is_array() ||
        entry[1].empty())
      throw std::runtime_error("Malformed CUDA-Q output_names entry.");
    indexedQubits.emplace_back(entry[0].get<std::size_t>(),
                               entry[1][0].get<std::size_t>());
  }
  std::ranges::sort(indexedQubits);

  std::vector<std::size_t> qubits;
  qubits.reserve(indexedQubits.size());
  std::ranges::transform(indexedQubits, std::back_inserter(qubits),
                         [](const auto &entry) { return entry.second; });
  return qubits;
}

std::string projectResult(const std::string &bits,
                          const std::vector<std::size_t> &measuredQubits) {
  if (measuredQubits.empty() || measuredQubits.size() == bits.size())
    return bits;

  std::vector<std::size_t> positions;
  positions.reserve(measuredQubits.size());
  std::ranges::transform(
      measuredQubits, std::back_inserter(positions), [&](const auto qubit) {
        if (qubit >= bits.size())
          throw std::runtime_error(
              "CUDA-Q output_names references a qubit outside the QDMI "
              "result.");
        return bits.size() - qubit - 1;
      });
  std::ranges::sort(positions);

  std::string projected;
  projected.reserve(positions.size());
  std::ranges::transform(positions, std::back_inserter(projected),
                         [&](const auto position) { return bits[position]; });
  return projected;
}

void projectResults(std::map<std::string, std::size_t> &counts,
                    std::optional<std::vector<std::string>> &shots,
                    const std::vector<std::size_t> &measuredQubits) {
  if (measuredQubits.empty())
    return;

  std::map<std::string, std::size_t> projectedCounts;
  for (const auto &[bits, count] : counts)
    projectedCounts[projectResult(bits, measuredQubits)] += count;
  counts = std::move(projectedCounts);
  if (shots)
    std::ranges::transform(*shots, shots->begin(), [&](const auto &bits) {
      return projectResult(bits, measuredQubits);
    });
}

cudaq::observe_result
observeResultFromCounts(const cudaq::observe_policy &policy,
                        cudaq::sample_result data) {
  double sum = 0.0;
  for (const auto &term : policy.spin) {
    if (term.is_identity())
      sum += term.evaluate_coefficient().real();
    else
      sum += data.expectation(term.get_term_id()) *
             term.evaluate_coefficient().real();
  }
  return cudaq::observe_result(sum, policy.spin, std::move(data));
}

template <typename ShotType>
std::size_t resolveShots(const std::optional<int> configuredShots,
                         const ShotType policyShots) {
  if (policyShots > 0)
    return policyShots;
  if (configuredShots && *configuredShots > 0)
    return static_cast<std::size_t>(*configuredShots);
  return 1000;
}

std::vector<cudaq::KernelExecution>
runCodegen(const cudaq::CompiledModule &module,
           const cudaq::CompileTarget &target) {
  if (module.getMlirArtifacts().empty())
    throw std::runtime_error("QDMI does not support launching a CompiledModule "
                             "without MLIR artifacts.");
  cudaq_internal::compiler::Compiler compiler(target, {});
  return compiler.emitKernelExecutions(module);
}

} // namespace

namespace cudaq {

class QDMIState {
public:
  QDMIState(qdmi::Device device, ProgramFormat format, JobParameters parameters,
            std::string deviceId, std::string basis,
            std::optional<std::vector<std::pair<std::size_t, std::size_t>>>
                connectivity,
            std::optional<TemporaryFile> connectivityFile)
      : device(std::move(device)), format(format),
        jobParameters(std::move(parameters)), deviceId(std::move(deviceId)),
        basis(std::move(basis)), connectivity(std::move(connectivity)),
        connectivityFile(std::move(connectivityFile)) {}

  qdmi::Device device;
  ProgramFormat format;
  JobParameters jobParameters;
  std::string deviceId;
  std::string basis;
  std::optional<std::vector<std::pair<std::size_t, std::size_t>>> connectivity;
  std::optional<TemporaryFile> connectivityFile;
};

namespace {

sample_result normalizeJobResult(qdmi::Job &job, const KernelExecution &code,
                                 const detail::ExecutionContextType execType) {
  static_cast<void>(job.wait());
  const auto status = job.check();
  if (status != QDMI_JOB_STATUS_DONE) {
    const auto id = job.getId();
    if (status == QDMI_JOB_STATUS_FAILED)
      throw std::runtime_error("QDMI job '" + id + "' failed.");
    if (status == QDMI_JOB_STATUS_CANCELED)
      throw std::runtime_error("QDMI job '" + id + "' was canceled.");
    throw std::runtime_error("QDMI job '" + id +
                             "' stopped in a non-terminal state.");
  }

  std::map<std::string, std::size_t> counts;
  std::optional<std::vector<std::string>> shots;
  std::string countsError;
  std::string shotsError;
  try {
    counts = job.getCounts();
  } catch (const std::exception &error) {
    countsError = error.what();
  }
  try {
    shots = job.getShots();
  } catch (const std::exception &error) {
    shotsError = error.what();
  }
  if (counts.empty()) {
    if (!shots)
      throw std::runtime_error(
          "QDMI device returned neither histogram nor shot results. "
          "Histogram error: " +
          countsError + "; shot error: " + shotsError);
    counts = countsFromShots(*shots);
  }

  projectResults(counts, shots, getMeasuredQubits(code.output_names));
  const auto registerName = execType == detail::ExecutionContextType::observe
                                ? code.name
                                : std::string(GlobalRegisterName);
  ExecutionResult executionResult(toCountsDictionary(counts), registerName);
  if (shots)
    executionResult.sequentialData = std::move(*shots);

  sample_result result(std::move(executionResult));
  if (!code.mapping_reorder_idx.empty())
    result.reorder(code.mapping_reorder_idx, registerName);
  return result;
}

qdmi::Job submitJob(const QDMIState &state, const KernelExecution &code,
                    const std::size_t shots) {
  if (state.format.encoding == ProgramEncoding::qirModule) {
    const auto bitcode = decodeQirBitcode(code.code);
    const auto bytes = std::as_bytes(std::span(bitcode));
    dumpProgram(bytes);
    return state.device.submitJob(
        bytes, state.format.qdmi, shots, state.jobParameters[0],
        state.jobParameters[1], state.jobParameters[2], state.jobParameters[3],
        state.jobParameters[4]);
  }

  const auto program = materializeTextProgram(code.code, state.format);
  dumpProgram(program);
  return state.device.submitJob(program, state.format.qdmi, shots,
                                state.jobParameters[0], state.jobParameters[1],
                                state.jobParameters[2], state.jobParameters[3],
                                state.jobParameters[4]);
}

std::vector<qdmi::Job> submitAllJobs(const QDMIState &state,
                                     const std::vector<KernelExecution> &codes,
                                     const std::size_t shots) {
  std::vector<qdmi::Job> jobs;
  jobs.reserve(codes.size());
  for (const auto &code : codes)
    jobs.emplace_back(submitJob(state, code, shots));
  return jobs;
}

sample_result executeJobs(const std::shared_ptr<QDMIState> &state,
                          const std::vector<KernelExecution> &codes,
                          const detail::ExecutionContextType execType,
                          const std::size_t shots) {
  if (!state)
    throw std::runtime_error("QDMI QPU is not configured.");
  if (execType == detail::ExecutionContextType::run)
    throw std::runtime_error("QDMI does not support cudaq::run.");

  auto jobs = submitAllJobs(*state, codes, shots);

  sample_result result;
  for (std::size_t index = 0; index < jobs.size(); ++index) {
    auto jobResult = normalizeJobResult(jobs[index], codes[index], execType);
    if (index != 0)
      result += jobResult;
    else
      result = std::move(jobResult);
  }
  return result;
}

sample_result collectJobs(std::vector<qdmi::Job> &jobs,
                          const std::vector<KernelExecution> &codes,
                          const detail::ExecutionContextType execType) {
  if (jobs.size() != codes.size())
    throw std::runtime_error("QDMI future metadata is inconsistent.");

  sample_result result;
  for (std::size_t index = 0; index < jobs.size(); ++index) {
    auto jobResult = normalizeJobResult(jobs[index], codes[index], execType);
    if (index != 0)
      result += jobResult;
    else
      result = std::move(jobResult);
  }
  return result;
}

detail::future submitJobsAsync(QDMIQPU &qpu,
                               const std::shared_ptr<QDMIState> &state,
                               std::vector<KernelExecution> codes,
                               const detail::ExecutionContextType execType,
                               const std::size_t shots) {
  if (!state)
    throw std::runtime_error("QDMI QPU is not configured.");

  struct PendingJobs {
    std::shared_ptr<QDMIState> state;
    std::vector<qdmi::Job> jobs;
    std::vector<KernelExecution> codes;
  };

  auto pending = std::make_shared<PendingJobs>();
  pending->state = state;
  pending->codes = std::move(codes);
  pending->jobs = submitAllJobs(*state, pending->codes, shots);

  std::vector<detail::future::Job> serializedJobs;
  std::map<std::string, std::string> serializedConfig{
      {"schema", "1"}, {"device", state->deviceId}};
  serializedJobs.reserve(pending->jobs.size());
  for (std::size_t index = 0; index < pending->jobs.size(); ++index) {
    const auto id = pending->jobs[index].getId();
    serializedJobs.emplace_back(id, pending->codes[index].name);
    serializedConfig["output_names." + id] =
        pending->codes[index].output_names->dump();
    serializedConfig["reorderIdx." + id] =
        nlohmann::json(pending->codes[index].mapping_reorder_idx).dump();
  }

  auto promise = std::make_shared<std::promise<sample_result>>();
  auto future = promise->get_future();
  QuantumTask task = [promise, pending, execType]() mutable {
    try {
      promise->set_value(collectJobs(pending->jobs, pending->codes, execType));
    } catch (...) {
      promise->set_exception(std::current_exception());
    }
  };
  qpu.enqueue(task);
  return detail::future(std::move(future), std::move(serializedJobs), "qdmi",
                        std::move(serializedConfig), execType);
}

class QDMIJobResultRetriever final : public detail::JobResultRetriever {
public:
  sample_result
  retrieve(const std::vector<Job> &jobs,
           const std::map<std::string, std::string> &config,
           const detail::ExecutionContextType resultType) override {
    if (getValue(config, "schema") != "1")
      throw std::runtime_error(
          "Persisted QDMI future has an unsupported metadata schema.");

    const auto persistedDevice = getValue(config, "device");
    if (!persistedDevice)
      throw std::runtime_error(
          "Persisted QDMI future is missing its stable device ID.");

    const auto *runtimeTarget = get_platform().get_runtime_target();
    if (!runtimeTarget || runtimeTarget->name != "qdmi")
      throw std::runtime_error(
          "Select the CUDA-Q QDMI target before reopening a QDMI future.");
    const auto activeDevice =
        getValue(runtimeTarget->runtimeConfig, "qdmi_device");
    if (!activeDevice || *activeDevice != *persistedDevice)
      throw std::runtime_error(
          "The active QDMI device does not match the persisted future.");

    auto device = qdmi::Session::openDevice(
        *persistedDevice, makeSessionConfig(runtimeTarget->runtimeConfig));
    std::vector<qdmi::Job> reopened;
    std::vector<KernelExecution> codes;
    reopened.reserve(jobs.size());
    codes.reserve(jobs.size());
    for (const auto &[id, name] : jobs) {
      const auto outputNames = config.find("output_names." + id);
      const auto reorder = config.find("reorderIdx." + id);
      if (outputNames == config.end() || reorder == config.end())
        throw std::runtime_error("Persisted QDMI job '" + id +
                                 "' is missing result metadata.");

      KernelExecution code;
      code.name = name;
      try {
        code.output_names =
            cudaq_json(nlohmann::json::parse(outputNames->second));
        code.mapping_reorder_idx = nlohmann::json::parse(reorder->second)
                                       .get<std::vector<std::size_t>>();
      } catch (const nlohmann::json::exception &error) {
        throw std::runtime_error(
            "Persisted QDMI job '" + id +
            "' has invalid result metadata: " + error.what());
      }
      reopened.emplace_back(device.retrieveJobById(id));
      codes.emplace_back(std::move(code));
    }
    return collectJobs(reopened, codes, resultType);
  }
};

} // namespace

QDMIQPU::QDMIQPU() : QPU() {}
QDMIQPU::~QDMIQPU() = default;

void QDMIQPU::enqueue(QuantumTask &task) { execution_queue->enqueue(task); }
void QDMIQPU::setShots(const int shots) { nShots = shots; }
void QDMIQPU::clearShots() { nShots.reset(); }

void QDMIQPU::setNoiseModel(const noise_model *model) {
  if (model)
    throw std::runtime_error(
        "Noise modeling is not supported by the QDMI backend.");
  noiseModel = nullptr;
}

void QDMIQPU::configureExecutionContext(ExecutionContext &context) const {
  if (context.executionManager)
    context.executionManager->configureExecutionContext(context);
}

void QDMIQPU::finalizeExecutionContext(ExecutionContext &context) const {
  if (context.executionManager)
    context.executionManager->finalizeExecutionContext(context);
}

void QDMIQPU::beginExecution() {
  if (auto *context = getExecutionContext();
      context && context->executionManager)
    context->executionManager->beginExecution();
}

void QDMIQPU::endExecution() {
  if (auto *context = getExecutionContext();
      context && context->executionManager)
    context->executionManager->endExecution();
}

void QDMIQPU::setTargetBackend(const std::string &) {
  const auto *runtimeTarget = get_platform().get_runtime_target();
  if (!runtimeTarget)
    throw std::runtime_error("QDMI target configuration is unavailable.");
  if (!runtimeTarget->config.BackendConfig)
    throw std::runtime_error("QDMI backend configuration is unavailable.");

  auto targetBackend = *runtimeTarget->config.BackendConfig;
  targetConfig = runtimeTarget->config;
  backendConfig = runtimeTarget->runtimeConfig;
  if (getValue(backendConfig, "emulate") == "true")
    throw std::runtime_error("QDMI does not support CUDA-Q emulation mode.");

  const auto deviceId = getValue(backendConfig, "qdmi_device");
  if (!deviceId)
    throw std::runtime_error("A stable QDMI device ID is required.");

  auto device =
      qdmi::Session::openDevice(*deviceId, makeSessionConfig(backendConfig));
  const auto format =
      selectProgramFormat(device.getSupportedProgramFormats(),
                          getValue(backendConfig, "qdmi_program_format"));
  const auto qubitCount = device.getQubitsNum();
  auto connectivity = queryConnectivity(device);
  auto connectivityFile = writeConnectivity(connectivity, qubitCount);
  auto parameters = getJobParameters(backendConfig);
  if (format.qdmi == QDMI_PROGRAM_FORMAT_IQMJSON && !parameters[4])
    if (auto mapping = makeIQMQubitMapping(device))
      parameters[4] = qdmi::CustomJobParameter{std::move(*mapping)};

  auto basis = queryBasis(device);
  if (!basis.excludedOperations.empty()) {
    std::string excluded;
    for (const auto &name : basis.excludedOperations) {
      if (!excluded.empty())
        excluded += ", ";
      excluded += name;
    }
    CUDAQ_DBG("QDMI operations not used for CUDA-Q basis conversion: {}.",
              excluded);
  }
  state = std::make_shared<QDMIState>(
      std::move(device), format, std::move(parameters), *deviceId,
      std::move(basis.value), std::move(connectivity),
      std::move(connectivityFile));
  numQubits = qubitCount;
  this->connectivity = state->connectivity;

  backendConfig["qdmi_basis"] = state->basis;
  backendConfig["qdmi_connectivity"] =
      serializeConnectivity(state->connectivityFile);
  targetBackend.CodegenEmission = std::string(format.codegen);
  if (format.qdmi == QDMI_PROGRAM_FORMAT_IQMJSON)
    targetBackend.JITMidLevelPipeline = std::string(iqmPipeline);
  targetConfig.BackendConfig = std::move(targetBackend);

  CUDAQ_INFO("Opened QDMI device '{}' ({} qubits) through '{}' transport.",
             state->device.getName(), qubitCount, format.name);
}

CompileTarget QDMIQPU::makeCompileTarget() const {
  if (!state)
    throw std::runtime_error("QDMI QPU is not configured.");
  CompileTarget target(targetConfig, backendConfig, /*emulate=*/false);
  target.supportConditionalsOnMeasureResults = false;
  target.pipelineConfig.replaceStateWithKernel = true;
  target.overrideAOTCompilation = true;
  if (state->format.qdmi != QDMI_PROGRAM_FORMAT_IQMJSON)
    target.pipelineConfig.postObservePasses =
        "basis-conversion{basis=" + state->basis + "}";
  return target;
}

CompileTarget QDMIQPU::getCompileTarget(const sample_policy &) {
  auto target = makeCompileTarget();
  target.pipelineConfig.addMeasurements = true;
  return target;
}

CompileTarget QDMIQPU::getCompileTarget(const observe_policy &policy) {
  auto target = makeCompileTarget();
  target.pauliTermSplitObservable = policy.spin;
  return target;
}

CompileTarget QDMIQPU::getCompileTarget(const other_policies &,
                                        ExecutionContext *) {
  throw std::runtime_error(
      "QDMI supports cudaq::sample() and cudaq::observe().");
}

sample_result QDMIQPU::launchKernel(const sample_policy &policy,
                                    const CompiledModule &module, KernelArgs) {
  const auto codes = runCodegen(module, getCompileTarget(policy));
  return executeJobs(state, codes, detail::ExecutionContextType::sample,
                     resolveShots(nShots, policy.options.shots));
}

async_sample_result QDMIQPU::launchKernel(const async_sample_policy &policy,
                                          const CompiledModule &module,
                                          KernelArgs) {
  auto codes = runCodegen(module, getCompileTarget(policy.inner));
  return async_sample_result(submitJobsAsync(
      *this, state, std::move(codes), detail::ExecutionContextType::sample,
      resolveShots(nShots, policy.inner.options.shots)));
}

observe_result QDMIQPU::launchKernel(const observe_policy &policy,
                                     const CompiledModule &module, KernelArgs) {
  const auto codes = runCodegen(module, getCompileTarget(policy));
  return observeResultFromCounts(
      policy, executeJobs(state, codes, detail::ExecutionContextType::observe,
                          resolveShots(nShots, policy.options.shots)));
}

async_observe_result QDMIQPU::launchKernel(const async_observe_policy &policy,
                                           const CompiledModule &module,
                                           KernelArgs) {
  auto codes = runCodegen(module, getCompileTarget(policy.inner));
  return async_observe_result(
      submitJobsAsync(*this, state, std::move(codes),
                      detail::ExecutionContextType::observe,
                      resolveShots(nShots, policy.inner.options.shots)),
      &policy.inner.spin);
}

} // namespace cudaq

CUDAQ_REGISTER_TYPE(cudaq::QPU, cudaq::QDMIQPU, qdmi)
static cudaq::detail::JobResultRetriever::RegistryType::Add<
    cudaq::QDMIJobResultRetriever>
    qdmiJobResultRetrieverRegistration("qdmi");
