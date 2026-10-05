/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// @file server_process.h
/// @brief Header-only fork/exec harness for two-process tests: spawn a server
///        binary, wait for its readiness line, read the `key=value` endpoint
///        description out of it, then shut it down and collect the summary
///        lines it prints on the way out.
///
/// The readiness handshake, rather than a sleep, is what makes a two-process
/// test deterministic: the server publishes its endpoint (an ephemeral port, a
/// rendezvous port, an RDMA QP -- whatever the transport uses) on `stdout`,
/// and the client only dials once that line has been read.  Both `stdout` and
/// `stderr` are folded into one pipe so a server that dies during bring-up
/// leaves its diagnostic in `output()` instead of the test seeing a bare
/// timeout.
///
/// Every complete line read from the child is also echoed to the test's own
/// `stdout` behind a `[server]` prefix, so `ctest -V` shows the server's side
/// of a run that passed rather than only of one that failed.  Nothing reads
/// the pipe except the waits for a specific line, so `stop()` drains what is
/// left before closing it -- without that, the lines a server prints while
/// shutting down would never be seen.
///
/// The READY line is parsed as free-form whitespace-separated `key=value`
/// tokens into `fields()`, so one harness serves any transport: the `udp`
/// provider publishes `transport=udp port=N`, cpu_roce publishes
/// `transport=cpu_roce port=N roce_ip=A` in rendezvous mode and
/// `qp=`/`rkey=`/`buffer_addr=` in hsb_fpga mode.  Nothing here knows which.
///
/// Usage:
/// \code
///   ServerProcess server;
///   ASSERT_TRUE(server.start({binary, "--transport=udp", "--port=0"},
///                            "CUDAQ_REALTIME_SERVER_READY"))
///       << server.output();
///   ... talk to server.port() ...
///   const std::string line = server.stopAndReadLine("..._PROCESSED", 5s);
/// \endcode

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fcntl.h>
#include <map>
#include <poll.h>
#include <signal.h>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace cudaq::realtime::testing {

class ServerProcess {
public:
  ServerProcess() = default;
  ServerProcess(const ServerProcess &) = delete;
  ServerProcess &operator=(const ServerProcess &) = delete;

  ~ServerProcess() {
    if (pid_ > 0) {
      ::kill(pid_, SIGKILL);
      int status = 0;
      ::waitpid(pid_, &status, 0);
      pid_ = -1;
    }
    closeFd();
  }

  /// Spawn `argv[0]` with `argv` and read its output until a line beginning
  /// with `ready_prefix` appears.  Returns false on spawn failure, on timeout,
  /// or if the child exits first (a rejected command line) -- in every case
  /// `output()` holds everything the child printed.
  bool start(const std::vector<std::string> &argv,
             const std::string &ready_prefix,
             std::chrono::milliseconds timeout = std::chrono::seconds(15)) {
    if (argv.empty())
      return false;

    int out_pipe[2] = {-1, -1};
    if (::pipe(out_pipe) != 0)
      return false;

    pid_ = ::fork();
    if (pid_ < 0) {
      ::close(out_pipe[0]);
      ::close(out_pipe[1]);
      return false;
    }
    if (pid_ == 0) {
      // Fold stderr into the same pipe: a bring-up failure prints there, and
      // the parent needs it to explain the missing READY line.
      ::dup2(out_pipe[1], STDOUT_FILENO);
      ::dup2(out_pipe[1], STDERR_FILENO);
      ::close(out_pipe[0]);
      ::close(out_pipe[1]);
      std::vector<std::string> args = argv;
      std::vector<char *> raw;
      raw.reserve(args.size() + 1);
      for (auto &a : args)
        raw.push_back(a.data());
      raw.push_back(nullptr);
      ::execv(args[0].c_str(), raw.data());
      std::perror(("execv " + args[0]).c_str());
      _exit(127);
    }

    ::close(out_pipe[1]);
    out_fd_ = out_pipe[0];
    // Non-blocking reads let the drain loop empty the pipe without an extra
    // poll() per byte and without ever blocking mid-line.
    ::fcntl(out_fd_, F_SETFL, ::fcntl(out_fd_, F_GETFL, 0) | O_NONBLOCK);

    std::string ready_line;
    if (!readLineWithPrefix(ready_prefix, timeout, ready_line))
      return false;
    parseFields(ready_line);
    return true;
  }

  /// Ask the server to shut down, then return the first line beginning with
  /// `prefix` that it prints on the way out (empty string if none appears).
  /// Servers whose counters are only final after their dispatch loop exits
  /// report them here rather than while running.
  std::string stopAndReadLine(const std::string &prefix,
                              std::chrono::milliseconds timeout) {
    if (pid_ <= 0)
      return {};
    ::kill(pid_, SIGTERM);
    std::string line;
    const bool found = readLineWithPrefix(prefix, timeout, line);
    reapBounded(std::chrono::milliseconds(1000));
    drain(); // anything printed after the line we waited for
    closeFd();
    return found ? line : std::string{};
  }

  /// SIGTERM, then SIGKILL if the child has not exited within a second.  A
  /// server that ignores SIGTERM must not hang the test until the `ctest`
  /// timeout.
  void stop() {
    if (pid_ > 0) {
      ::kill(pid_, SIGTERM);
      reapBounded(std::chrono::milliseconds(1000));
    }
    // After the reap: the child has closed its end, so this collects
    // everything it printed on the way out rather than racing it.
    drain();
    closeFd();
  }

  /// Reap a child that exited on its own and return its exit code (-1 when it
  /// died by signal, never ran, or had to be killed).  Used for the cases
  /// where the server is expected to reject its command line.
  int exitCode(std::chrono::milliseconds grace = std::chrono::seconds(5)) {
    if (pid_ <= 0)
      return -1;
    drain();
    const int code = reapBounded(grace);
    closeFd();
    return code;
  }

  /// `key=value` tokens parsed out of the READY line.
  const std::map<std::string, std::string> &fields() const { return fields_; }

  /// Convenience accessor for the near-universal `port=` field (0 when the
  /// transport does not publish one).
  std::uint16_t port() const {
    const auto it = fields_.find("port");
    if (it == fields_.end())
      return 0;
    try {
      return static_cast<std::uint16_t>(std::stoul(it->second));
    } catch (const std::exception &) {
      return 0;
    }
  }

  /// Everything read from the child so far, for failure messages.
  const std::string &output() const { return output_; }

private:
  bool readLineWithPrefix(const std::string &prefix,
                          std::chrono::milliseconds timeout,
                          std::string &line_out) {
    if (out_fd_ < 0)
      return false;
    // Deadline rather than a per-iteration tick count: charging a fixed cost
    // per poll() would time out a chatty server long before `timeout`.
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    for (;;) {
      const auto now = std::chrono::steady_clock::now();
      if (now >= deadline)
        return false;
      const auto remaining =
          std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now)
              .count();
      pollfd pfd{out_fd_, POLLIN, 0};
      const int ready =
          ::poll(&pfd, 1, static_cast<int>(remaining < 100 ? remaining : 100));
      if (ready < 0) {
        if (errno == EINTR)
          continue;
        return false;
      }
      if (ready == 0)
        continue;

      char c = 0;
      ssize_t n = 0;
      while ((n = ::read(out_fd_, &c, 1)) == 1) {
        output_.push_back(c);
        if (c != '\n') {
          partial_.push_back(c);
          continue;
        }
        const bool match = partial_.rfind(prefix, 0) == 0;
        echoLine(partial_);
        line_out = partial_;
        partial_.clear();
        if (match)
          return true;
      }
      if (n == 0)
        return false; // EOF: the child exited without printing the prefix
    }
  }

  // Read whatever is already buffered without waiting, so output() is complete
  // for a child that has already exited.  Split on newlines as we go so the
  // tail of the run is echoed a line at a time like the rest.
  void drain() {
    if (out_fd_ < 0)
      return;
    char buffer[512];
    ssize_t n = 0;
    while ((n = ::read(out_fd_, buffer, sizeof(buffer))) > 0) {
      output_.append(buffer, static_cast<std::size_t>(n));
      for (ssize_t i = 0; i < n; ++i) {
        if (buffer[i] != '\n') {
          partial_.push_back(buffer[i]);
          continue;
        }
        echoLine(partial_);
        partial_.clear();
      }
    }
    // A child killed mid-line still deserves to have that line shown.
    echoLine(partial_);
    partial_.clear();
  }

  // Echo one line of the child's output to the test's own stdout.  printf
  // rather than std::cout so it shares gtest's stream and buffering, which is
  // what keeps the interleaving with the test log in order.
  void echoLine(const std::string &line) const {
    if (line.empty())
      return;
    std::printf("[server] %s\n", line.c_str());
    std::fflush(stdout);
  }

  // Wait up to `grace` for a voluntary exit, then SIGKILL and wait.
  int reapBounded(std::chrono::milliseconds grace) {
    if (pid_ <= 0)
      return -1;
    const auto deadline = std::chrono::steady_clock::now() + grace;
    int status = 0;
    for (;;) {
      const pid_t done = ::waitpid(pid_, &status, WNOHANG);
      if (done == pid_)
        break;
      if (done < 0) {
        pid_ = -1;
        return -1;
      }
      if (std::chrono::steady_clock::now() >= deadline) {
        ::kill(pid_, SIGKILL);
        ::waitpid(pid_, &status, 0);
        pid_ = -1;
        return -1;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    pid_ = -1;
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
  }

  void parseFields(const std::string &ready_line) {
    std::istringstream tokens(ready_line);
    std::string token;
    while (tokens >> token) {
      const auto eq = token.find('=');
      if (eq == std::string::npos || eq == 0)
        continue;
      fields_[token.substr(0, eq)] = token.substr(eq + 1);
    }
  }

  void closeFd() {
    if (out_fd_ >= 0) {
      ::close(out_fd_);
      out_fd_ = -1;
    }
  }

  pid_t pid_ = -1;
  int out_fd_ = -1;
  std::string output_;  // everything read
  std::string partial_; // bytes of the line currently being assembled
  std::map<std::string, std::string> fields_;
};

} // namespace cudaq::realtime::testing
