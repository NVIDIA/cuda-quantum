/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "py_trace.h"
#include "cudaq/runtime/logger/chrome_tracer.h"
#include "cudaq/runtime/logger/spdlog_tracer.h"
#include "cudaq/runtime/logger/tracer.h"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>

#include <memory>
#include <string>
#include <utility>

namespace {

std::string formatKwargs(const nanobind::kwargs &kwargs) {
  if (kwargs.size() == 0)
    return {};
  std::string out = " (args = {";
  bool first = true;
  for (const auto &item : kwargs) {
    if (!first)
      out += ", ";
    first = false;
    out += nanobind::cast<std::string>(item.first);
    out += "=";
    out += nanobind::cast<std::string>(nanobind::str(item.second));
  }
  out += "})";
  return out;
}

class TraceSpan {
public:
  TraceSpan(std::string spanName, std::string spanArgs)
      : name(std::move(spanName)), args(std::move(spanArgs)) {}

  void enter() {
    handle = cudaq::Tracer::instance().beginSpan(cudaq::TraceContext{}, name,
                                                 /*tag=*/0, args, "python");
  }

  void exit() { cudaq::Tracer::instance().endSpan(std::move(handle)); }

private:
  std::string name;
  std::string args;
  cudaq::SpanHandle handle;
};

} // namespace

namespace cudaq {

void bindTrace(nanobind::module_ &mod) {
  auto trace = mod.def_submodule("trace");

  nanobind::class_<TraceSpan>(trace, "span")
      .def("__init__",
           [](TraceSpan *self, const std::string &name,
              nanobind::kwargs kwargs) {
             new (self) TraceSpan(name, formatKwargs(kwargs));
           })
      .def("__enter__", [](TraceSpan &self) { self.enter(); })
      .def(
          "__exit__",
          [](TraceSpan &self, nanobind::object, nanobind::object,
             nanobind::object) { self.exit(); },
          nanobind::arg("type").none(), nanobind::arg("value").none(),
          nanobind::arg("traceback").none());

  // Abstract base — used only for upcasting in set_backend / get_backend.
  nanobind::class_<TraceBackend>(trace, "TraceBackend");

  nanobind::class_<ChromeTraceBackend, TraceBackend>(trace, "ChromeBackend")
      .def(nanobind::new_([](std::string path) {
             return std::make_shared<ChromeTraceBackend>(std::move(path));
           }),
           nanobind::arg("path") = std::string{},
           "Construct a Chrome backend. If `path` is empty, the backend is "
           "pure in-memory and the destructor does not write a file; call "
           "to_json() / to_dict() / write_file() to retrieve events. If "
           "`path` is set, the destructor writes Chrome Trace Event Format "
           "JSON to `path`.")
      .def("to_json", &ChromeTraceBackend::toJson,
           "Return captured events as a Chrome Trace Event Format JSON "
           "string (same bytes the destructor would write to file).")
      .def(
          "to_dict",
          [](ChromeTraceBackend &self) {
            auto jsonMod = nanobind::module_::import_("json");
            return jsonMod.attr("loads")(self.toJson());
          },
          "Return captured events as a parsed Python dict.")
      .def("write_file", &ChromeTraceBackend::writeFile,
           nanobind::arg("path") = nanobind::none(),
           "Write the current buffer as JSON to `path`, or to the ctor path "
           "if omitted. Non-destructive: the buffer stays intact and the "
           "backend keeps capturing.")
      .def("clear", &ChromeTraceBackend::clear, "Drop all buffered events.");

  nanobind::class_<SpdlogTraceBackend, TraceBackend>(trace, "SpdlogBackend")
      .def(
          nanobind::new_([] { return std::make_shared<SpdlogTraceBackend>(); }),
          "Route trace events through spdlog. Output respects the current "
          "CUDAQ log level.");

  trace.def(
      "set_backend",
      [](std::shared_ptr<TraceBackend> backend) {
        auto &t = Tracer::instance();
        t.setBackend(std::move(backend));
        t.setCaptureEnabled(true);
      },
      nanobind::arg("backend").none(false),
      "Install a TraceBackend and enable span capture.");

  trace.def(
      "get_backend", [] { return Tracer::instance().getBackend(); },
      "Return the currently-installed TraceBackend, or None.");

  trace.def(
      "reset_backend",
      [] {
        auto &t = Tracer::instance();
        t.setBackend(nullptr);
        t.setCaptureEnabled(false);
      },
      "Remove any installed backend and disable span capture. Subsequent "
      "spans early-return without emitting.");
}

} // namespace cudaq
