/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
// ---------------------------------------------------------------------------
// quill_formatters.h — fmtquill::formatter and quill::Codec specializations
//
// Enables CUDAQ_SYNTH_LOG_TRACE("synth.grid", "{}", foo) for all synthesizer
// domain types.
//
// Strategy (Alternative C from design doc):
//   - Each type has a to_string() member that formats on the calling thread.
//   - fmtquill::formatter<T>::format() delegates to T::to_string().
//   - quill::Codec<T> inherits DirectFormatCodec<T>, which invokes the
//     formatter on the calling thread and enqueues only the flat string bytes.
//   - No GMP/MPFR heap pointers ever enter the async ring buffer.
//
// This header is only compiled when LOGGING_BACKEND_QUILL is defined and
// is included by log_macros.h automatically — callers do not include it
// directly.
// ---------------------------------------------------------------------------

#ifdef LOGGING_BACKEND_QUILL

#include "quill/DirectFormatCodec.h"
#include "quill/bundled/fmt/base.h"

// Domain types
#include "Math/Geometry/Ellipse.h"
#include "Math/Geometry/GridOp.h"
#include "Math/Geometry/Interval.h"
#include "Math/Geometry/Rectangle.h"
#include "Math/Geometry/ToUpright.h"
#include "cudaq/Synthesis/Math/Integer.h"
#include "cudaq/Synthesis/Math/Real.h"
#include "cudaq/Synthesis/Math/Ring/Domega.h"
#include "cudaq/Synthesis/Math/Ring/Dsqrt2.h"
#include "cudaq/Synthesis/Math/Ring/Zomega.h"
#include "cudaq/Synthesis/Math/Ring/Zsqrt2.h"
#include "cudaq/Synthesis/Math/Unitary.h"

// ---------------------------------------------------------------------------
// fmtquill::formatter specializations
//
// Each specialization formats the type by calling its to_string() method.
// These must live in the fmtquill namespace (same as the primary template).
// ---------------------------------------------------------------------------

template <>
struct fmtquill::formatter<cudaq::synth::Integer> {
  constexpr auto parse(format_parse_context &ctx) { return ctx.begin(); }
  auto format(const cudaq::synth::Integer &val, format_context &ctx) const {
    return fmtquill::format_to(ctx.out(), "{}", val.to_string());
  }
};

template <>
struct fmtquill::formatter<cudaq::synth::Real> {
  constexpr auto parse(format_parse_context &ctx) { return ctx.begin(); }
  auto format(const cudaq::synth::Real &val, format_context &ctx) const {
    return fmtquill::format_to(ctx.out(), "{}", val.to_string());
  }
};

template <>
struct fmtquill::formatter<cudaq::synth::ZSqrt2> {
  constexpr auto parse(format_parse_context &ctx) { return ctx.begin(); }
  auto format(const cudaq::synth::ZSqrt2 &val, format_context &ctx) const {
    return fmtquill::format_to(ctx.out(), "{}", val.to_string());
  }
};

template <>
struct fmtquill::formatter<cudaq::synth::ZOmega> {
  constexpr auto parse(format_parse_context &ctx) { return ctx.begin(); }
  auto format(const cudaq::synth::ZOmega &val, format_context &ctx) const {
    return fmtquill::format_to(ctx.out(), "{}", val.to_string());
  }
};

template <>
struct fmtquill::formatter<cudaq::synth::DSqrt2> {
  constexpr auto parse(format_parse_context &ctx) { return ctx.begin(); }
  auto format(const cudaq::synth::DSqrt2 &val, format_context &ctx) const {
    return fmtquill::format_to(ctx.out(), "{}", val.to_string());
  }
};

template <>
struct fmtquill::formatter<cudaq::synth::DOmega> {
  constexpr auto parse(format_parse_context &ctx) { return ctx.begin(); }
  auto format(const cudaq::synth::DOmega &val, format_context &ctx) const {
    return fmtquill::format_to(ctx.out(), "{}", val.to_string());
  }
};

template <>
struct fmtquill::formatter<cudaq::synth::Interval> {
  constexpr auto parse(format_parse_context &ctx) { return ctx.begin(); }
  auto format(const cudaq::synth::Interval &val, format_context &ctx) const {
    return fmtquill::format_to(ctx.out(), "{}", val.to_string());
  }
};

template <>
struct fmtquill::formatter<cudaq::synth::Rectangle> {
  constexpr auto parse(format_parse_context &ctx) { return ctx.begin(); }
  auto format(const cudaq::synth::Rectangle &val, format_context &ctx) const {
    return fmtquill::format_to(ctx.out(), "{}", val.to_string());
  }
};

template <>
struct fmtquill::formatter<cudaq::synth::Ellipse> {
  constexpr auto parse(format_parse_context &ctx) { return ctx.begin(); }
  auto format(const cudaq::synth::Ellipse &val, format_context &ctx) const {
    return fmtquill::format_to(ctx.out(), "{}", val.to_string());
  }
};

template <>
struct fmtquill::formatter<cudaq::synth::GridOp> {
  constexpr auto parse(format_parse_context &ctx) { return ctx.begin(); }
  auto format(const cudaq::synth::GridOp &val, format_context &ctx) const {
    return fmtquill::format_to(ctx.out(), "{}", val.to_string());
  }
};

template <>
struct fmtquill::formatter<cudaq::synth::UprightResult> {
  constexpr auto parse(format_parse_context &ctx) { return ctx.begin(); }
  auto format(const cudaq::synth::UprightResult &val,
              format_context &ctx) const {
    return fmtquill::format_to(ctx.out(), "{}", val.to_string());
  }
};

template <>
struct fmtquill::formatter<cudaq::synth::DOmegaUnitary> {
  constexpr auto parse(format_parse_context &ctx) { return ctx.begin(); }
  auto format(const cudaq::synth::DOmegaUnitary &val,
              format_context &ctx) const {
    return fmtquill::format_to(ctx.out(), "{}", val.to_string());
  }
};

// ---------------------------------------------------------------------------
// quill::Codec specializations
//
// Each type inherits DirectFormatCodec<T>, which invokes fmtquill::formatter
// on the calling thread and enqueues only the serialized string bytes into
// Quill's ring buffer. This ensures no GMP/MPFR heap data enters the queue.
// ---------------------------------------------------------------------------

template <>
struct quill::Codec<cudaq::synth::Integer>
    : quill::DirectFormatCodec<cudaq::synth::Integer> {};

template <>
struct quill::Codec<cudaq::synth::Real>
    : quill::DirectFormatCodec<cudaq::synth::Real> {};

template <>
struct quill::Codec<cudaq::synth::ZSqrt2>
    : quill::DirectFormatCodec<cudaq::synth::ZSqrt2> {};

template <>
struct quill::Codec<cudaq::synth::ZOmega>
    : quill::DirectFormatCodec<cudaq::synth::ZOmega> {};

template <>
struct quill::Codec<cudaq::synth::DSqrt2>
    : quill::DirectFormatCodec<cudaq::synth::DSqrt2> {};

template <>
struct quill::Codec<cudaq::synth::DOmega>
    : quill::DirectFormatCodec<cudaq::synth::DOmega> {};

template <>
struct quill::Codec<cudaq::synth::Interval>
    : quill::DirectFormatCodec<cudaq::synth::Interval> {};

template <>
struct quill::Codec<cudaq::synth::Rectangle>
    : quill::DirectFormatCodec<cudaq::synth::Rectangle> {};

template <>
struct quill::Codec<cudaq::synth::Ellipse>
    : quill::DirectFormatCodec<cudaq::synth::Ellipse> {};

template <>
struct quill::Codec<cudaq::synth::GridOp>
    : quill::DirectFormatCodec<cudaq::synth::GridOp> {};

template <>
struct quill::Codec<cudaq::synth::UprightResult>
    : quill::DirectFormatCodec<cudaq::synth::UprightResult> {};

template <>
struct quill::Codec<cudaq::synth::DOmegaUnitary>
    : quill::DirectFormatCodec<cudaq::synth::DOmegaUnitary> {};

#endif // LOGGING_BACKEND_QUILL
