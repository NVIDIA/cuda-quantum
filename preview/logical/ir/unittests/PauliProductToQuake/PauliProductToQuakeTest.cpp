/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

// Unit test for qlx/Conversion/PauliProductToQuake.h.
//
// The decompositions are built with the real builders into a func.func, the IR
// is verified (which enforces Quake's wire linearity), and then interpreted on
// a small state vector so the *semantics* are checked, not only the op
// sequence:
//   * rotateProduct implements exp(-i angle P / 2), for both styles;
//   * measureProduct projects onto the P = +/-1 eigenspace, leaves the other
//     qubits in the projected state (non-destructive), and reports the bit;
//   * negated products flip the rotation angle / complement the bit;
//   * the pattern templates instantiate and convert the P3 ops.

#include "qlx/Conversion/PauliProductToQuake.h"

#include "qlx/Dialect/Cflow/IR/CflowDialect.h"
#include "qlx/Dialect/Event/IR/EventDialect.h"
#include "qlx/Dialect/Fabric/IR/FabricDialect.h"
#include "qlx/Dialect/LVM/IR/LVMDialect.h"
#include "qlx/Dialect/Phys/IR/PhysDialect.h"
#include "qlx/Dialect/Phys/IR/PhysOps.h"
#include "qlx/Dialect/QLX/IR/QLXDialect.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include <cmath>
#include <complex>
#include <functional>

using namespace mlir;
using namespace qlx::pauli_quake;
namespace pq = qlx::pauli_quake;
namespace quake = cudaq::quake;
using Complex = std::complex<double>;
using State = std::vector<Complex>;

namespace {

int failures = 0;
#define CHECK(cond, ...)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      ++failures;                                                              \
      llvm::errs() << "FAIL " << __FILE__ << ":" << __LINE__ << ": " #cond     \
                   << " [" << __VA_ARGS__ << "]\n";                            \
    }                                                                          \
  } while (0)

//===----------------------------------------------------------------------===//
// A tiny state-vector simulator over the emitted Quake ops.
//===----------------------------------------------------------------------===//

void apply1(State &psi, unsigned q, Complex m00, Complex m01, Complex m10,
            Complex m11) {
  const std::size_t bit = std::size_t{1} << q;
  for (std::size_t i = 0; i < psi.size(); ++i) {
    if (i & bit)
      continue;
    Complex a = psi[i], b = psi[i | bit];
    psi[i] = m00 * a + m01 * b;
    psi[i | bit] = m10 * a + m11 * b;
  }
}

void applyH(State &psi, unsigned q) {
  double r = 1 / std::sqrt(2.0);
  apply1(psi, q, r, r, r, -r);
}
void applyRx(State &psi, unsigned q, double t) {
  Complex c = std::cos(t / 2), s = Complex(0, -std::sin(t / 2));
  apply1(psi, q, c, s, s, c);
}
void applyRz(State &psi, unsigned q, double t) {
  apply1(psi, q, std::polar(1.0, -t / 2), 0, 0, std::polar(1.0, t / 2));
}
void applyCx(State &psi, unsigned control, unsigned target) {
  const std::size_t c = std::size_t{1} << control, t = std::size_t{1} << target;
  for (std::size_t i = 0; i < psi.size(); ++i)
    if ((i & c) && !(i & t))
      std::swap(psi[i], psi[i | t]);
}
void projectZ(State &psi, unsigned q, int outcome) {
  const std::size_t bit = std::size_t{1} << q;
  for (std::size_t i = 0; i < psi.size(); ++i)
    if (((i & bit) != 0) != (outcome != 0))
      psi[i] = 0;
}

/// Apply the Pauli word (qubit k gets word[k]) to a copy of `psi`.
State applyWord(const State &psi, llvm::ArrayRef<Pauli> word) {
  State out = psi;
  for (auto [q, pauli] : llvm::enumerate(word)) {
    State next(out.size(), 0);
    const std::size_t bit = std::size_t{1} << q;
    for (std::size_t i = 0; i < out.size(); ++i) {
      switch (pauli) {
      case Pauli::I:
        next[i] += out[i];
        break;
      case Pauli::X:
        next[i ^ bit] += out[i];
        break;
      case Pauli::Y:
        if (i & bit)
          next[i & ~bit] += Complex(0, -1) * out[i];
        else
          next[i | bit] += Complex(0, 1) * out[i];
        break;
      case Pauli::Z:
        next[i] += ((i & bit) ? -1.0 : 1.0) * out[i];
        break;
      }
    }
    out = std::move(next);
  }
  return out;
}

struct Interpreter {
  Interpreter(unsigned numQubits, State psi, int forcedOutcome)
      : numQubits(numQubits), psi(std::move(psi)),
        forcedOutcome(forcedOutcome) {}

  unsigned numQubits;
  State psi;
  int forcedOutcome; // outcome every mz is projected onto
  DenseMap<Value, unsigned> qubit;
  DenseMap<Value, double> floats;
  DenseMap<Value, int> bits;
  bool ok = true;

  double scalar(Value v) {
    auto it = floats.find(v);
    if (it == floats.end()) {
      ok = false;
      return 0;
    }
    return it->second;
  }

  void run(Block &block) {
    for (Operation &op : block) {
      if (auto c = dyn_cast<arith::ConstantOp>(op)) {
        if (auto f = dyn_cast<FloatAttr>(c.getValue()))
          floats[c.getResult()] = f.getValueAsDouble();
        else if (auto i = dyn_cast<IntegerAttr>(c.getValue()))
          bits[c.getResult()] = i.getInt() != 0;
      } else if (auto neg = dyn_cast<arith::NegFOp>(op)) {
        floats[neg.getResult()] = -scalar(neg.getOperand());
      } else if (auto mul = dyn_cast<arith::MulFOp>(op)) {
        floats[mul.getResult()] = scalar(mul.getLhs()) * scalar(mul.getRhs());
      } else if (auto x = dyn_cast<arith::XOrIOp>(op)) {
        bits[x.getResult()] = bits[x.getLhs()] ^ bits[x.getRhs()];
      } else if (auto w = dyn_cast<quake::NullWireOp>(op)) {
        qubit[w.getResult()] = nextWire++;
      } else if (auto h = dyn_cast<quake::HOp>(op)) {
        unsigned q = qubit[h.getTargets()[0]];
        applyH(psi, q);
        qubit[h.getResult(0)] = q;
      } else if (auto rx = dyn_cast<quake::RxOp>(op)) {
        unsigned q = qubit[rx.getTargets()[0]];
        applyRx(psi, q, scalar(rx.getParameters()[0]));
        qubit[rx.getResult(0)] = q;
      } else if (auto rz = dyn_cast<quake::RzOp>(op)) {
        unsigned q = qubit[rz.getTargets()[0]];
        applyRz(psi, q, scalar(rz.getParameters()[0]));
        qubit[rz.getResult(0)] = q;
      } else if (auto cx = dyn_cast<quake::XOp>(op)) {
        unsigned c = qubit[cx.getControls()[0]], t = qubit[cx.getTargets()[0]];
        applyCx(psi, c, t);
        qubit[cx.getResult(0)] = c;
        qubit[cx.getResult(1)] = t;
      } else if (auto ep = dyn_cast<quake::ExpPauliOp>(op)) {
        // exp(i theta P) = cos(theta) I + i sin(theta) P
        double theta = scalar(ep.getParameters()[0]);
        std::vector<Pauli> word;
        for (char ch : ep.getPauliLiteral()->str())
          word.push_back(*parsePauli(llvm::StringRef(&ch, 1)));
        // The word indexes the target list; apply each letter to the qubit
        // that target currently names.
        State cur = psi;
        for (auto [i, pauli] : llvm::enumerate(word)) {
          std::vector<Pauli> single(numQubits, Pauli::I);
          single[qubit[ep.getTargets()[i]]] = pauli;
          cur = applyWord(cur, single);
        }
        State rotated(psi.size());
        for (std::size_t i = 0; i < psi.size(); ++i)
          rotated[i] =
              std::cos(theta) * psi[i] + Complex(0, std::sin(theta)) * cur[i];
        psi = rotated;
        for (auto [i, r] : llvm::enumerate(ep.getResults()))
          qubit[r] = qubit[ep.getTargets()[i]];
      } else if (auto mz = dyn_cast<quake::MzOp>(op)) {
        unsigned q = qubit[mz.getTargets()[0]];
        projectZ(psi, q, forcedOutcome);
        qubit[mz.getWires()[0]] = q;
        bits[mz.getMeasOut()] = forcedOutcome;
      } else if (auto d = dyn_cast<quake::DiscriminateOp>(op)) {
        bits[d.getResult()] = bits[d.getMeasurement()];
      } else if (isa<quake::SinkOp, func::ReturnOp>(op)) {
        // nothing to simulate
      } else {
        llvm::errs() << "unexpected op in decomposition: " << op.getName()
                     << "\n";
        ok = false;
      }
    }
  }

  unsigned nextWire = 0;
};

State randomState(unsigned n, unsigned seed) {
  State psi(std::size_t{1} << n);
  unsigned x = seed * 2654435761u + 12345u;
  double norm = 0;
  for (Complex &a : psi) {
    x = x * 1664525u + 1013904223u;
    double re = (x >> 8) / double(1 << 24) - 0.5;
    x = x * 1664525u + 1013904223u;
    double im = (x >> 8) / double(1 << 24) - 0.5;
    a = Complex(re, im);
    norm += std::norm(a);
  }
  for (Complex &a : psi)
    a /= std::sqrt(norm);
  return psi;
}

double distance(const State &a, const State &b) {
  double d = 0;
  for (std::size_t i = 0; i < a.size(); ++i)
    d = std::max(d, std::abs(a[i] - b[i]));
  return d;
}

/// Build a one-function module with `n` fresh wires, let `build` emit the
/// decomposition (returning the final wires and an optional `i1` result), sink
/// the wires, verify, and interpret the function on `initial`.
struct Built {
  bool ok = false;
  State psi;
  int bit = -1;
};

Built buildAndRun(
    MLIRContext &context, unsigned n, const State &initial, int forcedOutcome,
    const std::function<LogicalResult(OpBuilder &, Location, ValueRange,
                                      SmallVectorImpl<Value> &, Value &)>
        &build) {
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  auto module = ModuleOp::create(loc);
  builder.setInsertionPointToStart(module.getBody());
  auto fn =
      func::FuncOp::create(builder, loc, "t", builder.getFunctionType({}, {}));
  Block *entry = fn.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  SmallVector<Value> wires;
  for (unsigned i = 0; i < n; ++i)
    wires.push_back(quake::NullWireOp::create(builder, loc,
                                              quake::WireType::get(&context)));

  SmallVector<Value> out;
  Value bit;
  Built result;
  if (failed(build(builder, loc, wires, out, bit))) {
    module.erase();
    return result;
  }
  for (Value w : out)
    quake::SinkOp::create(builder, loc, w);
  func::ReturnOp::create(builder, loc);

  if (failed(verify(module))) {
    llvm::errs() << "verification failed:\n";
    module.print(llvm::errs());
    module.erase();
    return result;
  }

  Interpreter interp(n, initial, forcedOutcome);
  interp.run(*entry);
  result.ok = interp.ok;
  result.psi = interp.psi;
  if (bit)
    result.bit = interp.bits[bit];
  module.erase();
  return result;
}

std::string name(llvm::ArrayRef<Pauli> word) {
  std::string s;
  for (Pauli p : word)
    s += "IXYZ"[static_cast<int>(p)];
  return s;
}

const std::vector<std::vector<Pauli>> &words() {
  using P = Pauli;
  static const std::vector<std::vector<Pauli>> w = {
      {P::Z},
      {P::X},
      {P::Y},
      {P::Z, P::Z},
      {P::X, P::Z},
      {P::Y, P::Z},
      {P::X, P::Y, P::Z},
      {P::Z, P::I, P::X},
      {P::I, P::Y, P::I, P::Y},
  };
  return w;
}

//===----------------------------------------------------------------------===//
// Tests
//===----------------------------------------------------------------------===//

void testParsing() {
  auto neg = parsePauliProduct("-XZ");
  CHECK(succeeded(neg) && neg->negated && neg->paulis.size() == 2 &&
            neg->paulis[0] == Pauli::X && neg->paulis[1] == Pauli::Z,
        "-XZ");
  auto pos = parsePauliProduct("+ZZ");
  CHECK(succeeded(pos) && !pos->negated && pos->paulis.size() == 2, "+ZZ");
  CHECK(failed(parsePauliProduct("XQ")), "bad letter rejected");
  CHECK(failed(parsePauli("")), "empty rejected");
}

/// rotateProduct implements exp(-i angle P / 2), for every word, both styles,
/// including identity padding.
void testRotation(MLIRContext &context, RotationStyle style, double angle) {
  for (const auto &word : words()) {
    unsigned n = word.size();
    State psi = randomState(n, n * 7 + (word[0] == Pauli::Y ? 1 : 0));
    State pPsi = applyWord(psi, word);
    State expected(psi.size());
    for (std::size_t i = 0; i < psi.size(); ++i)
      expected[i] = std::cos(angle / 2) * psi[i] -
                    Complex(0, std::sin(angle / 2)) * pPsi[i];

    Built built =
        buildAndRun(context, n, psi, 0,
                    [&](OpBuilder &b, Location loc, ValueRange wires,
                        SmallVectorImpl<Value> &out, Value &) {
                      Value a = pq::detail::f64(b, loc, angle);
                      auto r = rotateProduct(b, loc, wires, word, a, style);
                      if (failed(r))
                        return failure();
                      out.assign(r->begin(), r->end());
                      return success();
                    });
    CHECK(built.ok, name(word));
    CHECK(distance(built.psi, expected) < 1e-9,
          "rotation " << name(word) << " style "
                      << (style == RotationStyle::ExpPauli ? "exp_pauli"
                                                           : "decomposed"));
  }
}

/// A negated product rotates by -angle about the unsigned product. This is the
/// sign rule the pattern template applies; check it at the builder level.
void testNegatedRotation(MLIRContext &context) {
  std::vector<Pauli> word{Pauli::X, Pauli::Z};
  State psi = randomState(2, 3);
  Built plus = buildAndRun(context, 2, psi, 0,
                           [&](OpBuilder &b, Location loc, ValueRange wires,
                               SmallVectorImpl<Value> &out, Value &) {
                             auto r =
                                 rotateProduct(b, loc, wires, word,
                                               pq::detail::f64(b, loc, -0.9));
                             out.assign(r->begin(), r->end());
                             return success();
                           });
  Built minus = buildAndRun(context, 2, psi, 0,
                            [&](OpBuilder &b, Location loc, ValueRange wires,
                                SmallVectorImpl<Value> &out, Value &) {
                              // R_{-P}(0.9) built as the pattern does: negate
                              // the angle.
                              Value a = arith::NegFOp::create(
                                  b, loc, pq::detail::f64(b, loc, 0.9));
                              auto r = rotateProduct(b, loc, wires, word, a);
                              out.assign(r->begin(), r->end());
                              return success();
                            });
  CHECK(plus.ok && minus.ok && distance(plus.psi, minus.psi) < 1e-9,
        "negated rotation");
}

/// measureProduct projects onto the P = (-1)^outcome eigenspace and keeps the
/// result in place; the reported bit is the outcome (complemented if inverted).
void testMeasurement(MLIRContext &context, bool invert) {
  for (const auto &word : words()) {
    unsigned n = word.size();
    bool hasSupport = false;
    for (Pauli p : word)
      hasSupport |= p != Pauli::I;
    if (!hasSupport)
      continue;
    State psi = randomState(n, n * 11 + 1);
    State pPsi = applyWord(psi, word);
    for (int outcome = 0; outcome < 2; ++outcome) {
      double sign = outcome == 0 ? 1.0 : -1.0;
      State expected(psi.size());
      for (std::size_t i = 0; i < psi.size(); ++i)
        expected[i] = 0.5 * (psi[i] + sign * pPsi[i]);

      Built built =
          buildAndRun(context, n, psi, outcome,
                      [&](OpBuilder &b, Location loc, ValueRange wires,
                          SmallVectorImpl<Value> &out, Value &bit) {
                        auto r = measureProduct(b, loc, wires, word, invert);
                        if (failed(r))
                          return failure();
                        out.assign(r->wires.begin(), r->wires.end());
                        bit = r->bit;
                        return success();
                      });
      CHECK(built.ok, name(word));
      CHECK(distance(built.psi, expected) < 1e-9,
            "projection " << name(word) << " outcome " << outcome);
      CHECK(built.bit == (outcome ^ int(invert)),
            "bit " << name(word) << " outcome " << outcome << " invert "
                   << invert << " got " << built.bit);
    }
  }
}

void testMeasurementErrors(MLIRContext &context) {
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  auto module = ModuleOp::create(loc);
  builder.setInsertionPointToStart(module.getBody());
  Value w0 =
      quake::NullWireOp::create(builder, loc, quake::WireType::get(&context));
  Value w1 =
      quake::NullWireOp::create(builder, loc, quake::WireType::get(&context));
  std::vector<Pauli> identity{Pauli::I, Pauli::I};
  CHECK(
      failed(measureProduct(builder, loc, ValueRange{w0, w1}, identity, false)),
      "all-identity measurement has nothing to measure");
  std::vector<Pauli> one{Pauli::Z};
  CHECK(failed(measureProduct(builder, loc, ValueRange{w0, w1}, one, false)),
        "arity mismatch");
  auto rot = rotateProduct(builder, loc, ValueRange{w0, w1}, identity,
                           pq::detail::f64(builder, loc, 0.5));
  CHECK(succeeded(rot) && (*rot)[0] == w0 && (*rot)[1] == w1,
        "all-identity rotation is the identity");
  module.erase();
}

//===----------------------------------------------------------------------===//
// The pattern templates, instantiated on the P3 ops
//===----------------------------------------------------------------------===//

struct PhysQubits {
  static FailureOr<SmallVector<Value>> wires(ValueRange operands) {
    SmallVector<Value> result(operands.begin(), operands.end());
    for (Value v : result)
      if (!isa<quake::WireType>(v.getType()))
        return failure();
    return result;
  }
};

struct PhysMeasureTraits : PhysQubits {
  static FailureOr<PauliProduct> product(qlx::phys::MeasureProductOp op) {
    auto paulis = parsePaulis(op.getPaulis());
    if (failed(paulis))
      return failure();
    return PauliProduct{static_cast<bool>(op.getInvert()), *paulis};
  }
  static FailureOr<SmallVector<Value>>
  qubits(qlx::phys::MeasureProductOp, qlx::phys::MeasureProductOp::Adaptor a,
         ConversionPatternRewriter &) {
    return wires(a.getInputs());
  }
  static StringAttr registerName(qlx::phys::MeasureProductOp op) {
    return op.getRecordIdAttr();
  }
  static void replace(qlx::phys::MeasureProductOp op,
                      ConversionPatternRewriter &rewriter, ValueRange wires,
                      Value bit) {
    SmallVector<Value> results(wires.begin(), wires.end());
    results.push_back(bit);
    rewriter.replaceOp(op, results);
  }
};

struct PhysRotateTraits : PhysQubits {
  static FailureOr<PauliProduct> product(qlx::phys::RotateProductOp op) {
    auto paulis = parsePaulis(op.getPaulis());
    if (failed(paulis))
      return failure();
    return PauliProduct{false, *paulis};
  }
  static FailureOr<SmallVector<Value>>
  qubits(qlx::phys::RotateProductOp, qlx::phys::RotateProductOp::Adaptor a,
         ConversionPatternRewriter &) {
    return wires(a.getInputs());
  }
  static Value angle(qlx::phys::RotateProductOp op,
                     ConversionPatternRewriter &rewriter) {
    return pq::detail::f64(rewriter, op.getLoc(),
                           op.getAngle().convertToDouble());
  }
  static void replace(qlx::phys::RotateProductOp op,
                      ConversionPatternRewriter &rewriter, ValueRange wires) {
    rewriter.replaceOp(op, wires);
  }
};

constexpr const char *kPhysModule = R"mlir(
module attributes {qlx.profiles = ["p3"]} {
  phys.instrument @mpp {
    kind = "measure_product", variadic, record_schema = "bit",
    preserves_inputs,
    process = "{\22kind\22:\22builtin\22,\22name\22:\22measure_product\22,\22parameters\22:{}}"
  }
  phys.machine @arch {
    phys.resource_class @qubits {
      count = 2 : i64, kind = "qubit", native_actions = [],
      native_instruments = [@mpp]
    }
  }
  phys.resource @q0 {
    capabilities = ["qlx.physical/native_pauli_product_rotation"],
    index = 0 : i64, kind = "qubit", resource_class = @qubits
  }
  phys.resource @q1 {
    capabilities = ["qlx.physical/native_pauli_product_rotation"],
    index = 1 : i64, kind = "qubit", resource_class = @qubits
  }
  phys.graph @mpp_graph on @arch : () -> !phys.record<@bit> {
    %0:2 = phys.acquire [@q0, @q1] {event_id = "a"}
      : !phys.state<@q0>, !phys.state<@q1>
    %1:3 = phys.measure_product %0#0, %0#1 {
      event_id = "m", instrument = @mpp, paulis = ["X", "Z"], record_id = "r"
    } : (!phys.state<@q0>, !phys.state<@q1>)
        -> (!phys.state<@q0>, !phys.state<@q1>, !phys.record<@bit>)
    phys.release %1#0, %1#1 {event_id = "x"} :
      !phys.state<@q0>, !phys.state<@q1>
    phys.return %1#2 : !phys.record<@bit>
  }
  phys.graph @negated_graph on @arch : () -> !phys.record<@bit> {
    %0:2 = phys.acquire [@q0, @q1] {event_id = "a2"}
      : !phys.state<@q0>, !phys.state<@q1>
    %1:3 = phys.measure_product %0#0, %0#1 {
      event_id = "m2", instrument = @mpp, invert, paulis = ["Z", "Z"],
      record_id = "r2"
    } : (!phys.state<@q0>, !phys.state<@q1>)
        -> (!phys.state<@q0>, !phys.state<@q1>, !phys.record<@bit>)
    phys.release %1#0, %1#1 {event_id = "x2"} :
      !phys.state<@q0>, !phys.state<@q1>
    phys.return %1#2 : !phys.record<@bit>
  }
  phys.graph @rpp on @arch : () -> () {
    %0:2 = phys.acquire [@q0, @q1] {event_id = "a3"}
      : !phys.state<@q0>, !phys.state<@q1>
    %1:2 = phys.rotate_product %0#0, %0#1 {
      angle = 1.250000e-01 : f64, event_id = "r3", paulis = ["Y", "Z"]
    } : (!phys.state<@q0>, !phys.state<@q1>) -> (!phys.state<@q0>, !phys.state<@q1>)
    phys.release %1#0, %1#1 {event_id = "x3"} :
      !phys.state<@q0>, !phys.state<@q1>
    phys.return
  }
}
)mlir";

template <typename OpTy>
unsigned count(ModuleOp module) {
  unsigned n = 0;
  module.walk([&](OpTy) { ++n; });
  return n;
}

void testPatternsOnPhysOps(MLIRContext &context) {
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kPhysModule, &context);
  CHECK(module, "fixture parses");
  if (!module)
    return;

  TypeConverter converter;
  converter.addConversion([](Type t) { return t; });
  converter.addConversion([&](qlx::phys::StateType) -> Type {
    return quake::WireType::get(&context);
  });
  converter.addConversion([&](qlx::phys::RecordType) -> Type {
    return IntegerType::get(&context, 1);
  });
  auto cast = [](OpBuilder &b, Type type, ValueRange inputs,
                 Location loc) -> Value {
    return UnrealizedConversionCastOp::create(b, loc, type, inputs)
        .getResult(0);
  };
  converter.addSourceMaterialization(cast);
  converter.addTargetMaterialization(cast);

  ConversionTarget target(context);
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  target
      .addIllegalOp<qlx::phys::MeasureProductOp, qlx::phys::RotateProductOp>();

  RewritePatternSet patterns(&context);
  patterns.add<
      MeasureProductToQuake<qlx::phys::MeasureProductOp, PhysMeasureTraits>,
      RotateProductToQuake<qlx::phys::RotateProductOp, PhysRotateTraits>>(
      converter, &context);
  CHECK(succeeded(applyPartialConversion(*module, target, std::move(patterns))),
        "partial conversion");

  CHECK(count<qlx::phys::MeasureProductOp>(*module) == 0, "no measure_product");
  CHECK(count<qlx::phys::RotateProductOp>(*module) == 0, "no rotate_product");
  CHECK(count<quake::MzOp>(*module) == 2, "two measurements");
  CHECK(count<quake::RzOp>(*module) == 1, "the one rotation emits one Rz");
  CHECK(count<quake::DiscriminateOp>(*module) == 2, "two bits");
  CHECK(count<arith::XOrIOp>(*module) == 1,
        "only the inverted one complements");
  // The converted module is deliberately not verified: `phys.graph` rejects
  // the boundary `builtin.unrealized_conversion_cast`s between `!phys.state`
  // and `!quake.wire` ("physical linear type crosses unsupported regionless
  // operation"). Pointwise conversion of just these ops therefore cannot yield
  // a valid module inside a graph; that needs the surrounding ops converted
  // too, which is the full Pn-to-Quake conversion this test does not attempt.
}

} // namespace

int main() {
  DialectRegistry registry;
  registry
      .insert<qlx::QLXDialect, qlx::lvm::LVMDialect, qlx::phys::PhysDialect,
              qlx::cflow::CflowDialect, qlx::event::EventDialect,
              qlx::fabric::FabricDialect, func::FuncDialect,
              arith::ArithDialect, cudaq::cc::CCDialect, quake::QuakeDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  testParsing();
  for (double angle : {0.7, -1.3, 0.0}) {
    testRotation(context, RotationStyle::Decomposed, angle);
    testRotation(context, RotationStyle::ExpPauli, angle);
  }
  testNegatedRotation(context);
  testMeasurement(context, /*invert=*/false);
  testMeasurement(context, /*invert=*/true);
  testMeasurementErrors(context);
  testPatternsOnPhysOps(context);

  if (failures) {
    llvm::errs() << failures << " check(s) failed\n";
    return 1;
  }
  llvm::outs() << "all Pauli-product decomposition checks passed\n";
  return 0;
}
