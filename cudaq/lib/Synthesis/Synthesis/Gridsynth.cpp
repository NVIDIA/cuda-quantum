/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Synthesis/Synthesis/Gridsynth.h"

#include "Support/StreamOps.h"
#include "llvm/Support/Debug.h"

#include "Math/Diophantine.h"
#include "Math/Geometry/GridOp.h"
#include "Math/Geometry/Rectangle.h"
#include "Math/Geometry/ToUpright.h"
#include "Math/Geometry/UnitDisk.h"
#include "Math/Grid/Tdgp.h"
#include "cudaq/Synthesis/Math/Real.h"
#include "cudaq/Synthesis/Math/Unitary.h"
#include "cudaq/Synthesis/Synthesis/KmmSynthesize.h"

#include <algorithm>
#include <cstdint>
#include <optional>

#define DEBUG_TYPE "cudaq-synth"

using namespace cudaq::synth;

namespace {

//===----------------------------------------------------------------------===//
// EpsilonRegion
//===----------------------------------------------------------------------===//

/// The epsilon-region R_epsilon for the approximate z-rotation synthesis
/// problem (Ross & Selinger, arXiv:1403.2975, sec. 7.1, equation (14)).
///
/// For a target rotation R_z(theta) and precision epsilon > 0, R_epsilon is
/// the lens-shaped region
///
///   R_epsilon = { u in closed unit disk | dot(u, z) >= 1 - epsilon^2/2 }
///
/// where z = (cos(-theta/2), sin(-theta/2)) is the point on the unit circle
/// corresponding to the target rotation. The dot-product condition is
/// equivalent to |R_z(theta) - U| <= epsilon for single-qubit unitaries U
/// (equation (13)).
///
/// The class carries both the *exact* region definition (used by contains()
/// and intersect()) and a *bounding ellipse* that encloses R_epsilon. The
/// ellipse feeds the upright preprocessing in to_upright() (Theorem 5.16);
/// the exact region is what the TDGP filter ultimately checks.
class EpsilonRegion : public ConvexSet {
private:
  // Half-plane threshold d = 1 - epsilon^2/2; the membership test is
  // dot(u, z) >= d.
  Real dot_threshold;

  // Components of the direction vector z = (cos(-theta/2), sin(-theta/2)).
  Real z_x;
  Real z_y;

  // Ellipse enclosing R_epsilon (input to to_upright).
  Ellipse bounding_ellipse;

  EpsilonRegion(Real dot_threshold_, Real z_x_, Real z_y_,
                Ellipse bounding_ellipse_)
      : dot_threshold(std::move(dot_threshold_)), z_x(std::move(z_x_)),
        z_y(std::move(z_y_)), bounding_ellipse(std::move(bounding_ellipse_)) {}

  struct Precomputed {
    Real dot_threshold;
    Real z_x;
    Real z_y;
  };

  /// Precompute the three scalars that depend on (theta, epsilon):
  ///   dot_threshold = 1 - epsilon^2/2   (equation (13))
  ///   z             = (cos(-theta/2), sin(-theta/2))
  /// Uses mpfr_sin_cos to get both trig values from a single argument.
  static Precomputed compute_precomputed(const Real &theta,
                                         const Real &epsilon) {
    Precomputed pre;
    pre.dot_threshold = 1 - ((epsilon * epsilon) / 2);

    Real half_angle = -theta / 2;
    mpfr_sin_cos(pre.z_y.get_mpfr(), pre.z_x.get_mpfr(), half_angle.get_mpfr(),
                 MPFR_RNDN);
    return pre;
  }

  /// Build the ellipse that circumscribes R_epsilon.
  ///
  /// In the frame where z points along +x, R_epsilon is the circular segment
  /// cut off by the line x = d (with d = dot_threshold). It spans
  /// x in [d, 1] and |y| <= sqrt(1 - d^2). The ellipse centered at (d, 0)
  /// with semi-axes
  ///
  ///   along z:          1 - d              = epsilon^2/2
  ///   perpendicular:    sqrt(1 - d^2)      = epsilon * sqrt(1 - epsilon^2/4)
  ///
  /// contains that segment and touches it at (1, 0) and at (d, +/-sqrt(1-d^2)),
  /// so it is the tightest ellipse of this family. Writing x = d + t*(1-d) with
  /// t in [0, 1], membership reduces to t <= 1, which is exactly the segment's
  /// x-range. Hence the containment is exact rather than asymptotic.
  ///
  /// Ellipse stores { (u - p)^T D (u - p) <= 1 }, so we pass the center
  /// p = d*z and D = lambda_x * z z^T + lambda_y * z_perp z_perp^T with
  /// lambda = 1/semi-axis^2, rotated back into the original frame.
  ///
  /// Returns failure() if Ellipse::create rejects the parameters. Callers
  /// reach this only for 0 < epsilon < |1 - e^{i*pi/8}|, where d > 0.92 and
  /// the matrix is comfortably positive definite.
  static llvm::FailureOr<Ellipse>
  make_bounding_ellipse(const Real &z_x, const Real &z_y,
                        const Real &dot_threshold) {
    Real one_minus_d = 1 - dot_threshold;
    Real lambda_x = 1 / (one_minus_d * one_minus_d);
    Real lambda_y = 1 / (one_minus_d * (1 + dot_threshold));

    Real A = lambda_x * z_x * z_x + lambda_y * z_y * z_y;
    Real B = (lambda_x - lambda_y) * z_x * z_y;
    Real C = lambda_x * z_y * z_y + lambda_y * z_x * z_x;
    Real Dx = dot_threshold * z_x;
    Real Dy = dot_threshold * z_y;

    return Ellipse::create(A, B, C, Dx, Dy);
  }

public:
  /// Build the epsilon-region for target angle theta and precision epsilon.
  /// Returns failure() if the enclosing ellipse is degenerate (does not
  /// occur for epsilon > 0).
  static llvm::FailureOr<EpsilonRegion> create(const Real &theta,
                                               const Real &epsilon) {
    Precomputed pre = compute_precomputed(theta, epsilon);
    llvm::FailureOr<Ellipse> ell_or =
        make_bounding_ellipse(pre.z_x, pre.z_y, pre.dot_threshold);
    if (llvm::failed(ell_or))
      return llvm::failure();
    return EpsilonRegion(std::move(pre.dot_threshold), std::move(pre.z_x),
                         std::move(pre.z_y), std::move(*ell_or));
  }

  const Ellipse &ellipse() const { return bounding_ellipse; }

  /// Exact membership test for R_epsilon: u is inside iff u lies in the unit
  /// disk and dot(u, z) >= dot_threshold. The check uses exact DSqrt2
  /// arithmetic for the disk constraint; MPFR rounding on the dot-product
  /// side is absorbed by the cached widened bounds further downstream.
  bool contains(const DOmega &u) const override {
    if (!(DSqrt2::from_domega(u.conj() * u) <= DSqrt2{1}))
      return false;

    // Re(u) and Im(u) share the same sqrt(2)^k factor. coords_into computes
    // both from a single inv_scale so pow_sqrt2 runs once instead of twice
    // (as it would via u.real() + u.imag()).
    static PrecisionCachedReal sqrt2_over_2_cache;
    const Real &sqrt2_over_2 =
        sqrt2_over_2_cache.get([] { return Real::sqrt2() / 2; });
    Real inv_scale = 1 / u.scale();
    Real u_real, u_imag;
    coords_into(u, inv_scale, sqrt2_over_2, u_real, u_imag);

    Real cos_similarity = u_real * z_x + u_imag * z_y;
    return cos_similarity >= dot_threshold;
  }

  /// Intersect the ray u(t) = u0 + t*v with R_epsilon, returning the
  /// parameter interval [t_lo, t_hi] for which u(t) lies inside, or nullopt
  /// if the ray misses R_epsilon entirely.
  ///
  /// The implementation does this in two stages: first intersect with the
  /// unit disk (a quadratic in t), then intersect the resulting interval
  /// with the half-plane dot(u(t), z) >= d (a linear in t, sign-dependent).
  std::optional<std::pair<Real, Real>>
  intersect(const DOmega &u0, const DOmega &v) const override {
    static const Real tolerance(1e-30);
    using Roots = std::pair<Real, Real>;

    // Unit-disk intersection: |u(t)|^2 <= 1 reduces to a*t^2 + b*t + c <= 0.
    DOmega a = v.conj() * v;
    DOmega b = DOmega::from_int(2) * (u0.conj() * v);
    DOmega c = u0.conj() * u0 - DOmega::from_dsqrt2(DSqrt2{1});

    std::optional<Roots> quad_solution =
        solve_quadratic(a.real(), b.real(), c.real());
    if (!quad_solution)
      return std::nullopt;

    auto &&[t0, t1] = quad_solution.value();

    // Half-plane constraint: dot(u(t), z) >= d, i.e. (z . v) t >= d - (z . u0).
    Real z_dot_v = z_x * v.real() + z_y * v.imag();
    Real rhs = dot_threshold - (z_x * u0.real() + z_y * u0.imag());

    if (z_dot_v > tolerance) {
      // Positive slope: clip t from below by rhs / z_dot_v.
      Real t_min = std::max(t0, rhs / z_dot_v);
      if (t_min > t1)
        return std::nullopt;
      return std::make_pair(t_min, t1);
    }

    if (z_dot_v < -tolerance) {
      // Negative slope: the inequality flips, so clip t from above.
      Real t_max = std::min(t1, rhs / z_dot_v);
      if (t0 > t_max)
        return std::nullopt;
      return std::make_pair(t0, t_max);
    }

    // z . v ~= 0: the ray is (numerically) parallel to the boundary line.
    // The half-plane is then satisfied for every t (if rhs <= 0) or for no
    // t at all.
    if (rhs <= tolerance)
      return std::make_pair(t0, t1);
    return std::nullopt;
  }

  /// Compact human-readable dump of the region parameters. Intended for
  /// LLVM_DEBUG / CUDAQ_SYNTH_OPEN_SUB diagnostic streams; not used in
  /// hot paths.
  std::string to_string() const {
    char prefix[256];
    mpfr_snprintf(prefix, sizeof(prefix),
                  "z=(%.6Rf,%.6Rf),dot_threshold=%.25Rf,", z_x.get_mpfr(),
                  z_y.get_mpfr(), dot_threshold.get_mpfr());
    return std::string(prefix) + bounding_ellipse.to_string();
  }
};

//===----------------------------------------------------------------------===//
// Zero-T shortcut
//===----------------------------------------------------------------------===//

/// The nearest zero-T (Clifford) approximation of R_z(theta), or nullopt if
/// it is not within epsilon.
///
/// Every Clifford+T unitary with denominator exponent k = 0 has (z, w) equal
/// to (unit, 0) or (0, unit): |z|^2 + |w|^2 = 1 with both terms totally
/// non-negative in Z[sqrt(2)], whose smallest positive such element is
/// 2 - sqrt(2) > 1/2, forces one of them to vanish. The off-diagonal family
/// is at operator-norm distance >= 1 from any R_z, so the best zero-T
/// candidate is diagonal:
///
///   diag(omega^-m, omega^m) = R_z(m*pi/2)
///
/// and the error is minimized by m = round(theta / (pi/2)), giving
/// ||R_z(theta) - R_z(m*pi/2)|| = 2*|sin((theta - m*pi/2)/4)| <= 2*sin(pi/16).
///
/// Ross & Selinger Lemma 7.2 is the corresponding existence statement: a
/// zero-T solution always exists once epsilon >= |1 - e^{i*pi/8}|
/// = 2*sin(pi/16) ~= 0.39018. We test the candidate's actual error instead of
/// that threshold, so the shortcut also fires for the (T-optimal) tighter
/// cases where theta happens to sit near a multiple of pi/2.
std::optional<DOmegaUnitary> nearest_zero_t_unitary(const Real &theta,
                                                    const Real &epsilon) {
  Integer m = round_to_integer(theta / (Real::pi() / 2));

  // R_z has period 4*pi, so only m mod 8 matters for omega^m. Reducing before
  // narrowing as theta may be arbitrarily large.
  int32_t m_mod8 = static_cast<int32_t>(m % Integer(8));
  int32_t exponent = (-m_mod8) & 0b111;

  DOmegaUnitary candidate(mul_by_omega_power(DOmega::from_int(1), exponent),
                          DOmega::from_int(0), 0);
  if (rz_approximation_error(candidate, theta) <= epsilon)
    return candidate;
  return std::nullopt;
}

} // namespace

namespace cudaq::synth {

//===----------------------------------------------------------------------===//
// Search bound
//===----------------------------------------------------------------------===//

int64_t details::max_denominator_exponent(const Real &epsilon) {
  // Slack above the expected k, leaving generous room for the probabilistic
  // tail of the Diophantine step before the search is called off.
  constexpr int64_t kSlack = 30;

  // log2(1/epsilon) is read off the MPFR exponent rather than computed: for
  // epsilon = m * 2^e with 0.5 <= m < 1, floor(log2(epsilon)) is e - 1. That
  // stays exact for arbitrarily small epsilon, where converting to a double
  // first would flush to zero and hand back an unusable bound.
  const mpfr_exp_t e = mpfr_get_exp(epsilon.get_mpfr());
  const int64_t log2_inv_epsilon = 1 - static_cast<int64_t>(e);

  // A loose epsilon (>= 1) makes log2(1/epsilon) negative; the slack alone is
  // the floor there.
  return std::max<int64_t>(0, 2 * log2_inv_epsilon) + kSlack;
}

mpfr_prec_t details::required_precision(const Real &epsilon) {
  constexpr mpfr_prec_t kFloor = 64;
  constexpr mpfr_prec_t kGuardFactor = 4;

  // Same exponent-based log2(1/epsilon) as max_denominator_exponent, so an
  // epsilon below the double range yields a large precision rather than an
  // infinity that would be undefined to convert.
  const mpfr_exp_t e = mpfr_get_exp(epsilon.get_mpfr());
  const int64_t log2_inv_epsilon = 1 - static_cast<int64_t>(e);

  return std::max<mpfr_prec_t>(
      kFloor,
      kGuardFactor * static_cast<mpfr_prec_t>(log2_inv_epsilon) + kFloor);
}

//===----------------------------------------------------------------------===//
// gridsynth_unitary
//===----------------------------------------------------------------------===//

llvm::FailureOr<DOmegaUnitary> gridsynth_unitary(const Real &theta,
                                                 const Real &epsilon,
                                                 int32_t diophantine_timeout_ms,
                                                 int32_t factoring_timeout_ms) {
  CUDAQ_SYNTH_OPEN_SUB("gridsynth_unitary");
  LLVM_DEBUG(cudaq::synth::dbgs() << "theta=" << theta << "\n";
             cudaq::synth::dbgs() << "eps=" << epsilon << "\n";
             cudaq::synth::dbgs() << "diophantine_timeout="
                                  << diophantine_timeout_ms << "ms" << "\n";
             cudaq::synth::dbgs()
             << "factoring_timeout=" << factoring_timeout_ms << "ms" << "\n");

  // Reject NaN / infinity / non-positive inputs here.
  if (!theta.is_finite()) {
    CUDAQ_SYNTH_CLOSE_FAILURE("theta must be finite");
    return llvm::failure();
  }
  if (!epsilon.is_finite() || !(epsilon > 0)) {
    CUDAQ_SYNTH_CLOSE_FAILURE("epsilon must be finite and strictly positive");
    return llvm::failure();
  }

  // Loose tolerances admit a zero-T answer. Taking it here also keeps the
  // epsilon-region construction below confined to epsilon < 2*sin(pi/16).
  if (std::optional<DOmegaUnitary> clifford =
          nearest_zero_t_unitary(theta, epsilon)) {
    CUDAQ_SYNTH_CLOSE_SUCCESS("zero-T Clifford within epsilon");
    return *clifford;
  }

  // Step 0: build the epsilon-region and the closed-unit-disk constraint
  // applied to the sqrt(2)-conjugate (the latter is needed because
  // |conj(u) * u| <= 1 only if conj_sq2(u) also lies in the unit disk).
  llvm::FailureOr<EpsilonRegion> region_or =
      EpsilonRegion::create(theta, epsilon);
  if (llvm::failed(region_or)) {
    CUDAQ_SYNTH_CLOSE_FAILURE("degenerate epsilon-region");
    return llvm::failure();
  }
  LLVM_DEBUG(cudaq::synth::dbgs()
             << "epsilon-region: " << region_or->to_string() << '\n');

  UnitDisk unit_disk;

  // Step 0b: upright preprocessing (Theorem 5.16). to_upright() finds a
  // grid operator G such that G(R_epsilon) and conj_sq2(G)(closed unit
  // disk) are both 1/6-upright; the resulting bounding boxes drive the
  // efficient grid-point enumeration in Lemma 5.8.
  llvm::FailureOr<UprightResult> transformed_or =
      to_upright(region_or->ellipse(), UnitDisk::as_ellipse());
  if (llvm::failed(transformed_or)) {
    CUDAQ_SYNTH_CLOSE_FAILURE("to_upright preprocessing failed");
    return llvm::failure();
  }
  UprightResult &transformed = *transformed_or;

  // Fattened y-intervals absorb the floating-point edge effects that can
  // otherwise reject valid grid points sitting exactly on the boundary.
  // The 1e-4 relative pad is small enough not to admit spurious candidates
  // (the TDGP filter rechecks membership exactly).
  Real epsilon_factor = Real(1e-4);
  Interval bboxA_y_fattened =
      fatten(transformed.bboxA.I_y(),
             transformed.bboxA.I_y().width() * epsilon_factor);
  Interval bboxB_y_fattened =
      fatten(transformed.bboxB.I_y(),
             transformed.bboxB.I_y().width() * epsilon_factor);

  // Log the bounding-box widths once, outside the k-loop, so each iteration's
  // log block stays focused on its own per-k data.
  LLVM_DEBUG(cudaq::synth::dbgs()
             << "bboxA=" << transformed.bboxA.I_x().width() << " x "
             << transformed.bboxA.I_y().width()
             << ", bboxB=" << transformed.bboxB.I_x().width() << " x "
             << transformed.bboxB.I_y().width() << '\n');
  LLVM_DEBUG(cudaq::synth::dbgs()
             << "bboxA_y_fat=" << bboxA_y_fattened.width()
             << ", bboxB_y_fat=" << bboxB_y_fattened.width() << '\n');

  llvm::FailureOr<GridOp> opG_inv_or = inv(transformed.opG);
  if (llvm::failed(opG_inv_or)) {
    CUDAQ_SYNTH_CLOSE_FAILURE("inv(opG) failed");
    return llvm::failure();
  }
  GridOp opG_inv = *opG_inv_or;

  // Steps 1-2: main loop over denominator exponents k = 0, 1, 2, ...
  //
  // At each k the TDGP enumerates candidates u in (1/sqrt(2)^k) * Z[omega]
  // with u in R_epsilon and conj_sq2(u) in the closed unit disk
  // (Definition 5.20). The T-count of the final circuit is 2k-2 or 2k
  // (Lemma 7.3), so scanning k from 0 upwards finds the T-optimal
  // approximation.
  const int64_t k_max = details::max_denominator_exponent(epsilon);
  LLVM_DEBUG(cudaq::synth::dbgs() << "k_max=" << k_max << '\n');

  Integer k = 0;
  for (; k <= k_max; k++) {
    CUDAQ_SYNTH_FENCE();
    CUDAQ_SYNTH_OPEN_SUB("k = " + std::to_string(static_cast<int64_t>(k)));

    TdgpStepper stepper(k, *region_or, unit_disk, opG_inv, transformed.bboxA,
                        transformed.bboxB, bboxA_y_fattened, bboxB_y_fattened);
    for (const DOmega &z : stepper) {
      // Step 2(a): residue gate.
      //
      // If conj(z) * z has residue 0 (i.e. is even in the Z[omega] residue
      // ring), then xi = 1 - conj(z) * z lands on an odd integer part and
      // the Diophantine equation is provably unsolvable. Lemma 8.4 says the
      // generic grid candidates satisfy n = conj_sq2(xi) * xi == 1 (mod 8)
      // when n != 0, which matches the solvability condition; the residue
      // check here is the cheapest test that filters out the unsolvable
      // ones before paying for factoring.
      //
      // Compute conj(z) * z once and reuse.
      DOmega z_conj_z = z.conj() * z;
      if (z_conj_z.residue() == 0)
        continue;

      // Step 2(b-c): solve conj(t) * t = xi for xi = 1 - conj(z) * z in
      // D[sqrt(2)]. DSqrt2::from_domega is well-defined because conj(z) * z
      // is real and lies in D[sqrt(2)] for any z in D[omega].
      DSqrt2 xi = DSqrt2(1) - DSqrt2::from_domega(z_conj_z);
      llvm::FailureOr<DOmega> w_or =
          diophantine_dyadic(xi, diophantine_timeout_ms, factoring_timeout_ms);

      if (llvm::succeeded(w_or)) {
        // We now have z and w with conj(z) * z + conj(w) * w = 1, so
        // U = [[ z, -conj(w) ], [ w, conj(z) ]] (equation (12), n = 0) is a
        // valid Clifford+T unitary approximating R_z(theta).

        DOmega z_reduced = to_lde(z);
        DOmega w_reduced = to_lde(*w_or);

        // Align the two components to a common denominator exponent so the
        // unitary's k is well-defined.
        if (z_reduced.k() > w_reduced.k())
          w_reduced = with_denom_exp(w_reduced, z_reduced.k());
        else if (z_reduced.k() < w_reduced.k())
          z_reduced = with_denom_exp(z_reduced, w_reduced.k());

        // Pick between two equivalent unitary representations that differ
        // by one T-gate (Lemma 7.3): if z + w admits a smaller LDE, the
        // straight pair (z, w) wins. Otherwise rotating w by omega gains
        // one denominator slot.
        DOmegaUnitary u_approx(DOmega::from_int(0), DOmega::from_int(0), 0);
        if (to_lde(z_reduced + w_reduced).k() < z_reduced.k())
          u_approx = DOmegaUnitary(z_reduced, w_reduced, 0);
        else
          u_approx = DOmegaUnitary(z_reduced, mul_by_omega(w_reduced), 0);

        std::string k_str = std::to_string(static_cast<int64_t>(k));
        CUDAQ_SYNTH_CLOSE_SUCCESS("Diophantine succeeded at k=" + k_str);
        CUDAQ_SYNTH_CLOSE_SUCCESS("synthesized at k=" + k_str);
        return u_approx;
      }
    }

    // No candidate at this k survived the Diophantine step (either no grid
    // points or every candidate timed out / proved unsolvable). Move to the
    // next k, accepting a larger T-count budget.
    CUDAQ_SYNTH_CLOSE_FAILURE("no candidates");
  }

  // Past k_max the answer would exceed the Ross & Selinger T-count bound by a
  // wide margin, so the search is not converging: the Diophantine solver is
  // starving on its timeouts rather than closing in. Report that instead of
  // scanning k forever..
  CUDAQ_SYNTH_CLOSE_FAILURE("k exceeded k_max without a solution");
  return llvm::failure();
}

//===----------------------------------------------------------------------===//
// gridsynth
//===----------------------------------------------------------------------===//

llvm::FailureOr<Circuit> gridsynth(const Real &theta, const Real &epsilon,
                                   int32_t diophantine_timeout_ms,
                                   int32_t factoring_timeout_ms) {
  CUDAQ_SYNTH_OPEN("gridsynth");
  LLVM_DEBUG(cudaq::synth::dbgs()
             << "theta=" << theta << ", eps=" << epsilon << '\n');

  llvm::FailureOr<DOmegaUnitary> u_or = gridsynth_unitary(
      theta, epsilon, diophantine_timeout_ms, factoring_timeout_ms);
  if (llvm::failed(u_or)) {
    CUDAQ_SYNTH_CLOSE_FAILURE("synthesis failed");
    return llvm::failure();
  }

  llvm::FailureOr<Circuit> circuit = kmm_synthesize(*u_or);
  if (llvm::succeeded(circuit)) {
    CUDAQ_SYNTH_CLOSE_SUCCESS(
        std::to_string((*circuit).size()) +
        " gates, T-count=" + std::to_string((*circuit).t_count()));
  } else {
    CUDAQ_SYNTH_CLOSE_FAILURE("kmm_synthesize failed");
  }
  return circuit;
}

} // namespace cudaq::synth
