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
#include "cudaq/Synthesis/Synthesis/KmmSynthesize.h"

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
///   R_epsilon = { u in closed unit disk | dot(u, z) >= sqrt(1 - epsilon^2/4) }
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
  // Half-plane threshold d = sqrt(1 - epsilon^2/4); the membership test is
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
  ///   dot_threshold = sqrt(1 - epsilon^2/4) = cos(arcsin(epsilon/2))
  ///   z             = (cos(-theta/2), sin(-theta/2))
  /// Uses mpfr_sin_cos to get both trig values from a single argument.
  static Precomputed compute_precomputed(const Real &theta,
                                         const Real &epsilon) {
    Precomputed pre;
    pre.dot_threshold = sqrt(1 - ((epsilon * epsilon) / 4));

    Real half_angle = -theta / 2;
    mpfr_sin_cos(pre.z_y.get_mpfr(), pre.z_x.get_mpfr(), half_angle.get_mpfr(),
                 MPFR_RNDN);
    return pre;
  }

  /// Build the enclosing ellipse from the rotation D1 by -theta/2, an
  /// anisotropic scaling D2 = diag(64/epsilon^4, 4/epsilon^2) chosen to fit
  /// the lens, and the rotation D3 by +theta/2 back. The resulting quadratic
  /// form coefficients are:
  ///
  ///   A x^2 + 2B xy + C y^2 + Dx x + Dy y <= 1
  ///
  /// Returns failure() if Ellipse::create rejects the parameters; this
  /// should not happen for epsilon > 0.
  static llvm::FailureOr<Ellipse>
  make_bounding_ellipse(const Real &z_x, const Real &z_y,
                        const Real &dot_threshold, const Real &epsilon) {
    Real inv_eps2 = 1 / (epsilon * epsilon);
    Real inv_eps4 = inv_eps2 * inv_eps2;
    Real lambda_x = 64 * inv_eps4;       // D2(0, 0)
    const Real &lambda_y = 4 * inv_eps2; // D2(1, 1)

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
        make_bounding_ellipse(pre.z_x, pre.z_y, pre.dot_threshold, epsilon);
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
    Real cos_similarity = u.real() * z_x + u.imag() * z_y;
    return DSqrt2::from_domega(u.conj() * u) <= DSqrt2{1} &&
           cos_similarity >= dot_threshold;
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
  /// LLVM_DEBUG / CUDAQ_CUDAQ_SYNTH_OPEN_SUB diagnostic streams; not used in
  /// hot paths.
  std::string to_string() const {
    char prefix[256];
    mpfr_snprintf(prefix, sizeof(prefix),
                  "z=(%.6Rf,%.6Rf),dot_threshold=%.25Rf,", z_x.get_mpfr(),
                  z_y.get_mpfr(), dot_threshold.get_mpfr());
    return std::string(prefix) + bounding_ellipse.to_string();
  }
};

} // namespace

namespace cudaq::synth {

//===----------------------------------------------------------------------===//
// gridsynth_unitary
//===----------------------------------------------------------------------===//

llvm::FailureOr<DOmegaUnitary> gridsynth_unitary(const Real &theta,
                                                 const Real &epsilon,
                                                 int32_t diophantine_timeout_ms,
                                                 int32_t factoring_timeout_ms) {
  CUDAQ_CUDAQ_SYNTH_OPEN_SUB("gridsynth_unitary");
  LLVM_DEBUG(cudaq::synth::dbgs() << "theta=" << theta << "\n";
             cudaq::synth::dbgs() << "eps=" << epsilon << "\n";
             cudaq::synth::dbgs() << "diophantine_timeout="
                                  << diophantine_timeout_ms << "ms" << "\n";
             cudaq::synth::dbgs()
             << "factoring_timeout=" << factoring_timeout_ms << "ms" << "\n");

  // Step 0: build the epsilon-region and the closed-unit-disk constraint
  // applied to the sqrt(2)-conjugate (the latter is needed because
  // |conj(u) * u| <= 1 only if conj_sq2(u) also lies in the unit disk).
  llvm::FailureOr<EpsilonRegion> region_or =
      EpsilonRegion::create(theta, epsilon);
  if (llvm::failed(region_or)) {
    CUDAQ_CUDAQ_SYNTH_CLOSE_FAILURE("degenerate epsilon-region");
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
    CUDAQ_CUDAQ_SYNTH_CLOSE_FAILURE("to_upright preprocessing failed");
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
    CUDAQ_CUDAQ_SYNTH_CLOSE_FAILURE("inv(opG) failed");
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
  Integer k = 0;
  while (true) {
    CUDAQ_SYNTH_FENCE();
    CUDAQ_CUDAQ_SYNTH_OPEN_SUB("k = " +
                               std::to_string(static_cast<int64_t>(k)));

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
      if ((z * z.conj()).residue() == 0)
        continue;

      // Step 2(b-c): solve conj(t) * t = xi for xi = 1 - conj(z) * z in
      // D[sqrt(2)]. DSqrt2::from_domega is well-defined because conj(z) * z
      // is real and lies in D[sqrt(2)] for any z in D[omega].
      DSqrt2 xi = DSqrt2(1) - DSqrt2::from_domega(z.conj() * z);
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
        CUDAQ_CUDAQ_SYNTH_CLOSE_SUCCESS("Diophantine succeeded at k=" + k_str);
        CUDAQ_CUDAQ_SYNTH_CLOSE_SUCCESS("synthesized at k=" + k_str);
        return u_approx;
      }
    }

    // No candidate at this k survived the Diophantine step (either no grid
    // points or every candidate timed out / proved unsolvable). Move to the
    // next k, accepting a larger T-count budget.
    CUDAQ_CUDAQ_SYNTH_CLOSE_FAILURE("no candidates");
    k++;
  }
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
    CUDAQ_CUDAQ_SYNTH_CLOSE_FAILURE("synthesis failed");
    return llvm::failure();
  }

  llvm::FailureOr<Circuit> circuit = kmm_synthesize(*u_or);
  if (llvm::succeeded(circuit)) {
    CUDAQ_CUDAQ_SYNTH_CLOSE_SUCCESS(
        std::to_string((*circuit).size()) +
        " gates, T-count=" + std::to_string((*circuit).t_count()));
  } else {
    CUDAQ_CUDAQ_SYNTH_CLOSE_FAILURE("kmm_synthesize failed");
  }
  return circuit;
}

} // namespace cudaq::synth
