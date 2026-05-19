/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Synthesis/Synthesis/Gridsynth.h"

#include "Support/LogMacros.h"

#include "Math/Diophantine.h"
#include "Math/Geometry/GridOp.h"
#include "Math/Geometry/Rectangle.h"
#include "Math/Geometry/ToUpright.h"
#include "Math/Geometry/UnitDisk.h"
#include "Math/Grid/Tdgp.h"
#include "cudaq/Synthesis/Math/Real.h"
#include "cudaq/Synthesis/Synthesis/KmmSynthesize.h"

using namespace cudaq::synth;

namespace {

/// EpsilonRegion: the ε-region R_ε for the approximate z-rotation synthesis
/// problem.
///
/// Reference: Ross & Selinger, arXiv:1403.2975, §7.1, equation (14).
///
/// For a target z-rotation
///   R_z(θ) = `diag`(e^{-iθ/2}, e^{iθ/2})
/// and precision ε > 0, the ε-region is
///
///   R_ε = { u ∈ D̄ | u · z ≥ `sqrt`(1 - ε²/4) },
///
/// where
///   z = (cos(-θ/2), sin(-θ/2))
/// is the point on the unit circle corresponding to the target rotation, and D̄
/// is the closed unit disc.
///
/// Geometrically, R_ε is the intersection of the unit disc with a half‑plane:
/// a "lens" around z on the unit circle. The dot‑product condition
///   u · z ≥ 1 - ε² / 2
/// is equivalent to
///   ‖R_z(θ) - U‖ ≤ ε
/// for single-qubit unitaries U (equation (13) in the paper).
///
/// For the grid/ellipse machinery, we also maintain a *bounding ellipse* E that
/// contains R_ε. This is what the to_upright algorithm takes as input
/// (Theorem 5.16), and it is constructed by a specific linear transform
/// depending on θ and ε.
///
/// Important: the membership test inside() still uses the *exact* half‑plane ∩
/// disc definition of R_ε, not the ellipse approximation.
class EpsilonRegion : public ConvexSet {
private:
  // Threshold d = `sqrt`(1 - ε²/4) appearing in the half-plane inequality u·z ≥
  // d.
  Real dot_threshold;

  // Components of the direction vector z = (cos(-θ/2), sin(-θ/2)).
  Real z_x;
  Real z_y;

  // Ellipse that encloses R_ε, used for the grid/ellipse algorithms.
  Ellipse bounding_ellipse;

  /// Private constructor: takes pre-validated components.
  EpsilonRegion(Real dot_threshold_, Real z_x_, Real z_y_,
                Ellipse bounding_ellipse_)
      : dot_threshold(std::move(dot_threshold_)), z_x(std::move(z_x_)),
        z_y(std::move(z_y_)), bounding_ellipse(std::move(bounding_ellipse_)) {}

  /// Helper bundle for precomputed constants.
  struct Precomputed {
    Real dot_threshold;
    Real z_x;
    Real z_y;
  };

  /// Compute:
  ///   dot_threshold = `sqrt`(1 - ε²/4)  (= cos(arcsin(ε/2)))
  ///   z = (cos(-θ/2), sin(-θ/2)),
  /// using mpfr_sin_cos to obtain both trig values at once.
  ///
  /// The condition u · z ≥ dot_threshold is equivalent to |sin(φ)| ≤ ε/2,
  /// where φ is the angle between u and z.
  static Precomputed compute_precomputed(const Real &theta,
                                         const Real &epsilon) {
    Precomputed pre;
    pre.dot_threshold = sqrt(1 - ((epsilon * epsilon) / 4));

    Real half_angle = -theta / 2;
    mpfr_sin_cos(pre.z_y.get_mpfr(), pre.z_x.get_mpfr(), half_angle.get_mpfr(),
                 MPFR_RNDN);
    return pre;
  }

  /// Construct the enclosing ellipse E from the already‑computed z components
  /// and ε.  Returns failure() if the resulting parameters are not positive
  /// definite (should not happen for valid ε > 0).
  ///
  ///   D1 = rotation by -θ/2:   [[ z_x, -z_y], [ z_y,  z_x]]
  ///   D2 = `diag`(64/ε⁴, 4/ε²):  anisotropic scaling to fit the lens shape
  ///   D3 = rotation by  θ/2:   [[ z_x,  z_y], [-z_y,  z_x]]
  ///
  /// The resulting ellipse is the image of the unit disc under D1·D2·D3,
  /// written in the quadratic form expected by Ellipse.
  static FailureOr<Ellipse> make_bounding_ellipse(const Real &z_x,
                                                  const Real &z_y,
                                                  const Real &dot_threshold,
                                                  const Real &epsilon) {
    Real inv_eps2 = 1 / (epsilon * epsilon);
    Real inv_eps4 = inv_eps2 * inv_eps2;
    Real lambda_x = 64 * inv_eps4;       // D2(0,0)
    const Real &lambda_y = 4 * inv_eps2; // D2(1,1)

    // Quadratic form coefficients for the ellipse:
    //   A x² + 2B `xy` + C y² + D x + E y ≤ 1
    Real A = lambda_x * z_x * z_x + lambda_y * z_y * z_y;
    Real B = (lambda_x - lambda_y) * z_x * z_y;
    Real C = lambda_x * z_y * z_y + lambda_y * z_x * z_x;
    Real Dx = dot_threshold * z_x;
    Real Dy = dot_threshold * z_y;

    return Ellipse::create(A, B, C, Dx, Dy);
  }

public:
  /// Factory: creates the ε-region for target angle θ and precision ε.
  /// Returns failure() if the bounding ellipse is degenerate.
  static FailureOr<EpsilonRegion> create(const Real &theta,
                                         const Real &epsilon) {
    Precomputed pre = compute_precomputed(theta, epsilon);
    auto ell_or =
        make_bounding_ellipse(pre.z_x, pre.z_y, pre.dot_threshold, epsilon);
    if (failed(ell_or))
      return failure();
    return EpsilonRegion(std::move(pre.dot_threshold), std::move(pre.z_x),
                         std::move(pre.z_y), std::move(*ell_or));
  }

  const Ellipse &ellipse() const { return bounding_ellipse; }

  /// Exact membership test for the ε‑region R_ε:
  ///
  ///   u ∈ R_ε  ⇔  ‖u‖² ≤ 1   and   u · z ≥ d,
  ///
  /// where d = 1 - ε²/2 and z = (cos(-θ/2), sin(-θ/2)).
  /// A small tolerance is used to compensate for MPFR rounding.
  bool contains(const DOmega &u) const override {
    Real cos_similarity = u.real() * z_x + u.imag() * z_y;
    return DSqrt2::from_domega(u.conj() * u) <= DSqrt2{1} &&
           cos_similarity >= dot_threshold;
  }

  /// Intersect a ray with R_ε.
  ///
  /// The ray is parameterized as u(t) = u0 + t·v.
  /// We first intersect with the unit disc:
  ///   ‖u(t)‖² ≤ 1  ⇒  a t² + b t + c ≤ 0
  /// and solve the quadratic a t² + b t + c = 0 for [t0, t1].
  ///
  /// Then intersect [t0, t1] with the half‑plane constraint u(t) · z ≥ d:
  ///   (z·v) t ≥ d - z·u0.
  ///
  /// Returns [t_start, t_end] if there is a non‑empty intersection, or
  /// std::nullopt if the ray never passes through R_ε.
  std::optional<std::pair<Real, Real>>
  intersect(const DOmega &u0, const DOmega &v) const override {
    static const Real tolerance(1e-30);
    using Roots = std::pair<Real, Real>;

    // Intersection with unit disc: a t² + b t + c = 0
    DOmega a = v.conj() * v;
    DOmega b = DOmega::from_int(2) * (u0.conj() * v);
    DOmega c = u0.conj() * u0 - DOmega::from_dsqrt2(DSqrt2{1});

    std::optional<Roots> quad_solution =
        solve_quadratic(a.real(), b.real(), c.real());
    if (!quad_solution)
      return std::nullopt;

    auto &&[t0, t1] = quad_solution.value();

    // Half‑plane: z · u(t) ≥ d  ⇒  (z·v) t ≥ d - z·u0.
    Real z_dot_v = z_x * v.real() + z_y * v.imag();
    Real rhs = dot_threshold - (z_x * u0.real() + z_y * u0.imag());

    // z·v > 0: inequality is t ≥ `rhs` / (z·v).
    if (z_dot_v > tolerance) {
      Real t_min = std::max(t0, rhs / z_dot_v);
      if (t_min > t1)
        return std::nullopt;
      return std::make_pair(t_min, t1);
    }

    // z·v < 0: inequality flips: t ≤ `rhs` / (z·v).
    if (z_dot_v < -tolerance) {
      Real t_max = std::min(t1, rhs / z_dot_v);
      if (t0 > t_max)
        return std::nullopt;
      return std::make_pair(t0, t_max);
    }

    // z·v ≈ 0: the ray is (numerically) parallel to the boundary line.
    // Then the inequality is either always satisfied or never satisfied,
    // depending on `rhs` = d - z·u0.
    if (rhs <= tolerance)
      return std::make_pair(t0, t1);
    return std::nullopt;
  }

  /// Returns a compact human-readable string of the epsilon-region parameters.
  ///
  /// Intended for debug/trace logging only. Pass as a macro argument so
  /// Quill's level check keeps this call lazy — it is only evaluated when
  /// the active log level is TRACE or lower.
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

FailureOr<DOmegaUnitary> gridsynth_unitary(const Real &theta,
                                           const Real &epsilon,
                                           i32 diophantine_timeout_ms,
                                           i32 factoring_timeout_ms) {

  CUDAQ_SYNTH_LOG_DEBUG("synth.gridsynth",
                        "gridsynth_unitary: theta={}, eps={}, "
                        "dioph_timeout={}ms, fact_timeout={}ms",
                        theta, epsilon, diophantine_timeout_ms,
                        factoring_timeout_ms);

  // ---- Step 0: Setup ----
  // Construct the ε-region R_ε (§7.1, equation 14) and the unit disk
  // The unit disk serves as the constraint for u● (the √2-conjugate
  // must have norm ≤ 1 for u†u ≤ 1).
  FailureOr<EpsilonRegion> region_or = EpsilonRegion::create(theta, epsilon);
  if (failed(region_or)) {
    CUDAQ_SYNTH_LOG_ERROR("synth.gridsynth",
                          "gridsynth_unitary: degenerate epsilon-region "
                          "(eps={} may be too small or theta is degenerate)",
                          epsilon);
    return failure();
  }
  CUDAQ_SYNTH_LOG_TRACE("synth.gridsynth",
                        "gridsynth_unitary: epsilon-region: {}",
                        region_or->to_string());

  UnitDisk unit_disk;

  // ---- Step 0b: Upright preprocessing (Theorem 5.16) ----
  // Find a special grid operator G such that G(R_ε) and G●(D̄) are
  // both 1/6-upright. This preprocessing enables efficient enumeration
  // of grid points via bounding-box reduction (Lemma 5.8).
  FailureOr<UprightResult> transformed_or =
      to_upright(region_or->ellipse(), UnitDisk::as_ellipse());
  if (failed(transformed_or)) {
    CUDAQ_SYNTH_LOG_WARN("synth.gridsynth",
                         "gridsynth_unitary: to_upright preprocessing failed");
    return failure();
  }
  UprightResult &transformed = *transformed_or;

  CUDAQ_SYNTH_LOG_DEBUG("synth.gridsynth",
                        "gridsynth_unitary: to_upright done -- opG={}, "
                        "bboxA={}x{}, bboxB={}x{}",
                        transformed.opG, transformed.bboxA.I_x().width(),
                        transformed.bboxA.I_y().width(),
                        transformed.bboxB.I_x().width(),
                        transformed.bboxB.I_y().width());

  // Fatten the y-intervals of the bounding boxes by a small relative amount.
  // This guards against numerical edge effects where valid grid points
  // near the boundary might be missed due to floating-point rounding.
  Real epsilon_factor = Real(1e-4);
  Interval bboxA_y_fattened =
      fatten(transformed.bboxA.I_y(),
             transformed.bboxA.I_y().width() * epsilon_factor);
  Interval bboxB_y_fattened =
      fatten(transformed.bboxB.I_y(),
             transformed.bboxB.I_y().width() * epsilon_factor);

  FailureOr<GridOp> opG_inv_or = inv(transformed.opG);
  if (failed(opG_inv_or))
    return failure();
  GridOp opG_inv = *opG_inv_or;

  // ---- Steps 1-2: Main loop over denominator exponents k ----
  // For each k = 0, 1, 2, ..., enumerate candidates u ∈ (1/√2^k)·Z[ω]
  // with u ∈ R_ε and u● ∈ D̄ (the scaled TDGP, Definition 5.20).
  // The T-count of the final circuit will be 2k-2 or 2k (Lemma 7.3),
  // so iterating k from 0 ensures optimality.
  Integer k = 0;
  while (true) {
    CUDAQ_SYNTH_LOG_DEBUG("synth.gridsynth",
                          "gridsynth_unitary: trying denominator exponent k={}",
                          static_cast<i64>(k));

    // Step 1: Solve the scaled TDGP for denominator exponent k.
    // Returns candidates u ∈ (1/√2^k)·Z[ω] ∩ R_ε with u● ∈ D̄ lazily.
    // Step 2: For each candidate u (here called z), attempt Diophantine
    // completion to find w with w†w = 1 - z†z.
    for (const DOmega &z :
         solve_tdgp(k, *region_or, unit_disk, opG_inv, transformed.bboxA,
                    transformed.bboxB, bboxA_y_fattened, bboxB_y_fattened)) {
      // Step 2(a): Check residue condition.
      // If z†z ≡ 0 (mod 2), then ξ = 1 - z†z has odd integer part,
      // and the Diophantine equation is known to be unsolvable
      // (the factoring structure is incompatible). Skip such candidates.
      // By Lemma 8.4, for the "generic" candidates from the grid problem,
      // the value n = ξ●·ξ satisfies n ≡ 1 (mod 8) (when n ≠ 0), which
      // is exactly the solvability condition.
      if ((z * z.conj()).residue() == 0)
        continue;

      // Step 2(b-c): Compute ξ = 1 - z†z ∈ D[√2] and solve t†t = ξ.
      // DSqrt2::from_domega computes z†z as an element of D[√2]
      // (since z†z is always real and in D[√2] for z ∈ D[ω]).
      DSqrt2 xi = DSqrt2(1) - DSqrt2::from_domega(z.conj() * z);
      auto w_or =
          diophantine_dyadic(xi, diophantine_timeout_ms, factoring_timeout_ms);

      if (succeeded(w_or)) {
        CUDAQ_SYNTH_LOG_DEBUG(
            "synth.gridsynth",
            "gridsynth_unitary: Diophantine succeeded at k={}",
            static_cast<i64>(k));
        // SUCCESS: We have z and w with z†z + w†w = 1.
        // Construct U = [[z, -w†], [w, z†]] (equation 12, with n = 0).

        DOmega z_reduced = to_lde(z);
        DOmega w_reduced = to_lde(*w_or);

        // Align denominator exponents of z and w to the same k.
        if (z_reduced.k() > w_reduced.k())
          w_reduced = with_denom_exp(w_reduced, z_reduced.k());
        else if (z_reduced.k() < w_reduced.k())
          z_reduced = with_denom_exp(z_reduced, w_reduced.k());

        // Optimization: if z + w has a lower denominator exponent than z,
        // use the pair (z, w) directly. Otherwise, multiply w by ω to
        // reduce the combined denominator exponent by 1.
        // This corresponds to choosing between the two equivalent
        // representations of the unitary that differ by a T-gate
        // (Lemma 7.3: T-count is either 2k-2 or 2k).
        DOmegaUnitary u_approx(DOmega::from_int(0), DOmega::from_int(0), 0);
        if (to_lde(z_reduced + w_reduced).k() < z_reduced.k())
          u_approx = DOmegaUnitary(z_reduced, w_reduced, 0);
        else
          u_approx = DOmegaUnitary(z_reduced, mul_by_omega(w_reduced), 0);

        return u_approx;
      }
    }

    // No candidate at this k succeeded (either no grid points, or all
    // Diophantine solves failed/timed out). Increment k and try again
    // with a larger denominator exponent (higher T-count).
    k++;
  }
}

FailureOr<Circuit> gridsynth(const Real &theta, const Real &epsilon,
                             i32 diophantine_timeout_ms,
                             i32 factoring_timeout_ms) {
  CUDAQ_SYNTH_LOG_INFO("synth.gridsynth", "gridsynth: theta={}, eps={}", theta,
                       epsilon);

  FailureOr<DOmegaUnitary> u_or = gridsynth_unitary(
      theta, epsilon, diophantine_timeout_ms, factoring_timeout_ms);
  if (failed(u_or)) {
    CUDAQ_SYNTH_LOG_ERROR("synth.gridsynth",
                          "gridsynth: synthesis failed for theta={}, eps={}",
                          theta, epsilon);
    return failure();
  }

  FailureOr<Circuit> circuit = kmm_synthesize(*u_or);
  if (succeeded(circuit)) {
    CUDAQ_SYNTH_LOG_INFO("synth.gridsynth",
                         "gridsynth: success -- {} gates, T-count={}",
                         (*circuit).size(), (*circuit).t_count());
  }
  return circuit;
}

} // namespace cudaq::synth
