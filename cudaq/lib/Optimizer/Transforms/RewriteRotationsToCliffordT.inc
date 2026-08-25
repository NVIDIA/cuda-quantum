/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

constexpr double quarterTurn = llvm::numbers::pi / 4.0;

/// Reconstruct the correctly rounded value of mathematical k*pi/4 in the
/// source floating-point semantics. IEEE quad provides guard bits for this
/// multiplication, avoiding classification based on an already-rounded value
/// of pi/4.
static llvm::APFloat
canonicalQuarterTurns(int64_t quarterTurns,
                      const llvm::fltSemantics &semantics) {
  llvm::APFloat value(
      llvm::APFloat::IEEEquad(),
      "0.785398163397448309615660845819875721049292349843776455243736");
  llvm::APFloat factor = llvm::APFloat::getZero(llvm::APFloat::IEEEquad());
  factor.convertFromAPInt(
      llvm::APInt(64, static_cast<uint64_t>(quarterTurns), true), true,
      llvm::APFloat::rmNearestTiesToEven);
  value.multiply(factor, llvm::APFloat::rmNearestTiesToEven);
  bool losesInfo = false;
  value.convert(semantics, llvm::APFloat::rmNearestTiesToEven, &losesInfo);
  return value;
}

/// Classify an angle against the nearest k*pi/4 value. Exact mode compares the
/// folded constant with the canonical value rounded to the same floating-point
/// semantics. Threshold mode uses the operator-norm distance appropriate to the
/// operation.
template <typename QOP>
static std::optional<int64_t>
classifyCliffordTAngle(const llvm::APFloat &angle, double epsilon) {
  const double angleAsDouble = angle.convertToDouble();
  if (!std::isfinite(angleAsDouble))
    return std::nullopt;

  const double scaled = angleAsDouble / quarterTurn;
  constexpr double signedInt64Limit = 0x1p63;
  if (!std::isfinite(scaled) || scaled < -signedInt64Limit ||
      scaled >= signedInt64Limit)
    return std::nullopt;

  const double nearest = std::nearbyint(scaled);
  const auto quarterTurns = static_cast<int64_t>(nearest);
  const llvm::APFloat canonical =
      canonicalQuarterTurns(quarterTurns, angle.getSemantics());
  if (epsilon == 0.0) {
    if (!angle.bitwiseIsEqual(canonical))
      return std::nullopt;
  } else {
    const double delta =
        std::abs(angleAsDouble - canonical.convertToDouble());
    // For `delta = abs(theta - canonical)`, the spectral-norm distance
    // `||U(theta) - U(canonical)||_2` is `2*sin(delta/2)` for
    // `R1(theta) = diag(1, exp(i*theta))` and `2*sin(delta/4)` for axis
    // rotations `U(theta) = exp(-i*theta*sigma/2)`.
    constexpr double angleDivisor =
        std::is_same_v<QOP, cudaq::quake::R1Op> ? 2.0 : 4.0;
    const double distance = 2.0 * std::sin(delta / angleDivisor);
    if (!std::isfinite(distance) || distance > epsilon)
      return std::nullopt;
  }

  return quarterTurns;
}

/// Emit the simplified representative of a `quarterTurns*pi/4` rotation.
template <typename QOP>
static Value emitSimplifiedRotation(PatternRewriter &rewriter, QOP rotation,
                                    int64_t quarterTurns) {
  static_assert(llvm::is_one_of<QOP, cudaq::quake::R1Op,
                                cudaq::quake::RxOp, cudaq::quake::RyOp,
                                cudaq::quake::RzOp>::value);

  const Location location = rotation.getLoc();
  cudaq::opt::decomp::QuakeOperatorCreator creator(rewriter);
  Value target = rotation.getTarget();
  const unsigned rotationClass =
      static_cast<unsigned>(llvm::mod(quarterTurns, 8));
  const bool needsBasisChange =
      rotationClass != 0 && rotationClass != 4 &&
      llvm::is_one_of<QOP, cudaq::quake::RxOp,
                      cudaq::quake::RyOp>::value;

  // Non-`R1` rotations (`Rx`, `Ry`, and `Rz`) need a scalar
  // exp(-i*k*pi/8) to make their named-gate replacements exactly equal rather
  // than equal up to global phase.
  if constexpr (!std::is_same_v<QOP, cudaq::quake::R1Op>) {
    const FloatType phaseType =
        cast<FloatType>(rotation.getParameter().getType());
    const llvm::APFloat phaseAngle = [&] {
      // Compute the canonical angle in the source semantics.
      llvm::APFloat value = canonicalQuarterTurns(
          quarterTurns, phaseType.getFloatSemantics());
      // Divide the angle by two.
      value =
          llvm::scalbn(value, -1, llvm::APFloat::rmNearestTiesToEven);
      // Negate it for the required phase.
      value.changeSign();
      return value;
    }();
    const FloatAttr phaseAttr = rewriter.getFloatAttr(phaseType, phaseAngle);
    const Value phaseValue = arith::ConstantOp::create(
        rewriter, location, phaseType, phaseAttr);
    SmallVector<Value> noControls;
    creator.create<cudaq::quake::PhaseOp>(
        location, ValueRange{phaseValue}, noControls, target);
  }

  // Conjugate the remaining X/Y classes into the Z basis.
  if (needsBasisChange) {
    if constexpr (std::is_same_v<QOP, cudaq::quake::RxOp>) {
      creator.create<cudaq::quake::HOp>(location, target);
    } else if constexpr (std::is_same_v<QOP, cudaq::quake::RyOp>) {
      creator.create<cudaq::quake::SOp>(location, true, target);
      creator.create<cudaq::quake::HOp>(location, target);
    }
  }

  switch (rotationClass) {
  case 0: // `0`: identity.
    return target;
  case 1: // `pi/4`: `T`.
    creator.create<cudaq::quake::TOp>(location, target);
    break;
  case 2: // `pi/2`: `S`.
    creator.create<cudaq::quake::SOp>(location, target);
    break;
  case 3: // `3*pi/4`: `S` followed by `T`.
    creator.create<cudaq::quake::SOp>(location, target);
    creator.create<cudaq::quake::TOp>(location, target);
    break;
  case 4: // `pi`: direct Pauli gate; no basis change.
    if constexpr (std::is_same_v<QOP, cudaq::quake::RxOp>) {
      creator.create<cudaq::quake::XOp>(location, target);
    } else if constexpr (std::is_same_v<QOP, cudaq::quake::RyOp>) {
      creator.create<cudaq::quake::YOp>(location, target);
    } else {
      creator.create<cudaq::quake::ZOp>(location, target);
    }
    break;
  case 5: // `5*pi/4`: adjoint `S` followed by adjoint `T`.
    creator.create<cudaq::quake::SOp>(location, true, target);
    creator.create<cudaq::quake::TOp>(location, true, target);
    break;
  case 6: // `3*pi/2`: adjoint `S`.
    creator.create<cudaq::quake::SOp>(location, true, target);
    break;
  case 7: // `7*pi/4`: adjoint `T`.
    creator.create<cudaq::quake::TOp>(location, true, target);
    break;
  default:
    llvm_unreachable("quarter-turn class must be reduced modulo eight");
  }

  // Restore the original axis after the Z-basis gate sequence.
  if (needsBasisChange) {
    if constexpr (std::is_same_v<QOP, cudaq::quake::RxOp>) {
      creator.create<cudaq::quake::HOp>(location, target);
    } else if constexpr (std::is_same_v<QOP, cudaq::quake::RyOp>) {
      creator.create<cudaq::quake::HOp>(location, target);
      creator.create<cudaq::quake::SOp>(location, target);
    }
  }
  return target;
}

template <typename QOP>
class RewriteRotationToCliffordT : public OpRewritePattern<QOP> {
public:
  RewriteRotationToCliffordT(MLIRContext *context, double epsilon,
                             Pass::Statistic &stat)
      : OpRewritePattern<QOP>(context), epsilon(epsilon), stat(stat) {}

  LogicalResult matchAndRewrite(QOP rotation,
                                PatternRewriter &rewriter) const override {
    if (!rotation.getControls().empty() ||
        rotation.getNegatedQubitControls() || rotation.getTargets().size() != 1)
      return failure();

    if (!isa<cudaq::quake::WireType>(rotation.getTarget().getType()))
      return failure();

    FloatAttr angleAttr;
    if (!matchPattern(rotation.getParameter(), m_Constant(&angleAttr)))
      return failure();
    llvm::APFloat angle = angleAttr.getValue();
    if (rotation.isAdj())
      angle.changeSign();

    const auto quarterTurns = classifyCliffordTAngle<QOP>(angle, epsilon);
    // Skip angles that are not within `epsilon` of a multiple of `pi/4`.
    if (!quarterTurns)
      return failure();

    rewriter.replaceOp(
        rotation, emitSimplifiedRotation(rewriter, rotation, *quarterTurns));
    ++stat;
    return success();
  }

private:
  double epsilon;
  Pass::Statistic &stat;
};

static void populateRotationsToCliffordTPatterns(RewritePatternSet &patterns,
                                                 double epsilon,
                                                 Pass::Statistic &stat) {
  patterns.add<RewriteRotationToCliffordT<cudaq::quake::R1Op>,
               RewriteRotationToCliffordT<cudaq::quake::RxOp>,
               RewriteRotationToCliffordT<cudaq::quake::RyOp>,
               RewriteRotationToCliffordT<cudaq::quake::RzOp>>(
      patterns.getContext(), epsilon, stat);
}
