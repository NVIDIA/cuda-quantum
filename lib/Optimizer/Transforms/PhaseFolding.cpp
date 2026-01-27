/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_PHASEFOLDING
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "phase-folding"

using namespace mlir;

#define RAW(X) quake::X
// AXIS-SPECIFIC: Defines which operations break a circuit into subcircuits
#define CIRCUIT_BREAKERS(MACRO)                                                \
  MACRO(YOp), MACRO(ZOp), MACRO(HOp), MACRO(R1Op), MACRO(RxOp),                \
      MACRO(PhasedRxOp), MACRO(RyOp), MACRO(U2Op), MACRO(U3Op)
#define RAW_CIRCUIT_BREAKERS CIRCUIT_BREAKERS(RAW)

// AXIS-SPECIFIC: could allow controlled y and z here
static bool isCNOT(Operation *op) {
  if (auto xop = dyn_cast<quake::XOp>(op))
    return xop.getControls().size() == 1;
  return false;
}

/// Currently, only `!quake.ref`s generated directly from
/// `quake.alloca`s are supported. This is with the assumption that
/// the `factor-quantum-alloc` pass was run before, so any veqs, etc...
/// with variable indices are excluded to prevent side effects from
/// breaking a circuit without it being noticed. This does unfortunately
/// restrict the possible optimizations, so future work to recognize
/// these possible side effects could be beneficial.
static bool isSupportedValue(Value ref) {
  if (!isa<quake::RefType>(ref.getType()))
    return false;

  if (!ref.getDefiningOp())
    return false;

  if (!isa<quake::AllocaOp>(ref.getDefiningOp()))
    return false;

  return true;
}

static bool isCircuitBreaker(Operation *op) {
  // TODO: it may be cleaner to only accept non-null input to
  // ensure the null case is explicitly handled by users
  if (!op)
    return true;

  if (!isQuakeOperation(op))
    return true;

  if (isa<RAW_CIRCUIT_BREAKERS, quake::NullWireOp>(op))
    return true;

  auto opi = dyn_cast<quake::OperatorInterface>(op);

  if (!opi)
    return true;

  // Only allow control in the case of CNOT
  if (opi.getControls().size() > 0 && !isCNOT(op))
    return true;

  // If any values are unsupported, the operation is also unsupported
  for (auto operand : quake::getQuantumOperands(op))
    if (!isSupportedValue(operand))
      return true;

  return false;
}

inline bool isTwoQubitOp(Operation *op) {
  return quake::getQuantumOperands(op).size() == 2;
}

namespace {

/// A netlist representation of a circuit is a list of lists,
/// with each sublist holding the operations on a particular
/// qubit in order. Multi-qubit operations will appear in the
/// lists of each of their operands.
class Netlist {
  SmallVector<SmallVector<Operation *>> netlists;
  SmallPtrSet<Operation *, 8> processed;

public:
  Netlist(mlir::func::FuncOp func) {
    func.walk([&](Operation *op) {
      if (auto allocaop = dyn_cast<quake::AllocaOp>(op)) {
        if (isa<quake::RefType>(allocaop.getType()))
          allocNetlist(allocaop);
        return;
      }

      if (isa<quake::OperatorInterface>(op))
        for (auto operand : quake::getQuantumOperands(op))
          if (isSupportedValue(operand))
            netlists[getIndexOf(operand)].push_back(op);
    });
  }

  void allocNetlist(Operation *refop) {
    auto nlindex = netlists.size();
    refop->setAttr(
        "nlindex",
        mlir::IntegerAttr::get(mlir::IntegerType::get(refop->getContext(), 64),
                               nlindex));
    auto nl = SmallVector<Operation *>();
    netlists.push_back(nl);
  }

  size_t getIndexOf(Value ref) {
    assert(isSupportedValue(ref));
    auto refop = ref.getDefiningOp();
    if (!refop->hasAttr("nlindex"))
      allocNetlist(refop);
    auto nlindex = refop->getAttrOfType<IntegerAttr>("nlindex").getInt();
    return nlindex;
  }

  size_t size() { return netlists.size(); }

  SmallVector<Operation *> *getNetlist(size_t index) {
    return &netlists[index];
  }

  void markProcessed(Operation *op) { processed.insert(op); }

  bool wasProcessed(Operation *op) { return processed.contains(op); }
};

/// A subcircuit is an connected portion of the netlist containing
/// only RZ, NOT, CNOT, and Swap gates. Currently it only accepts
/// `quake.ref` types produced directly by `quake.alloca`, to avoid
/// possible issues with aliasing of `quake.veq`s.
class Subcircuit {
protected:
  SmallVector<std::pair<Value, Operation *>> anchor_points;
  Netlist *container = nullptr;

  void addAnchorPoint(Value qubit, Operation *op) {
    anchor_points.push_back({qubit, op});
  }

  bool isTerminationPoint(Operation *op) {
    // Currently, each operation can only be part of one subcircuit (hence the
    // check for the processed flag)
    return (op->getBlock() != start->getBlock()) || isCircuitBreaker(op) ||
           container->wasProcessed(op);
  }

  class NetlistWrapper {
    Subcircuit *subcircuit = nullptr;
    SmallVector<Operation *> *nl = nullptr;
    Value def;
    // Inclusive
    size_t start_point;
    // Exclusive
    size_t end_point;

    size_t getIndexOf(Operation *op) {
      auto iter = std::find(nl->begin(), nl->end(), op);
      assert(iter != nl->end());
      return std::distance(nl->begin(), iter);
    }

    /// Returns `true` if processing should continue, `false` otherwise
    bool processOp(size_t op_idx) {
      auto op = (*nl)[op_idx];

      if (subcircuit->isTerminationPoint(op))
        return false;

      subcircuit->ops.insert(op);

      if (isTwoQubitOp(op)) {
        if (op->getOperand(0) == def)
          subcircuit->addAnchorPoint(op->getOperand(1), op);
        else
          subcircuit->addAnchorPoint(op->getOperand(0), op);
      } else if (!isa<quake::XOp>(op)) {
        // AXIS-SPECIFIC
        subcircuit->num_rot_gates++;
      }

      return true;
    }

    void processFrom(size_t index) {
      assert(index < nl->size());
      for (end_point = index + 1; end_point < nl->size(); end_point++)
        if (!processOp(end_point))
          break;
      for (start_point = index; start_point > 0; start_point--)
        if (!processOp(start_point))
          break;

      // Handle possible 0th element separately to prevent overflow
      // This is why start_point must be inclusive
      if (!processOp(start_point))
        start_point++;
    }

    void pruneFrom(size_t idx) {
      for (; idx < nl->size(); idx++) {
        auto op = (*nl)[idx];
        if (isTwoQubitOp(op)) {
          auto control = op->getOperand(0);
          auto target = op->getOperand(1);
          NetlistWrapper *otherWrapper = nullptr;
          if (def == control)
            otherWrapper = subcircuit->getWrapper(target);
          // If we are pruning along the target of a CNOT, we do not
          // need to prune along the control, as it will be unaffected
          else if (!isCNOT(op))
            otherWrapper = subcircuit->getWrapper(control);

          if (otherWrapper)
            otherWrapper->pruneFrom(op);
        } else if (isa<quake::RzOp>(op) && subcircuit->ops.contains(op)) {
          // AXIS-SPECIFIC
          subcircuit->num_rot_gates--;
        }
        subcircuit->ops.remove(op);
      }
    }

    void pruneFrom(Operation *op) {
      auto index = getIndexOf(op);
      if (index >= end_point)
        return;

      end_point = index;
      pruneFrom(index);
    }

  public:
    NetlistWrapper(Subcircuit *subcircuit, SmallVector<Operation *> *nl,
                   Operation *anchor_point, Value def)
        : subcircuit(subcircuit), nl(nl), def(def) {
      processFrom(getIndexOf(anchor_point));
    }

    void addNewAnchorPoint(Operation *op) {
      auto index = getIndexOf(op);
      if (index >= start_point)
        return;
      processFrom(index);
    }

    bool hasOps() { return end_point > start_point; }

    void prune() { pruneFrom(end_point); }

    Value getDef() { return def; }
  };

  SmallVector<NetlistWrapper *> qubits = {};
  SetVector<Operation *> ops = {};
  SmallVector<Operation *> ordered_ops = {};
  Operation *start = nullptr;
  size_t num_rot_gates = 0;

  void allocWrapper(Value ref, Operation *anchor_point) {
    auto nlindex = container->getIndexOf(ref);
    if (nlindex >= qubits.size())
      for (auto i = qubits.size(); i < container->size(); i++)
        qubits.push_back(nullptr);
    auto *nlw = new NetlistWrapper(this, container->getNetlist(nlindex),
                                   anchor_point, ref);
    qubits[nlindex] = nlw;
  }

  /// @brief Gets the NetlistWrapper for ref, if it exists
  /// @returns The NetlistWrapper for the Netlist for ref or
  /// `nullptr` if no such Netlist exists
  NetlistWrapper *getWrapper(Value ref) {
    if (!isSupportedValue(ref))
      return nullptr;

    auto nlindex = container->getIndexOf(ref);
    // Can still be nullptr if the wrapper hasn't been initialized
    return qubits[nlindex];
  }

  void processNextAnchorPoint() {
    auto next = anchor_points.back();
    anchor_points.pop_back();
    auto nl = getWrapper(next.first);
    if (nl)
      nl->addNewAnchorPoint(next.second);
    else
      allocWrapper(next.first, next.second);
  }

  void calculateInitialSubcircuit() {
    auto control = start->getOperand(0);
    auto target = start->getOperand(1);

    addAnchorPoint(control, start);
    addAnchorPoint(target, start);
    while (!anchor_points.empty())
      processNextAnchorPoint();
  }

  void pruneSubcircuit() {
    for (auto *netlist : qubits)
      if (netlist)
        netlist->prune();
    // Clean up
    for (size_t i = 0; i < qubits.size(); i++) {
      if (qubits[i] && !qubits[i]->hasOps()) {
        delete qubits[i];
        qubits[i] = nullptr;
      }
    }
  }

public:
  /// @brief Constructs a subcircuit containing only RZ, NOT, CNOT, and Swap
  /// gates, using the Netlist representation `netlist`.
  /// @details A subcircuit is an connected portion of the netlist containing
  /// only RZ, NOT, CNOT, and Swap gates.
  ///
  /// First, we construct an initial subcircuit:
  /// We start by walking forward and backward along the netlist from the
  /// initial anchor point, which is at `cnot` along the control qubit, and add
  /// any allowed gates to the subcircuit. If a CNOT or Swap gate is
  /// encountered, an anchor point is added at the gate for the other qubit,
  /// which will later be walked. If a disallowed gate is encountered, we stop
  /// walking and add a termination point.
  ///
  /// Then, we prune the subcircuit, starting at the earlist ending termination
  /// point (i.e., earliest termination point encountered while walking forward)
  /// along each qubit, and walk forward, adjusting the termination boundary for
  /// any connected qubits, and removing gates after the termination
  /// boundary from the subcircuit.
  Subcircuit(Operation *cnot, Netlist *netlist)
      : container(netlist), start(cnot) {
    assert(isCNOT(cnot));
    qubits = SmallVector<NetlistWrapper *>(netlist->size(), nullptr);
    calculateInitialSubcircuit();
    pruneSubcircuit();
    for (auto op : ops)
      netlist->markProcessed(op);
  }

  ~Subcircuit() {
    for (auto wrapper : qubits)
      if (wrapper)
        delete wrapper;
  }

  /// @brief Gets the !quake.refs used in the subcircuit
  SmallVector<Value> getRefs() {
    SmallVector<Value> refs;
    for (auto wrapper : qubits)
      if (wrapper)
        refs.push_back(wrapper->getDef());

    return refs;
  }

  /// @brief Gets the number of !quake.refs used in the subcircuit
  size_t numRefs() {
    size_t count = 0;
    for (auto wrapper : qubits)
      if (wrapper)
        count++;
    return count;
  }

  /// @brief Gets the operations in the subcircuit
  /// ordered by location in the containing block
  SmallVector<Operation *> getOrderedOps() {
    if (ordered_ops.size() == 0 && ops.size() > 0) {
      ordered_ops = SmallVector<Operation *>(ops.begin(), ops.end());
      auto less = [&](Operation *a, Operation *b) {
        return a->isBeforeInBlock(b);
      };
      std::sort(ordered_ops.begin(), ordered_ops.end(), less);
    }

    return ordered_ops;
  }

  /// @brief Gets the number of RZs in the subcircuit
  size_t getNumRotations() { return num_rot_gates; }

  /// @returns The percentage of operations in the subcircuit
  /// that are `quake.rz`s.
  float getRotationWeight() {
    return (float)getNumRotations() / (float)getNumOps();
  }

  /// @brief Gets the number of operations in the subcircuit
  size_t getNumOps() { return ops.size(); }
};

struct PhaseVariable {
public:
  size_t idx;
  // Q: do we really need the initial_wire here? I think it's just useful for
  // debugging
  Value initial_wire;
  PhaseVariable(size_t index, Value wire) : idx(index), initial_wire(wire) {}

  bool operator==(PhaseVariable other) { return idx == other.idx; }
};

/// A `Phase` is an exclusive sum of all of the `PhaseVariable`s involved in the
/// current state of a qubit, as well as 1, representing inversion from a NOT
/// gate. The simplest Phase contains exactly the `PhaseVariable` representing
/// the initial state of a qubit in a subcircuit. There are two operations on
/// `Phase`s to generate new `Phase`s: `Phase::sum` sums two Phases,
/// corresponding to the effect of a CNOT on the target qubit. `Phase::invert`
/// inverts a Phase, corresponding to the effect of a NOT a qubit.
///
/// Generally, a Phase is an exclusive sum of products.
/// However, our Phases are currently only exclusive sums;
/// products are not currently supported.
///
/// Any two rotations on qubits with equal Phases can be merged into one
/// rotation.
class Phase {
  SetVector<PhaseVariable *> vars;
  bool isInverted;

public:
  Phase() : isInverted(false) {}

  Phase(PhaseVariable *var) : isInverted(false) { vars.insert(var); }

  /// @brief Two phases are equal if they contain exactly the same vars
  /// and have the same inversion flag.
  bool operator==(Phase other) {
    for (auto var : vars)
      if (!other.vars.contains(var))
        return false;
    for (auto var : other.vars)
      if (!vars.contains(var))
        return false;
    return isInverted == other.isInverted;
  }

  /// @brief Returns a new phase equal to the sum of `p1` and `p2`
  /// @returns A new phase containing the exclusive or of `p1` and `p2`
  static Phase sum(Phase &p1, Phase &p2) {
    Phase new_phase = Phase();
    for (auto var : p1.vars)
      new_phase.vars.insert(var);
    for (auto var : p2.vars)
      if (new_phase.vars.contains(var))
        new_phase.vars.remove(var);
      else
        new_phase.vars.insert(var);
    new_phase.isInverted = (p1.isInverted != p2.isInverted);
    return new_phase;
  }

  /// @brief Returns a new phase equal to `p1` with the opposite inversion flag
  static Phase invert(Phase &p1) {
    auto new_phase = Phase();
    new_phase.vars.insert(p1.vars.begin(), p1.vars.end());
    new_phase.isInverted = !p1.isInverted;
    return new_phase;
  }

  void dump() {
    llvm::outs() << "Phase: ";
    if (isInverted)
      llvm::outs() << "!";
    llvm::outs() << "{";
    auto first = true;
    for (auto var : vars) {
      if (!first)
        llvm::outs() << " ^ ";
      llvm::outs() << var->idx;
      first = false;
    }
    llvm::outs() << "}\n";
  }
};

class PhaseStorage {
  SmallVector<Phase> phases;
  SmallVector<quake::RzOp> rotations;
  size_t numCombined = 0;

  /// @brief Merges the rotation at prev_idx with rzop by adding their
  /// rotation angles and overwriting rzop's angle with the new
  /// angle. The old rotation is erased. We keep the latter rotation
  /// to ensure that dynamic rotation angles (e.g., dependent on
  /// measurement results) are indeed available, as earlier angles
  /// will always be available later, but not vice-versa.
  void combineRotations(size_t prev_idx, quake::RzOp rzop) {
    auto old_rzop = rotations[prev_idx];
    auto rot_arg1 = old_rzop.getOperand(0);
    auto rot_arg2 = rzop.getOperand(0);
    auto builder = OpBuilder(rzop);
    auto new_rot_arg =
        builder.create<arith::AddFOp>(rzop.getLoc(), rot_arg1, rot_arg2);
    rzop->setOperand(0, new_rot_arg.getResult());
    old_rzop.erase();
    rotations[prev_idx] = rzop;
    numCombined++;
  }

public:
  /// @brief registers a new rotation op for the given phase
  /// @returns true if the rotation was combined, false otherwise
  bool addOrCombineRotationForPhase(quake::RzOp op, Phase phase) {
    for (size_t i = 0; i < phases.size(); i++)
      if (phases[i] == phase) {
        combineRotations(i, op);
        return true;
      }

    phases.push_back(phase);
    rotations.push_back(op);
    return false;
  }

  size_t getNumCombined() { return numCombined; }
};

class PhaseFoldingPass
    : public cudaq::opt::impl::PhaseFoldingBase<PhaseFoldingPass> {
  using PhaseFoldingBase::PhaseFoldingBase;

  // Perform the phase folding optimization over a subcircuit
  void doPhaseFolding(Subcircuit *subcircuit) {
    SmallVector<PhaseVariable *> phase_vars;
    SmallVector<Phase> current_phases;
    PhaseStorage store;
    size_t i = 0;
    // Initialize the phases and phase variables for each qubit in the circuit,
    // the initial phase contains only the phase variable for that qubit
    for (auto ref : subcircuit->getRefs()) {
      auto phase_idx = i++;
      auto new_phase_var = new PhaseVariable(phase_idx, ref);
      auto defop = ref.getDefiningOp();
      assert(defop);
      auto builder = OpBuilder(defop);
      // Q: is using attributes actually better than using a map of some sort?
      // Does it matter?
      defop->setAttr("phaseidx", builder.getUI32IntegerAttr(phase_idx));
      phase_vars.push_back(new_phase_var);
      current_phases.push_back(Phase(new_phase_var));
    }

    // A helper function to look up the phase for a quake.ref
    auto getPhase = [&](Value ref) {
      auto defop = ref.getDefiningOp();
      auto idx = defop->getAttrOfType<IntegerAttr>("phaseidx").getUInt();
      return current_phases[idx];
    };

    // A helper function to set the phase for a quake.ref
    auto setPhase = [&](Value ref, Phase phase) {
      auto defop = ref.getDefiningOp();
      auto idx = defop->getAttrOfType<IntegerAttr>("phaseidx").getUInt();
      current_phases[idx] = phase;
    };

    // Process all ops in the subcircuit, tracking phases and greedily trying to
    // merge rotations
    for (auto op : subcircuit->getOrderedOps()) {
      if (::isCNOT(op)) {
        // Controlled not, set new target phase to XOR of control and old target
        // phases
        auto control = op->getOperand(0);
        auto target = op->getOperand(1);
        auto control_phase = getPhase(control);
        auto target_phase = getPhase(target);
        auto new_target_phase = Phase::sum(target_phase, control_phase);
        setPhase(target, new_target_phase);
      } else if (isa<quake::XOp>(op)) {
        // Simple not, invert target phase
        // AXIS-SPECIFIC: Would want to handle y and z gates here too
        auto target = op->getOperand(0);
        auto target_phase = getPhase(target);
        auto new_target_phase = Phase::invert(target_phase);
        setPhase(target, new_target_phase);
      } else if (auto rzop = dyn_cast<quake::RzOp>(op)) {
        // Rotation, try to merge by looking up in store
        auto target = op->getOperand(1);
        auto target_phase = getPhase(target);
        store.addOrCombineRotationForPhase(rzop, target_phase);
      } else if (auto swap = dyn_cast<quake::SwapOp>(op)) {
        // Swap phases
        auto target1 = op->getOperand(0);
        auto target2 = op->getOperand(1);
        auto target1_phase = getPhase(target1);
        auto target2_phase = getPhase(target2);
        setPhase(target1, target2_phase);
        setPhase(target2, target1_phase);
      }
    }

    for (auto phase_var : phase_vars)
      delete phase_var;
  }

public:
  void runOnOperation() override {
    auto func = getOperation();
    mlir::DefaultTimingManager tm;
    tm.setEnabled(false);
    auto root = tm.getRootTimer();
    root.start();
    auto netlistBuild = root.nest("Building netlist");
    netlistBuild.start();
    // Get the netlist represention for the qubits in the function,
    // this will walk the whole function once
    auto nl = Netlist(func);
    netlistBuild.stop();
    SmallVector<Subcircuit *> subcircuits;

    auto subcircuitBuild = root.nest("Building subcircuits");
    subcircuitBuild.start();
    func.walk([&](quake::XOp op) {
      // AXIS-SPECIFIC: controlled not only
      if (!::isCNOT(op) || nl.wasProcessed(op))
        return;

      if (!isSupportedValue(op.getOperand(0)) ||
          !isSupportedValue(op.getOperand(1)))
        return;

      // Build a subcircuit from the CNOT
      auto subcircuit = new Subcircuit(op, &nl);
      // Ensure we're above thresholds
      if (subcircuit->getNumOps() < minimumBlockLength ||
          subcircuit->getRotationWeight() < minimumrzWeight) {
        LLVM_DEBUG(llvm::dbgs() << "Subcircuit below threshold, skipping!\n");
        delete subcircuit;
        return;
      }
      subcircuits.push_back(subcircuit);
    });
    subcircuitBuild.stop();

    // Performing the actual optimization over subcircuits after collecting them
    // A) allows for eventually parallelizing the optimization, and
    // B) avoids rewriting the AST as it is being walked above, causing an
    // error. This does introduce a requirement that each operation belongs to
    // at most one subcircuit.
    auto rotationMerging = root.nest("Merging rotations by phase");
    rotationMerging.start();
    for (auto subcircuit : subcircuits) {
      doPhaseFolding(subcircuit);
      // Clean up
      delete subcircuit;
    }
    rotationMerging.stop();

    root.stop();
    tm.setDisplayMode(mlir::DefaultTimingManager::DisplayMode::Tree);
  }
};

/// Phase folding pass pipeline command-line options.
struct PhaseFoldingPipelineOptions
    : public PassPipelineOptions<PhaseFoldingPipelineOptions> {
  PassOptions::Option<unsigned> minimumBlockLength{
      *this, "min-length",
      llvm::cl::desc(
          "Minimum subcircuit length to run phase folding. (default: 20)"),
      llvm::cl::init(20)};
  PassOptions::Option<double> minimumrzWeight{
      *this, "min-rz-weight",
      llvm::cl::desc("Minimumn percentage of rz ops in subcircuit to run phase "
                     "folding. (default: 0.2)"),
      llvm::cl::init(0.2)};
};
} // namespace

/// Add a pass pipeline to apply the requisite passes to fully unroll loops.
/// When converting to a quantum circuit, the static control program is fully
/// expanded to eliminate control flow. This pipeline will raise an error if any
/// loop in the module cannot be fully unrolled and signalFailure is set.
static void createPhaseFoldingPipeline(OpPassManager &pm, unsigned min_length,
                                       double min_rz_weight) {
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createFactorQuantumAllocations());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createCSEPass());
  cudaq::opt::PhaseFoldingOptions pfo{min_length, min_rz_weight};
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createPhaseFolding(pfo));
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createCombineQuantumAllocations());
}

void cudaq::opt::registerPhaseFoldingPipeline() {
  PassPipelineRegistration<PhaseFoldingPipelineOptions>(
      "phase-folding-pipeline",
      "Performs the phase-polynomial based rotation merging optimization.",
      [](OpPassManager &pm, const PhaseFoldingPipelineOptions &pfpo) {
        createPhaseFoldingPipeline(pm, pfpo.minimumBlockLength,
                                   pfpo.minimumrzWeight);
      });
}
