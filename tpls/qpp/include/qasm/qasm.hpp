/*
 * This file is part of Quantum++.
 *
 * Copyright (c) 2013 - 2022 softwareQ Inc. All rights reserved.
 *
 * MIT License
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/**
 * \file qasm/qasm.hpp
 * \brief OpenQASM to QCircuit interface
 */

#ifndef QASM_QASM_HPP_
#define QASM_QASM_HPP_

#ifndef USE_OPENQASM2_SPECS
#define USE_OPENQASM2_SPECS false
#endif

namespace qpp::qasm {

namespace ast = qasmtools::ast;
namespace parser = qasmtools::parser;

static std::unordered_map<ast::symbol,
                          std::function<cmat(const std::vector<double>&)>>
    known_matrices{
        ///< generators for various gate constants
        {"cx",
         [](const std::vector<double>&) {
             return Gates::get_no_thread_local_instance().CNOT;
         }},
        {"id",
         [](const std::vector<double>&) {
             return Gates::get_no_thread_local_instance().Id2;
         }},
        {"x",
         [](const std::vector<double>&) {
             return USE_OPENQASM2_SPECS
                        ? Gates::get_no_thread_local_instance().X * (-1_i)
                        : Gates::get_no_thread_local_instance().X;
         }},
        {"y",
         [](const std::vector<double>&) {
             return USE_OPENQASM2_SPECS
                        ? Gates::get_no_thread_local_instance().Y * (-1_i)
                        : Gates::get_no_thread_local_instance().Y;
         }},
        {"z",
         [](const std::vector<double>&) {
             return USE_OPENQASM2_SPECS
                        ? Gates::get_no_thread_local_instance().Z * (-1_i)
                        : Gates::get_no_thread_local_instance().Z;
         }},
        {"h",
         [](const std::vector<double>&) {
             return USE_OPENQASM2_SPECS
                        ? Gates::get_no_thread_local_instance().H * (-1_i)
                        : Gates::get_no_thread_local_instance().H;
         }},
        {"s",
         [](const std::vector<double>&) {
             return USE_OPENQASM2_SPECS
                        ? Gates::get_no_thread_local_instance().S *
                              std::exp(-1_i * pi / 4.0)
                        : Gates::get_no_thread_local_instance().S;
         }},
        {"sdg",
         [](const std::vector<double>&) {
             return USE_OPENQASM2_SPECS
                        ? (Gates::get_no_thread_local_instance().S.adjoint() *
                           std::exp(1_i * pi / 4.0))
                              .eval()
                        : Gates::get_no_thread_local_instance().S.adjoint();
         }},
        {"t",
         [](const std::vector<double>&) {
             return USE_OPENQASM2_SPECS
                        ? Gates::get_no_thread_local_instance().T *
                              std::exp(-1_i * pi / 8.0)
                        : Gates::get_no_thread_local_instance().T;
         }},
        {"tdg",
         [](const std::vector<double>&) {
             return USE_OPENQASM2_SPECS
                        ? (Gates::get_no_thread_local_instance().T.adjoint() *
                           std::exp(1_i * pi / 8.0))
                              .eval()
                        : Gates::get_no_thread_local_instance().T.adjoint();
         }},
        {"rx",
         [](const std::vector<double>& args) {
             assert(!args.empty());
             return Gates::get_no_thread_local_instance().RX(args[0]);
         }},
        {"rz",
         [](const std::vector<double>& args) {
             assert(!args.empty());
             return Gates::get_no_thread_local_instance().RZ(args[0]);
         }},
        {"ry",
         [](const std::vector<double>& args) {
             assert(!args.empty());
             return Gates::get_no_thread_local_instance().RY(args[0]);
         }},
        {"cz",
         [](const std::vector<double>&) {
             return USE_OPENQASM2_SPECS
                        ? Gates::get_no_thread_local_instance().CZ * (-1)
                        : Gates::get_no_thread_local_instance().CZ;
         }},
        {"cy",
         [](const std::vector<double>&) {
             cmat mat{cmat::Identity(4, 4)};
             mat.block(2, 2, 2, 2) = Gates::get_no_thread_local_instance().Y;
             return mat;
         }},
        {"swap",
         [](const std::vector<double>&) {
             return Gates::get_no_thread_local_instance().SWAP;
         }},
        {"ch",
         [](const std::vector<double>&) {
             cmat mat{cmat::Identity(4, 4)};
             mat.block(2, 2, 2, 2) = Gates::get_no_thread_local_instance().H;
             return USE_OPENQASM2_SPECS ? mat * std::exp(-1_i * pi / 4.0) : mat;
         }},
        {"ccx",
         [](const std::vector<double>&) {
             return USE_OPENQASM2_SPECS
                        ? Gates::get_no_thread_local_instance().TOF *
                              (-std::exp(-1_i * pi / 8.0))
                        : Gates::get_no_thread_local_instance().TOF;
         }},
        {"crz", [](const std::vector<double>& args) {
             assert(!args.empty());
             cmat mat{cmat::Identity(4, 4)};
             mat.block(2, 2, 2, 2) =
                 Gates::get_no_thread_local_instance().RZ(args[0]);
             return mat;
         }}};

/**
 * \class qpp::qasm::Value
 * \brief Interface class for OpenQASM values during evaluation
 *
 * Allows environments to contain different types of mappings depending on
 * the type of variable. See deriving classes
 */
class Value {
  public:
    virtual ~Value() = default;
};

/**
 * \class qpp::qasm::Circuit
 * \brief QASM circuit values
 * \see qpp::qasm::Value
 */
class Circuit : public Value {
  public:
    const std::vector<ast::symbol>& c_params_;   ///< the classical parameters
    const std::vector<ast::symbol>& q_params_;   ///< the quantum parameters
    const std::list<ast::ptr<ast::Gate>>& body_; ///< the circuit body

    Circuit(const std::vector<ast::symbol>& c_params,
            const std::vector<ast::symbol>& q_params,
            const std::list<ast::ptr<ast::Gate>>& body)
        : c_params_(c_params), q_params_(q_params), body_(body) {}
};

/**
 * \class qpp::qasm::Register
 * \brief QASM register values
 * \see qpp::qasm::Value
 */
class Register : public Value {
  public:
    [[maybe_unused]] bool quantum_; ///< whether the register is a qreg or creg
    std::vector<idx> indices_;      ///< the (virtual) register indices

    Register(bool quantum, std::vector<idx> indices)
        : quantum_(quantum), indices_(std::move(indices)) {}
};

/**
 * \class qpp::qasm::Qubit
 * \brief QASM qubit values
 * \see qpp::qasm::Value
 */
class Qubit : public Value {
  public:
    idx index_; ///< index of the qubit

    explicit Qubit(idx index) : index_(index) {}
};

/**
 * \class qpp::qasm::Number
 * \brief QASM number values
 * \see qpp::qasm::Value
 */
class Number : public Value {
  public:
    double value_; ///< the floating point number

    explicit Number(double value) : value_(value) {}
};

/**
 * \class qpp::qasm::Context
 * \brief QCircuit translation context
 *
 * Stores all information relevant to execution (i.e., generation of a
 * QCircuit). Each classical and quantum bit in the AST is mapped to a unique
 * classical or quantum index corresponding to a QCircuit qubit or bit.
 * Measurements are always non-destructive.
 */
class Context {
    QCircuit* circuit_; ///< pointer to the accumulating circuit

    // Hack for MSVC
    using hash_ident_uptr =
        std::unordered_map<ast::symbol, std::unique_ptr<Value>>;
    struct Environment {
        Environment() noexcept : val_(){};
        Environment(Environment&& rhs) noexcept : val_(std::move(rhs.val_)) {}
        hash_ident_uptr val_;
    };

    std::vector<Environment> env_{}; ///< environment stack
    [[maybe_unused]] std::list<idx>
        qubit_pool_{};                   ///< pool of unassigned physical qubits
    idx max_bit_ = static_cast<idx>(-1); ///< largest classical bit index
    idx max_qubit_ = static_cast<idx>(-1); ///< largest (virtual) qubit index

    // For controlled contexts
    std::vector<idx> cctrls_{}; ///< classical controls in the current context
    std::vector<idx> cctrl_shift_{}; ///< control shift

  public:
    /**
     * \brief Constructs a translation context with a given target QCircuit
     *
     * \param qc Pointer to a QCircuit object
     */
    explicit Context(QCircuit* qc) : circuit_(qc) {}

    /**
     * \brief Disables copy constructor
     */
    Context(const Context&) = delete;

    /**
     * \brief Disables assignment
     */
    Context& operator=(const Context&) = delete;

    /**
     * \brief The accumulating QCircuit
     *
     * \return Pointer to the QCircuit object
     */
    QCircuit* get_circuit() { return circuit_; }

    /*-------------- (Qu)bit management & allocation ---------------*/

    /**
     * \brief Allocate a classical bit
     *
     * \return Index of a fresh classical bit
     */
    idx alloc_bit() {
        idx ret = ++max_bit_;

        // check if circuit has enough classical bits
        if (max_bit_ >= circuit_->get_nc()) {
            circuit_->add_dit(1, ret);
        }

        return ret;
    }

    /**
     * \brief Allocate a virtual qubit
     *
     * \return Index of a fresh qubit
     */
    idx alloc_qubit() {
        idx ret = ++max_qubit_;

        // check if circuit has enough qubits
        if (max_qubit_ >= circuit_->get_nq()) {
            circuit_->add_qudit(1, ret);
        }

        return ret;
    }

    /*------------------ Environment management --------------*/
    /**
     * \brief Enter a new scope
     */
    void enter_scope() { env_.emplace_back(); }

    /**
     * \brief Exit current scope
     */
    void exit_scope() { env_.pop_back(); }

    /**
     * \brief Lookup an identifier in the environment
     *
     * Traverses all scopes from inner-most to outer-most and returns
     * the first mapping
     *
     * \param id Const reference to an identifier
     * \param loc Const reference to the location in the source file
     * \return Pointer to a value
     */
    Value* lookup(const ast::symbol& id, const parser::Position& loc) {
        for (auto table = env_.rbegin(); table != env_.rend(); table++) {
            auto it = table->val_.find(id);
            if (it != table->val_.end())
                return it->second.get();
        }

        std::stringstream context;
        context << "Bad lookup during QCircuit translation : " << loc << " , "
                << id;
        throw exception::CustomException("qpp::qasm::Context::lookup()",
                                         context.str());
    }

    /**
     * \brief Set the value of an identifier in the current scope
     * \note Assumes ownership of the value
     *
     * \param id Const reference to an identifier
     * \param val Unique pointer to a value
     */
    void set(const ast::symbol& id, std::unique_ptr<Value> val) {
        if (env_.empty())
            env_.emplace_back();
        env_.back().val_[id] = std::move(val);
    }

    /*------------------- Classical controls ----------------*/
    /**
     * \brief Enter a classically controlled context
     * \note QASM syntax does not allow nested classically controlled contexts
     *
     * \param cctrls Const reference to a list of classical controls
     * \param shift Const reference to a list of classical control values
     */
    void set_ccontrols(const std::vector<idx>& cctrls,
                       const std::vector<idx>& shift) {
        cctrls_ = cctrls;
        cctrl_shift_ = shift;
    }

    /**
     * \brief Exit a classically controlled context
     */
    void clear_ccontrols() {
        cctrls_.clear();
        cctrl_shift_.clear();
    }

    /**
     * \brief Whether the current context is classically controlled
     *
     * \return True if and only if the current context is classically controlled
     */
    bool ccontrolled() { return !cctrls_.empty(); }

    /**
     * \brief The current classical controls
     *
     * \return Const reference to the classical controls
     */
    const std::vector<idx>& get_cctrls() { return cctrls_; }

    /**
     * \brief The current classical control values
     *
     * \return Const reference to the classical control values
     */
    const std::vector<idx>& get_shift() { return cctrl_shift_; }
};

/**
 * \class qpp::qasm::QCircuitBuilder
 *
 * \brief Visitor for converting a QASM AST to a QCircuit
 */
class QCircuitBuilder final : public ast::Visitor {
    Context ctx; ///< QCircuit translation context
    double
        temp_value; ///< stores intermediate values when computing expressions

  public:
    /**
     * \brief Constructs a QCircuit builder for a given accumulating QCircuit
     *
     * \param qc Pointer to the accumulating QCircuit
     */
    explicit QCircuitBuilder(QCircuit* qc) : ctx(qc), temp_value(0) {}

    // Variables
    void visit(ast::VarAccess&) override {}

    // Expressions
    // - Set temp_value to the value of the expression
    void visit(ast::BExpr& expr) override {
        expr.lexp().accept(*this);
        double lval = temp_value;
        expr.rexp().accept(*this);
        double rval = temp_value;

        switch (expr.op()) {
            case ast::BinaryOp::Plus:
                temp_value = lval + rval;
                break;
            case ast::BinaryOp::Minus:
                temp_value = lval - rval;
                break;
            case ast::BinaryOp::Times:
                temp_value = lval * rval;
                break;
            case ast::BinaryOp::Divide:
                temp_value = lval / rval;
                break;
            case ast::BinaryOp::Pow:
                temp_value = pow(lval, rval);
                break;
            default:
                temp_value = 0;
        }
    }

    void visit(ast::UExpr& expr) override {
        expr.subexp().accept(*this);
        double val = temp_value;

        switch (expr.op()) {
            case ast::UnaryOp::Neg:
                temp_value = -val;
                break;
            case ast::UnaryOp::Sin:
                temp_value = std::sin(val);
                break;
            case ast::UnaryOp::Cos:
                temp_value = std::cos(val);
                break;
            case ast::UnaryOp::Tan:
                temp_value = std::tan(val);
                break;
            case ast::UnaryOp::Ln:
                temp_value = std::log(val);
                break;
            case ast::UnaryOp::Sqrt:
                temp_value = std::sqrt(val);
                break;
            case ast::UnaryOp::Exp:
                temp_value = std::exp(val);
                break;
            default:
                temp_value = 0;
        }
    }

    void visit(ast::PiExpr&) override { temp_value = qpp::pi; }

    void visit(ast::IntExpr& expr) override {
        temp_value = static_cast<double>(expr.value());
    }

    void visit(ast::RealExpr& expr) override { temp_value = expr.value(); }

    void visit(ast::VarExpr& expr) override {
        auto val =
            dynamic_cast<const Number*>(ctx.lookup(expr.var(), expr.pos()));
        temp_value = val->value_;
    }

    // Statements
    void visit(ast::MeasureStmt& stmt) override {
        auto q_args = var_access_as_qreg(stmt.q_arg());
        auto c_args = var_access_as_creg(stmt.c_arg());
        auto circuit = ctx.get_circuit();

        // apply measurements non-destructively
        for (idx i = 0; i < q_args.size(); i++) {
            circuit->measureZ(q_args[i], c_args[i], false);
        }
    }

    void visit(ast::ResetStmt& stmt) override {
        auto q_args = var_access_as_qreg(stmt.arg());
        auto circuit = ctx.get_circuit();

        circuit->reset(q_args);
    }

    void visit(ast::IfStmt& stmt) override {
        auto creg =
            dynamic_cast<const Register*>(ctx.lookup(stmt.var(), stmt.pos()));

        // create the shift
        int tmp = stmt.cond();
        std::vector<idx> shift(creg->indices_.size(), 0);
        for (idx i = 0; i < creg->indices_.size(); i++) {
            if (tmp % 2 == 0)
                shift[i] = 1;
            tmp >>= 1;
        }

        // apply controls to then branch
        ctx.set_ccontrols(creg->indices_, shift);
        stmt.then().accept(*this);
        ctx.clear_ccontrols();
    }

    // Gates
    void visit(ast::UGate& gate) override {
        gate.theta().accept(*this);
        double theta = temp_value;
        gate.phi().accept(*this);
        double phi = temp_value;
        gate.lambda().accept(*this);
        double lambda = temp_value;
        auto args = var_access_as_qreg(gate.arg());
        auto circuit = ctx.get_circuit();

        // generate the matrix
        cmat u{cmat::Zero(2, 2)};

#if USE_OPENQASM2_SPECS
        // standard QASM spec, as defined in
        // https://arxiv.org/pdf/1707.03429.pdf
        u << std::cos(theta / 2) * std::exp(-1_i * (phi + lambda) / 2.0),
            -(std::sin(theta / 2)) * std::exp(-1_i * (phi - lambda) / 2.0),
            std::sin(theta / 2) * std::exp(1_i * (phi - lambda) / 2.0),
            std::cos(theta / 2) * std::exp(1_i * (phi + lambda) / 2.0);
#else
        // Qiskit spec, as defined in
        // https://github.com/Qiskit/qiskit-terra/tree/master/qiskit/circuit/library/standard_gates
        // see https://github.com/vsoftco/qpp/issues/65
        u << std::cos(theta / 2),
            -(std::sin(theta / 2)) * std::exp(1_i * lambda),
            std::sin(theta / 2) * std::exp(1_i * phi),
            std::cos(theta / 2) * std::exp(1_i * (phi + lambda));
#endif

        // apply the gate
        for (auto i : args) {
            if (ctx.ccontrolled()) {
                circuit->cCTRL(u, ctx.get_cctrls(), i, ctx.get_shift(),
                               "Controlled-U");
            } else {
                circuit->gate(u, i, "U");
            }
        }
    }

    void visit(ast::CNOTGate& gate) override {
        auto ctrls = var_access_as_qreg(gate.ctrl());
        auto tgts = var_access_as_qreg(gate.tgt());
        auto circuit = ctx.get_circuit();

        // different combinations of controls and targets
        if (ctrls.size() == 1 && tgts.size() == 1) {
            if (ctx.ccontrolled()) {
                std::vector<idx> tmp{ctrls[0], tgts[0]};
                circuit->cCTRL_joint(Gates::get_no_thread_local_instance().CNOT,
                                     ctx.get_cctrls(), tmp, ctx.get_shift(),
                                     "CX");
            } else {
                circuit->gate(Gates::get_no_thread_local_instance().CNOT,
                              ctrls[0], tgts[0], "CX");
            }
        } else if (ctrls.size() > 1 && tgts.size() == 1) {
            for (idx ctrl : ctrls) {
                if (ctx.ccontrolled()) {
                    std::vector<idx> tmp{ctrl, tgts[0]};
                    circuit->cCTRL_joint(
                        Gates::get_no_thread_local_instance().CNOT,
                        ctx.get_cctrls(), tmp, ctx.get_shift(), "CX");
                } else {
                    circuit->gate(Gates::get_no_thread_local_instance().CNOT,
                                  ctrl, tgts[0], "CX");
                }
            }
        } else if (ctrls.size() == 1 && tgts.size() > 1) {
            for (idx tgt : tgts) {
                if (ctx.ccontrolled()) {
                    std::vector<idx> tmp{ctrls[0], tgt};
                    circuit->cCTRL_joint(
                        Gates::get_no_thread_local_instance().CNOT,
                        ctx.get_cctrls(), tmp, ctx.get_shift(), "CX");
                } else {
                    circuit->gate(Gates::get_no_thread_local_instance().CNOT,
                                  ctrls[0], tgt, "CX");
                }
            }
        } else if (ctrls.size() == tgts.size()) {
            for (idx i = 0; i < ctrls.size(); i++) {
                if (ctx.ccontrolled()) {
                    std::vector<idx> tmp{ctrls[i], tgts[i]};
                    circuit->cCTRL_joint(
                        Gates::get_no_thread_local_instance().CNOT,
                        ctx.get_cctrls(), tmp, ctx.get_shift(), "CX");
                } else {
                    circuit->gate(Gates::get_no_thread_local_instance().CNOT,
                                  ctrls[i], tgts[i], "CX");
                }
            }
        } // If registers have different lengths then it would be caught by the
          // parser
    }

    void visit(ast::BarrierGate&) override {}

    void visit(ast::DeclaredGate& dgate) override {
        auto gate =
            dynamic_cast<const Circuit*>(ctx.lookup(dgate.name(), dgate.pos()));

        // evaluate arguments
        std::vector<double> c_args(dgate.num_cargs());
        std::vector<std::vector<idx>> q_args(dgate.num_qargs());
        for (int i = 0; i < dgate.num_cargs(); i++) {
            dgate.carg(i).accept(*this);
            c_args[i] = temp_value;
        }
        for (int i = 0; i < dgate.num_qargs(); i++)
            q_args[i] = var_access_as_qreg(dgate.qarg(i));

        // map gate across registers
        idx mapping_size = 1;
        std::vector<bool> mapped(q_args.size(), false);
        for (idx i = 0; i < q_args.size(); i++) {
            if (q_args[i].size() > 1) {
                mapped[i] = true;
                mapping_size = q_args[i].size();
            }
        }

        auto it = known_matrices.find(dgate.name());
        if (it != known_matrices.end()) {
            // apply the known matrix directly
            auto circuit = ctx.get_circuit();
            auto mat = it->second(c_args);

            // map the gate across registers
            for (idx j = 0; j < mapping_size; j++) {
                // map virtual qubits to physical qubits
                std::vector<idx> mapped_args(q_args.size());
                for (idx i = 0; i < q_args.size(); i++) {
                    mapped_args[i] = mapped[i] ? q_args[i][j] : q_args[i][0];
                }

                // apply (possibly classical controlled) gate
                if (ctx.ccontrolled()) {
                    circuit->cCTRL_joint(mat, ctx.get_cctrls(), mapped_args,
                                         ctx.get_shift(), dgate.name());
                } else {
                    circuit->gate_joint(mat, mapped_args, dgate.name());
                }
            }
        } else {
            // push classical arguments onto a new scope
            ctx.enter_scope();
            for (idx i = 0; i < c_args.size(); i++) {
                ctx.set(gate->c_params_[i],
                        std::unique_ptr<Value>(new Number(c_args[i])));
            }

            // map the gate
            for (idx j = 0; j < mapping_size; j++) {
                ctx.enter_scope();
                for (idx i = 0; i < q_args.size(); i++) {
                    ctx.set(gate->q_params_[i],
                            std::unique_ptr<Value>(new Qubit(
                                mapped[i] ? q_args[i][j] : q_args[i][0])));
                }

                // evaluate the gate
                for (auto&& elem : gate->body_) {
                    elem->accept(*this);
                }

                ctx.exit_scope();
            }
            ctx.exit_scope();
        }
    }

    // Declarations
    void visit(ast::GateDecl& decl) override {
        ctx.set(decl.id(), std::unique_ptr<Value>(new Circuit(
                               decl.c_params(), decl.q_params(), decl.body())));
    }

    void visit(ast::OracleDecl& decl) override {
        std::stringstream context;
        context << "Oracle declarations not supported : " << decl.pos();
        throw exception::NotImplemented("qpp:qasm::QCircuitBuilder::visit()",
                                        context.str());
    }

    void visit(ast::RegisterDecl& decl) override {
        std::vector<idx> indices(decl.size());
        for (idx i = 0; i < (idx) decl.size(); i++) {
            indices[i] =
                decl.is_quantum() ? ctx.alloc_qubit() : ctx.alloc_bit();
        }
        ctx.set(decl.id(), std::unique_ptr<Value>(
                               new Register(decl.is_quantum(), indices)));
    }

    void visit(ast::AncillaDecl& decl) override {
        std::stringstream context;
        context << "Ancilla declarations not supported : " << decl.pos();
        throw exception::NotImplemented("qpp:qasm::QCircuitBuilder::visit()",
                                        context.str());
    }

    // Program
    void visit(ast::Program& prog) override {
        prog.foreach_stmt([this](auto& stmt) { stmt.accept(*this); });
    }

  private:
    /**
     * \brief Interpret the access as a classical register
     *
     * \param ap The variable access
     * \return List of bit indices corresponding to the variable access
     */
    std::vector<idx> var_access_as_creg(ast::VarAccess& ap) {
        auto reg =
            dynamic_cast<const Register*>(ctx.lookup(ap.var(), ap.pos()));

        if (ap.offset() == std::nullopt) {
            // creg
            return std::vector<idx>{reg->indices_};
        } else {
            // creg deref
            return std::vector<idx>{reg->indices_[ap.offset().value()]};
        }
    }

    /**
     * \brief Interpret the access as a quantum register
     *
     * \param ctx The variable access
     * \return List of virtual qubit indices corresponding to the variable
     * access
     */
    std::vector<idx> var_access_as_qreg(ast::VarAccess& ap) {
        if (ap.offset() == std::nullopt) {
            // qubit or qreg
            auto tmp = ctx.lookup(ap.var(), ap.pos());

            if (auto qubit = dynamic_cast<const Qubit*>(tmp)) {
                return std::vector<idx>{qubit->index_};
            } else {
                auto reg = dynamic_cast<const Register*>(tmp);

                return std::vector<idx>{reg->indices_};
            }
        } else {
            // qreg deref
            auto reg =
                dynamic_cast<const Register*>(ctx.lookup(ap.var(), ap.pos()));

            return std::vector<idx>{reg->indices_[ap.offset().value()]};
        }
    }
};

/**
 * \brief Reads an OpenQASM circuit from stdin and returns its qpp::QCircuit
 * representation
 *
 * \return qpp::QCircuit
 */
inline QCircuit read(std::istream& stream) {
    ast::ptr<ast::Program> program = parser::parse_stream(stream);

    std::unique_ptr<QCircuit> qc(
        new QCircuit(program->qubits(), program->bits()));
    QCircuitBuilder builder(qc.get());
    program->accept(builder);
    return *qc;
}

/**
 * \brief Reads an OpenQASM circuit from a file and returns its qpp::QCircuit
 * representation
 *
 * \return qpp::QCircuit
 */
inline QCircuit read_from_file(const std::string& fname) {
    ast::ptr<ast::Program> program = parser::parse_file(fname);

    std::unique_ptr<QCircuit> qc(
        new QCircuit(program->qubits(), program->bits()));
    QCircuitBuilder builder(qc.get());
    program->accept(builder);
    return *qc;
}

} /* namespace qpp::qasm */

#endif /* QASM_QASM_HPP_ */
