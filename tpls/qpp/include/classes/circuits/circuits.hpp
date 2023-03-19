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
 * \file classes/circuits/circuits.hpp
 * \brief Qudit quantum circuits
 */

#ifndef CLASSES_CIRCUITS_CIRCUITS_HPP_
#define CLASSES_CIRCUITS_CIRCUITS_HPP_

namespace qpp {
/**
 * \class qpp::QCircuit
 * \brief Quantum circuit description
 * \see qpp::QEngine
 */
class QCircuit : public IDisplay, public IJSON {
    friend class QEngine;

    idx nq_;           ///< number of qudits
    idx nc_;           ///< number of classical "dits"
    idx d_;            ///< qudit dimension
    std::string name_; ///< optional circuit name

    std::vector<bool>
        measured_; ///< keeps track of the destructively measured qudits
    std::vector<bool>
        measured_nd_; ///< keeps track of the non-destructively measured qudits
    std::vector<bool> clean_qudits_; ///< keeps track of clean (unused) qudits

    std::vector<bool>
        clean_dits_; ///< keeps track of clean (unused) classical dits
    std::vector<bool> measurement_dits_; ///< keeps track of classical dits used
    ///< in measurements

    std::unordered_map<std::size_t, cmat>
        cmat_hash_tbl_{}; ///< hash table with the matrices used in the circuit,
    ///< with [Key = std::size_t, Value = cmat]
    std::unordered_map<std::size_t, idx> gate_count_{}; ///< gate counts
    std::unordered_map<std::size_t, idx>
        measurement_count_{}; ///< measurement counts

    /**
     * \brief Adds matrix to the hash table
     *
     * \note Throws if a hash collision is detected, i.e., if two different
     * matrices have the same hash value
     *
     * \param hashU Hash value of U
     * \param U Complex matrix
     */
    void add_hash_(std::size_t hashU, const cmat& U) {
        // EXCEPTION CHECKS

        auto search = cmat_hash_tbl_.find(hashU);
        static internal::EqualEigen equal_eigen;
        if (search != cmat_hash_tbl_.end()) { // found the hash in the table
            // have a hash collision
            if (!equal_eigen(search->second, U))
                throw exception::CustomException("qpp::QCircuit::add_hash_()",
                                                 "Matrix hash collision");
        }
        // END EXCEPTION CHECKS
        cmat_hash_tbl_.insert({hashU, U});
    }

  public:
    /**
     * \brief Type of gate being executed in a gate step
     */
    enum class GateType {
        NONE, ///< represents no gate

        SINGLE, ///< unitary gate on a single qudit

        TWO, ///< unitary gate on 2 qudits

        THREE, ///< unitary gate on 3 qudits

        JOINT, ///< joint gate on multiple qudits

        FAN, ///< same unitary gate on multiple qudits

        SINGLE_CTRL_SINGLE_TARGET, ///< controlled 1 qudit unitary gate with
        ///< one control and one target

        SINGLE_CTRL_MULTIPLE_TARGET, ///< controlled 1 qudit unitary gate with
        ///< one control and multiple targets

        MULTIPLE_CTRL_SINGLE_TARGET, ///< controlled 1 qudit unitary gate with
        ///< multiple controls and single target

        MULTIPLE_CTRL_MULTIPLE_TARGET, ///< controlled 1 qudit unitary gate with
        ///< multiple controls and multiple
        ///< targets

        JOINT_CTRL, ///< controlled gate with multiple controls and joint
        ///< targets

        SINGLE_cCTRL_SINGLE_TARGET, ///< controlled 1 qudit unitary gate with
        ///< one classical control and one target

        SINGLE_cCTRL_MULTIPLE_TARGET, ///< controlled 1 qudit unitary gate with
        ///< one classical control and multiple
        ///< targets

        MULTIPLE_cCTRL_SINGLE_TARGET, ///< controlled 1 qudit unitary gate with
        ///< multiple classical controls and
        ///< single target

        MULTIPLE_cCTRL_MULTIPLE_TARGET, ///< controlled 1 qudit unitary gate
        ///< with multiple classical controls
        ///< and multiple targets

        JOINT_cCTRL, ///< controlled gate with multiple classical controls and
                     ///< joint targets
    };

    /**
     * \brief Extraction operator overload for qpp::QCircuit::GateType enum
     * class
     *
     * \param os Output stream passed by reference
     * \param gate_type qpp::QCircuit::GateType enum class
     * \return Reference to the output stream
     */
    friend std::ostream& operator<<(std::ostream& os,
                                    const GateType& gate_type) {
        switch (gate_type) {
            case GateType::NONE:
                os << "GATE NONE";
                break;
            case GateType::SINGLE:
                os << "SINGLE";
                break;
            case GateType::TWO:
                os << "TWO";
                break;
            case GateType::THREE:
                os << "THREE";
                break;
            case GateType::FAN:
                os << "FAN";
                break;
            case GateType::JOINT:
                os << "JOINT";
                break;
            case GateType::SINGLE_CTRL_SINGLE_TARGET:
                os << "SINGLE_CTRL_SINGLE_TARGET";
                break;
            case GateType::SINGLE_CTRL_MULTIPLE_TARGET:
                os << "SINGLE_CTRL_MULTIPLE_TARGET";
                break;
            case GateType::MULTIPLE_CTRL_SINGLE_TARGET:
                os << "MULTIPLE_CTRL_SINGLE_TARGET";
                break;
            case GateType::MULTIPLE_CTRL_MULTIPLE_TARGET:
                os << "MULTIPLE_CTRL_MULTIPLE_TARGET";
                break;
            case GateType::JOINT_CTRL:
                os << "JOINT_CTRL";
                break;
            case GateType::SINGLE_cCTRL_SINGLE_TARGET:
                os << "SINGLE_cCTRL_SINGLE_TARGET";
                break;
            case GateType::SINGLE_cCTRL_MULTIPLE_TARGET:
                os << "SINGLE_cCTRL_MULTIPLE_TARGET";
                break;
            case GateType::MULTIPLE_cCTRL_SINGLE_TARGET:
                os << "MULTIPLE_cCTRL_SINGLE_TARGET";
                break;
            case GateType::MULTIPLE_cCTRL_MULTIPLE_TARGET:
                os << "MULTIPLE_cCTRL_MULTIPLE_TARGET";
                break;
            case GateType::JOINT_cCTRL:
                os << "JOINT_cCTRL";
                break;
        }

        return os;
    }

    /**
     * \brief One step consisting only of gates/operators in the circuit
     */
    struct GateStep : IDisplay {
        GateType gate_type_ = GateType::NONE; ///< gate type
        std::size_t gate_hash_{};             ///< gate hash
        std::vector<idx> ctrl_{};             ///< control
        std::vector<idx> target_{}; ///< target where the gate is applied
        std::vector<idx> shift_{};  ///< shifts in CTRL gates
        std::string name_{};        ///< custom name of the gate(s)

        /**
         * \brief Default constructor
         */
        GateStep() = default;

        /**
         * \brief Constructs a gate step instance
         *
         * \param gate_type Gate type
         * \param gate_hash Hash of the quantum gate
         * \param ctrl Control qudit indexes
         * \param target Target qudit indexes
         * \param shift Optional gate shifts (for CTRL gates)
         * \param name Optional gate name
         */
        explicit GateStep(GateType gate_type, std::size_t gate_hash,
                          std::vector<idx> ctrl, std::vector<idx> target,
                          std::vector<idx> shift = {}, std::string name = {})
            : gate_type_{gate_type}, gate_hash_{gate_hash},
              ctrl_{std::move(ctrl)}, target_{std::move(target)},
              shift_{std::move(shift)}, name_{std::move(name)} {}

        /**
         * \brief Equality operator
         * \note Ignores gate names
         *
         * \param rhs GateStep against which the equality is being tested
         * \return True if the GateStep(s) are equal, false otherwise
         */
        bool operator==(const GateStep& rhs) const noexcept {
            return std::tie(rhs.target_, rhs.shift_, rhs.gate_type_,
                            rhs.gate_hash_, rhs.ctrl_) ==
                   std::tie(target_, shift_, gate_type_, gate_hash_, ctrl_);
        }

      private:
        /**
         * \brief qpp::IDisplay::display() override
         *
         * Writes to the output stream a textual representation of the
         * \a qpp::QCircuit::GateStep instance
         *
         * \param os Output stream passed by reference
         * \return Reference to the output stream
         */
        std::ostream& display(std::ostream& os) const override {
            os << gate_type_ << ", ";
            if (gate_type_ >= GateType::SINGLE_cCTRL_SINGLE_TARGET)
                os << "c_ctrl = " << disp(ctrl_, ", ") << ", ";
            else if (gate_type_ >= GateType::SINGLE_CTRL_SINGLE_TARGET)
                os << "ctrl = " << disp(ctrl_, ", ") << ", ";
            os << "target = " << disp(target_, ", ") << ", ";
            if (!shift_.empty())
                os << "shift = " << disp(shift_, ", ") << ", ";
            os << "name = " << '\"' << name_ << '\"';

            return os;
        }
    };

    /**
     * \brief Type of measurement being executed in a measurement step
     */
    enum class MeasureType {
        NONE, ///< represents no measurement

        MEASURE_Z, ///< Z measurement of single qudit

        MEASURE_Z_MANY, ///< Z measurement of joint qudits

        MEASURE_V, ///< measurement of single qudit in the orthonormal basis
        ///< or rank-1 projectors specified by the columns of matrix
        ///< \a V

        MEASURE_V_MANY, ///< joint measurement of multiple qudits in the
        ///< orthonormal basis or rank-1 projectors specified
        ///< by the columns of the matrix \a V

        MEASURE_Z_ND, ///< Z measurement of single qudit, non-destructive

        MEASURE_Z_MANY_ND, ///< Z measurement of multiple qudit, non-destructive

        MEASURE_V_ND, ///< measurement of single qudit in the orthonormal basis
        ///< or rank-1 projectors specified by the columns of
        ///< matrix \a V, non-destructive

        MEASURE_V_MANY_ND, ///< joint measurement of multiple qudits in the
        ///< orthonormal basis or rank-1 projectors specified
        ///< by the columns of the matrix \a V,
        ///< non-destructive

        RESET, ///< resets single qudit

        RESET_MANY, ///< resets multiple qudits

        DISCARD, ///< discards single qudit

        DISCARD_MANY, ///< discards multiple qudits
    };

    /**
     * \brief Extraction operator overload for qpp::QCircuit::MeasureType enum
     * class
     *
     * \param os Output stream passed by reference
     * \param measure_type qpp::QCircuit::MeasureType enum class
     * \return Reference to the output stream
     */
    friend std::ostream& operator<<(std::ostream& os,
                                    const MeasureType& measure_type) {
        switch (measure_type) {
            case MeasureType::NONE:
                os << "MEASURE NONE";
                break;
            case MeasureType::MEASURE_Z:
                os << "MEASURE_Z";
                break;
            case MeasureType::MEASURE_Z_MANY:
                os << "MEASURE_Z_MANY";
                break;
            case MeasureType::MEASURE_V:
                os << "MEASURE_V";
                break;
            case MeasureType::MEASURE_V_MANY:
                os << "MEASURE_V_MANY";
                break;
            case MeasureType::MEASURE_Z_ND:
                os << "MEASURE_Z_ND";
                break;
            case MeasureType::MEASURE_Z_MANY_ND:
                os << "MEASURE_Z_MANY_ND";
                break;
            case MeasureType::MEASURE_V_ND:
                os << "MEASURE_V_ND";
                break;
            case MeasureType::MEASURE_V_MANY_ND:
                os << "MEASURE_V_MANY_ND";
                break;
            case MeasureType::RESET:
                os << "RESET";
                break;
            case MeasureType::RESET_MANY:
                os << "RESET_MANY";
                break;
            case MeasureType::DISCARD:
                os << "DISCARD";
                break;
            case MeasureType::DISCARD_MANY:
                os << "DISCARD_MANY";
                break;
        }

        return os;
    }

    /**
     * \brief One step consisting only of measurements in the circuit
     */
    struct MeasureStep : IDisplay {
        MeasureType measurement_type_ = MeasureType::NONE; ///< measurement type
        std::vector<std::size_t> mats_hash_{}; ///< hashes of measurement
        ///< matrix/matrices
        std::vector<idx> target_{}; ///< target where the measurement is applied
        idx c_reg_{}; ///< index of the classical register where the measurement
        ///< result is being stored
        std::string name_{}; ///< custom name of the measurement(s)

        /**
         * \brief Default constructor
         */
        MeasureStep() = default;

        /**
         * \brief Constructs a measurement step instance
         *
         * \param measurement_type Measurement type
         * \param mats_hash Vector of hashes of the measurement matrix/matrices
         * \param target Target qudit indexes
         * \param c_reg Classical register where the value of the measurement is
         * stored
         * \param name Optional gate name
         */
        explicit MeasureStep(MeasureType measurement_type,
                             std::vector<std::size_t> mats_hash,
                             std::vector<idx> target, idx c_reg,
                             std::string name = {})
            : measurement_type_{measurement_type},
              mats_hash_{std::move(mats_hash)}, target_{std::move(target)},
              c_reg_{c_reg}, name_{std::move(name)} {}

        /**
         * \brief Equality operator
         * \note Ignores measurement names
         *
         * \param rhs MeasureStep against which the equality is being tested
         * \return True if the MeasureStep(s) are equal, false otherwise
         */
        bool operator==(const MeasureStep& rhs) const noexcept {
            return std::tie(rhs.target_, rhs.measurement_type_, rhs.mats_hash_,
                            rhs.c_reg_) ==
                   std::tie(target_, measurement_type_, mats_hash_, c_reg_);
        }

      private:
        /**
         * \brief qpp::IDisplay::display() override
         *
         * Writes to the output stream a textual representation of the
         * \a qpp::QCircuit::Measure instance
         *
         * \param os Output stream passed by reference
         * \return Reference to the output stream
         */
        std::ostream& display(std::ostream& os) const override {
            os << measurement_type_ << ", ";
            os << "target = " << disp(target_, ", ") << ", ";
            if (measurement_type_ != MeasureType::RESET &&
                measurement_type_ != MeasureType::RESET_MANY &&
                measurement_type_ != MeasureType::DISCARD &&
                measurement_type_ != MeasureType::DISCARD_MANY)
                os << "c_reg = " << c_reg_ << ", ";
            os << "name = " << '\"' << name_ << '\"';

            return os;
        }
    };

    /**
     * \brief Types of each step in the quantum circuit description
     */
    enum class StepType {
        NONE,        ///< represents no step
        GATE,        ///< quantum gate(s)
        MEASUREMENT, ///< measurement
        NOP,         ///< no-op
    };

    /**
     * \brief Quantum circuit resources
     */
    struct Resources : IDisplay, IJSON {
        idx nq{}, nc{}, d{};
        std::string name{};
        idx step_count{};
        idx gate_count{};
        idx gate_depth{};
        idx measurement_count{};
        idx measurement_depth{};
        idx total_depth{};

        /**
         * \brief qpp::IJSON::to_JSON() override
         *
         * Displays the quantum circuit resources in JSON format
         *
         * \param enclosed_in_curly_brackets If true, encloses the result in
         * curly brackets
         * \return String containing the JSON representation of the quantum
         * circuit resources
         */
        std::string
        to_JSON(bool enclosed_in_curly_brackets = true) const override {
            std::string result;

            if (enclosed_in_curly_brackets)
                result += "{";

            result += "\"nq\": " + std::to_string(nq) + ", ";
            result += "\"nc\": " + std::to_string(nc) + ", ";
            result += "\"d\": " + std::to_string(d) + ", ";

            result += "\"step count\": " + std::to_string(step_count) + ", ";
            result +=
                "\"total gate count\": " + std::to_string(gate_count) + ", ";
            result +=
                "\"total gate depth\": " + std::to_string(gate_depth) + ", ";
            result += "\"total measurement count\": " +
                      std::to_string(measurement_count) + ", ";
            result += "\"total measurement depth\": " +
                      std::to_string(measurement_depth) + ", ";
            result += "\"total depth\": " + std::to_string(total_depth);

            if (enclosed_in_curly_brackets)
                result += "}";

            return result;
        }

      private:
        /**
         * \brief qpp::IDisplay::display() override
         *
         * Writes to the output stream a textual representation of the quantum
         * circuit resources
         *
         * \param os Output stream passed by reference
         * \return Reference to the output stream
         */
        std::ostream& display(std::ostream& os) const override {
            os << "[Resources]\n";

            os << "<QCircuit "
               << "nq: " << nq << ", nc: " << nc << ", d: " << d;
            if (!name.empty())
                os << ", name: \"" << name << '"';
            os << ">\n";

            os << "step count: " << step_count << '\n';
            os << "total gate count: " << gate_count << '\n';
            os << "total gate depth: " << gate_depth << '\n';
            os << "total measurement count: " << measurement_count << '\n';
            os << "total measurement depth: " << measurement_depth << '\n';
            os << "total depth: " << total_depth;

            return os;
        }
    };

  private:
    std::vector<GateStep> gates_{};           ///< gates
    std::vector<MeasureStep> measurements_{}; ///< measurements
    std::vector<StepType> step_types_{};      ///< type of each step

    /**
     * \brief Vector of qpp::QCircuit::MeasureStep
     *
     * \return Vector of qpp::QCircuit::MeasureStep
     */
    const std::vector<MeasureStep>& get_measurements_() const noexcept {
        return measurements_;
    }

    /**
     * \brief Vector of qpp::QCircuit::GateStep
     *
     * \return Vector of qpp::QCircuit::GateStep
     */
    const std::vector<GateStep>& get_gates_() const noexcept { return gates_; }

    /**
     * \brief Hash table with the matrices used in the circuit
     *
     * \return Hash table with the matrices used in the circuit
     */
    const std::unordered_map<std::size_t, cmat>&
    get_cmat_hash_tbl_() const noexcept {
        return cmat_hash_tbl_;
    }

  public:
    /**
     * \class qpp::QCircuit::iterator
     * \brief Quantum circuit description bound-checking (safe) forward iterator
     *
     * \note The iterator is a const_iterator by default
     */
    class iterator {
        ///< non-owning pointer to the parent const quantum circuit description
        const QCircuit* qc_{nullptr};

        /**
         * \class qpp::QCircuit::iterator::value_type_
         * \brief Value type class for qpp::QCircuit::iterator
         */
        class value_type_ : public IDisplay {
          public:
            ///< non-owning pointer to the grand-parent const quantum circuit
            ///< description
            const QCircuit* value_type_qc_;

            StepType type_{StepType::NONE}; ///< step type
            idx ip_{static_cast<idx>(-1)};  ///< instruction pointer
            std::vector<GateStep>::const_iterator
                gates_ip_{}; ///< gates instruction pointer
            std::vector<MeasureStep>::const_iterator
                measurements_ip_{}; ///< measurements instruction pointer

            /**
             * \brief Constructor
             *
             * \param value_type_qc Pointer to constant quantum circuit
             * description
             */
            explicit value_type_(const QCircuit* value_type_qc)
                : value_type_qc_{value_type_qc} {}

            // silence -Weffc++ class has pointer data members
            /**
             * \brief Default copy constructor
             */
            value_type_(const value_type_&) = default;

            // silence -Weffc++ class has pointer data members
            /**
             * \brief Default copy assignment operator
             *
             * \return Reference to the current instance
             */
            value_type_& operator=(const value_type_&) = default;

          private:
            /**
             * \brief qpp::IDisplay::display() override
             *
             * Writes to the output stream the textual representation of the
             * iterator de-referenced element
             *
             * \param os Output stream passed by reference
             * \return Reference to the output stream
             */
            std::ostream& display(std::ostream& os) const override {
                // field spacing for the step number
                idx text_width =
                    std::to_string(value_type_qc_->get_step_count()).size();

                os << std::left;
                os << std::setw(static_cast<int>(text_width)) << ip_ << ": ";
                os << std::right;

                // gate step
                if (type_ == StepType::GATE) {
                    idx pos = std::distance(std::begin(value_type_qc_->gates_),
                                            gates_ip_);
                    os << value_type_qc_->get_gates_()[pos];
                }
                // measurement step
                else if (type_ == StepType::MEASUREMENT) {
                    idx pos =
                        std::distance(std::begin(value_type_qc_->measurements_),
                                      measurements_ip_);

                    switch (
                        value_type_qc_->measurements_[pos].measurement_type_) {
                        case MeasureType::NONE:
                            break;
                        case MeasureType::MEASURE_Z:
                        case MeasureType::MEASURE_Z_MANY:
                        case MeasureType::MEASURE_V:
                        case MeasureType::MEASURE_V_MANY:
                            os << "|> ";
                            break;
                        case MeasureType::MEASURE_Z_ND:
                        case MeasureType::MEASURE_Z_MANY_ND:
                        case MeasureType::MEASURE_V_ND:
                        case MeasureType::MEASURE_V_MANY_ND:
                            os << "|] ";
                            break;
                        case MeasureType::RESET:
                        case MeasureType::RESET_MANY:
                            os << "|* ";
                            break;
                        case MeasureType::DISCARD:
                        case MeasureType::DISCARD_MANY:
                            os << "|x ";
                            break;
                    } /* end switch */

                    os << value_type_qc_->get_measurements_()[pos];
                }
                // no-op
                else if (type_ == StepType::NOP) {
                    os << "NOP";
                }
                // otherwise
                else {
                }

                return os;
            }
        }; /* class QCircuit::value_type_ */

        value_type_ elem_{nullptr};

      public:
        /**
         * \brief Default constructor
         */
        iterator() = default;

        // silence -Weffc++ class has pointer data members
        /**
         * \brief Default copy constructor
         */
        iterator(const iterator&) = default;

        // silence -Weffc++ class has pointer data members
        /**
         * \brief Default copy assignment operator
         *
         * \return Reference to the current instance
         */
        iterator& operator=(const iterator&) = default;

        /**
         * \brief Prefix increment operator
         *
         * \return Reference to the current instance
         */
        iterator& operator++() {
            // EXCEPTION CHECKS

            // protects against incrementing invalid iterators
            if (qc_ == nullptr) {
                throw exception::InvalidIterator(
                    "qpp::QCircuit::iterator::operator++()",
                    "No qpp::QCircuit assigned");
            }

            auto num_steps = qc_->get_step_count();
            // protects against incrementing an empty circuit iterator
            if (num_steps == 0) {
                throw exception::InvalidIterator(
                    "qpp::QCircuit::iterator::operator++()",
                    "Zero-sized qpp::QCircuit");
            }

            // protects against incrementing past the end
            if (elem_.ip_ == num_steps) {
                throw exception::InvalidIterator(
                    "qpp::QCircuit::iterator::operator++()",
                    "Incrementing past the end");
            }
            // END EXCEPTION CHECKS

            // gate step
            if (elem_.type_ == StepType::GATE) {
                std::advance(elem_.gates_ip_, 1);
            }
            // measurement step
            else if (elem_.type_ == StepType::MEASUREMENT) {
                std::advance(elem_.measurements_ip_, 1);
            }
            // no-op
            else if (elem_.type_ == StepType::NOP) {
            }
            // otherwise
            else {
            }

            // increment the instruction pointer
            ++elem_.ip_;

            // if we hit the end
            if (elem_.ip_ == qc_->get_step_count()) {
                elem_.type_ = StepType::NONE;
            } else {
                // set the next step type
                elem_.type_ = qc_->step_types_[elem_.ip_];
            }

            return *this;
        }

        /**
         * \brief Postfix increment operator
         *
         * \return Copy of the current instance before the increment
         */
        iterator operator++(int) {
            iterator retval = *this;
            ++(*this);
            return retval;
        }

        /**
         * \brief Equality operator
         *
         * \param rhs Iterator against which the equality is being tested
         * \return True if the iterators are equal, false otherwise
         */
        bool operator==(const iterator& rhs) const noexcept {
            return std::tie(elem_.type_, elem_.ip_, elem_.gates_ip_,
                            elem_.measurements_ip_) ==
                   std::tie(rhs.elem_.type_, rhs.elem_.ip_, rhs.elem_.gates_ip_,
                            rhs.elem_.measurements_ip_);
        }

        /**
         * \brief Inequality operator
         *
         * \param rhs Iterator against which the inequality is being tested
         * \return True if the iterators are not equal (bit by bit), false
         * otherwise
         */
        bool operator!=(const iterator& rhs) const noexcept {
            return !(*this == rhs);
        }

        /**
         * \brief Safe de-referencing operator
         *
         * \return Constant reference to the iterator element
         */
        const value_type_& operator*() const {
            // EXCEPTION CHECKS

            // protects against de-referencing past the last element or against
            // de-referencing invalid iterators
            auto num_steps = qc_->get_step_count();
            if (qc_ == nullptr)
                throw exception::InvalidIterator(
                    "qpp::QCircuit::iterator::operator*()",
                    "No qpp::QCircuit assigned");
            if (num_steps == 0)
                throw exception::InvalidIterator(
                    "qpp::QCircuit::iterator::operator*()",
                    "Zero-sized qpp::QCircuit");
            if (elem_.ip_ == num_steps)
                throw exception::InvalidIterator(
                    "qpp::QCircuit::iterator::operator*()",
                    "Dereferencing past the end");
            // END EXCEPTION CHECKS

            return elem_;
        }

        /**
         * \brief Sets the iterator to std::begin(this)
         *
         * \param qc Pointer to constant quantum circuit description
         */
        void set_begin_(const QCircuit* qc) {
            qc_ = qc;
            elem_ = value_type_{qc_};

            if (qc_ != nullptr) {
                if (qc_->get_step_count() != 0) // non-empty circuit
                {
                    elem_.type_ = qc_->step_types_[0];
                    elem_.ip_ = 0;
                }
                elem_.gates_ip_ = std::begin(qc_->gates_);
                elem_.measurements_ip_ = std::begin(qc_->measurements_);
            }
        }

        /**
         * \brief Sets the iterator to std::begin(this)
         *
         * \param qc Pointer to constant quantum circuit description
         */
        void set_end_(const QCircuit* qc) {
            qc_ = qc;
            elem_ = value_type_{qc_};

            if (qc_ != nullptr) {
                if (qc->get_step_count() != 0) {
                    elem_.ip_ = qc->get_step_count();
                }
                elem_.gates_ip_ = std::end(qc->gates_);
                elem_.measurements_ip_ = std::end(qc->measurements_);
            }
        }

        // iterator traits
        using difference_type = std::ptrdiff_t;              ///< iterator trait
        using value_type = value_type_;                      ///< iterator trait
        using pointer = const value_type*;                   ///< iterator trait
        using reference = const value_type&;                 ///< iterator trait
        using iterator_category = std::forward_iterator_tag; ///< iterator trait
    }; /* class QCircuit::iterator */

    using const_iterator = iterator; ///< both iterators are const_iterators

    /**
     * \brief Iterator to the first element
     *
     * \return Iterator to the first element
     */
    iterator begin() noexcept {
        iterator it;
        it.set_begin_(this);

        return it;
    }

    /**
     * \brief Constant iterator to the first element
     *
     * \return Constant iterator to the first element
     */
    const_iterator begin() const noexcept {
        iterator it;
        it.set_begin_(this);

        return it;
    }

    /**
     * \brief Constant iterator to the first element
     *
     * \return Constant iterator to the first element
     */
    const_iterator cbegin() const noexcept {
        iterator it;
        it.set_begin_(this);

        return it;
    }

    /**
     * \brief Iterator to the next to the last element
     *
     * \return Iterator to the next to the last element
     */
    iterator end() noexcept {
        iterator it;
        it.set_end_(this);

        return it;
    }

    /**
     * \brief Constant iterator to the next to the last element
     *
     * \return Constant iterator to the next to the last element
     */
    const_iterator end() const noexcept {
        iterator it;
        it.set_end_(this);

        return it;
    }

    /**
     * \brief Constant iterator to the next to the last element
     *
     * \return Constant iterator to the next to the last element
     */
    const_iterator cend() const noexcept {
        iterator it;
        it.set_end_(this);

        return it;
    }

    /**
     * \brief Constructs a quantum circuit description
     *
     * \note The measurement results can only be stored in the classical dits of
     * which number is specified by \a nc
     *
     * \param nq Number of qudits (optional, defaults to 1 so qpp::QCircuit is
     * default-constructible)
     * \param nc Number of classical dits (optional)
     * \param d Subsystem dimensions (optional, default is qubit, i.e., \a d =2)
     * \param name Circuit name (optional)
     */
    explicit QCircuit(idx nq = 1, idx nc = 0, idx d = 2, std::string name = {})
        : nq_{nq}, nc_{nc}, d_{d}, name_{std::move(name)}, measured_(nq, false),
          measured_nd_(nq, false), clean_qudits_(nq_, true),
          clean_dits_(nc_, true), measurement_dits_(nc_, false) {
        // EXCEPTION CHECKS

        // if (nq == 0)
        //    throw exception::ZeroSize("qpp::QCircuit::QCircuit()", "nq");
        if (d < 2)
            throw exception::OutOfRange("qpp::QCircuit::QCircuit()", "d");
        // END EXCEPTION CHECKS
    }

    /**
     * \brief Default virtual destructor
     */
    ~QCircuit() override = default;

    // static member functions
    /**
     * \brief Checks whether a gate step is a controlled gate
     *
     * \return True if the gate step is a controlled gate, false otherwise
     */
    inline static bool is_CTRL(const GateStep& gate_step) {
        switch (gate_step.gate_type_) {
            case GateType::SINGLE_CTRL_SINGLE_TARGET:
            case GateType::SINGLE_CTRL_MULTIPLE_TARGET:
            case GateType::MULTIPLE_CTRL_SINGLE_TARGET:
            case GateType::MULTIPLE_CTRL_MULTIPLE_TARGET:
            case GateType::JOINT_CTRL:
                return true;
            default:
                return false;
        }
    }

    /**
     * \brief Checks whether a gate step is a classically-controlled gate
     *
     * \return True if the gate step is a classically-controlled gate, false
     * otherwise
     */
    inline static bool is_cCTRL(const GateStep& gate_step) {
        switch (gate_step.gate_type_) {
            case GateType::SINGLE_cCTRL_SINGLE_TARGET:
            case GateType::SINGLE_cCTRL_MULTIPLE_TARGET:
            case GateType::MULTIPLE_cCTRL_SINGLE_TARGET:
            case GateType::MULTIPLE_cCTRL_MULTIPLE_TARGET:
            case GateType::JOINT_cCTRL:
                return true;
            default:
                return false;
        }
    }

    /**
     * \brief Checks whether a gate step is a regular gate (not a controlled
     * gate)
     *
     * \return True if the gate step is a regular gate, false otherwise
     */
    inline static bool is_non_CTRL(const GateStep& gate_step) {
        return !(is_CTRL(gate_step) || is_cCTRL(gate_step));
    }

    // getters
    /**
     * \brief Total number of qudits in the circuit
     *
     * \return Total number of qudits
     */
    idx get_nq() const noexcept { return nq_; }

    /**
     * \brief Total number of classical dits in the circuit
     *
     * \return Total number of classical dits
     */
    idx get_nc() const noexcept { return nc_; }

    /**
     * \brief Dimension of the comprising qudits
     *
     * \return Qudit dimension
     */
    idx get_d() const noexcept { return d_; }

    /**
     * \brief Quantum circuit description name
     *
     * \return Quantum circuit description name
     */
    std::string get_name() const { return name_; }
    /**
     * \brief Check whether qudit \a i was already measured (destructively)
     * \param i Qudit index
     * \return True if qudit \a i was already measured (destructively), false
     * otherwise
     */
    bool get_measured(idx i) const {
        // EXCEPTION CHECKS

        if (i >= nq_)
            throw exception::OutOfRange("qpp::QCircuit::get_measured()", "i");
        // END EXCEPTION CHECKS

        return measured_[i];
    }

    /**
     * \brief Vector of already measured (destructively) qudit indexes
     *
     * \return Vector of already measured (destructively) qudit indexes
     */
    std::vector<idx> get_measured() const {
        std::vector<idx> result;
        for (idx i = 0; i < nq_; ++i)
            if (get_measured(i))
                result.emplace_back(i);

        return result;
    }

    /**
     * \brief Check whether qudit \a i was already measured (non-destructively)
     * \param i Qudit index
     * \return True if qudit \a i was already measured (non-destructively),
     * false otherwise
     */
    bool get_measured_nd(idx i) const {
        // EXCEPTION CHECKS

        if (i >= nq_)
            throw exception::OutOfRange("qpp::QCircuit::get_measured_nd()",
                                        "i");
        // END EXCEPTION CHECKS

        return measured_nd_[i];
    }

    /**
     * \brief Vector of already measured (non-destructively) qudit indexes
     *
     * \return Vector of already measured (non-destructively) qudit indexes
     */
    std::vector<idx> get_measured_nd() const {
        std::vector<idx> result;
        for (idx i = 0; i < nq_; ++i)
            if (get_measured_nd(i))
                result.emplace_back(i);

        return result;
    }

    /**
     * \brief Vector of non-measured (destructively) qudit indexes
     *
     * \return Vector of non-measured (destructively) qudit indexes
     */
    std::vector<idx> get_non_measured() const {
        std::vector<idx> result;
        for (idx i = 0; i < nq_; ++i)
            if (!get_measured(i))
                result.emplace_back(i);

        return result;
    }

    /**
     * \brief Vector of classical dits that were used to store results of
     * measurements (either destructive or non-destructive)
     *
     * \return Vector of classical dits that participate in measurements
     */
    std::vector<idx> get_measurement_dits() const {
        std::vector<idx> result;
        for (idx i = 0; i < nc_; ++i)
            if (is_measurement_dit(i))
                result.emplace_back(i);

        return result;
    }

    /**
     * \brief Quantum circuit description gate count
     *
     * \param U Gate
     * \return Gate count
     */
    idx get_gate_count(const cmat& U) const {
        // EXCEPTION CHECKS

        // square matrix
        if (!internal::check_square_mat(U))
            throw exception::MatrixNotSquare("qpp::get_gate_count()", "U");
        // END EXCEPTION CHECKS

        idx result = 0;
        std::size_t hashU = hash_eigen(U);
        // gate hash not found in the hash table
        try {
            result = gate_count_.at(hashU);
        } catch (...) {
            return 0;
        }

        return result;
    }

    /**
     * \brief Quantum circuit description total gate count
     *
     * \return Total gate count
     */
    idx get_gate_count() const {
        idx result = 0;

        for (auto&& elem : gate_count_)
            result += elem.second;

        return result;
    }

    /**
     * \brief Quantum circuit description gate depth
     *
     * \param U Gate
     * \return Gate depth
     */
    idx get_gate_depth(const cmat& U) const {
        // EXCEPTION CHECKS

        // square matrix
        if (!internal::check_square_mat(U))
            throw exception::MatrixNotSquare("qpp::get_gate_depth()", "U");
        // END EXCEPTION CHECKS

        bool found = false;
        std::vector<idx> heights(nc_ + nq_, 0);

        cmat all_gates(1, 1);
        all_gates << static_cast<double>(std::hash<std::string>{}("all gates"));
        bool compute_total_depth = false;
        if (U.rows() == 1 && U.cols() == 1)
            if (U == all_gates)
                compute_total_depth = true;

        std::size_t hashU = hash_eigen(U);
        // iterate over all steps in the circuit
        for (auto&& step : *this) {
            // gates
            if (step.type_ == StepType::GATE) {
                GateStep gate_step = *step.gates_ip_;

                if (gate_step.gate_hash_ != hashU && !compute_total_depth)
                    continue; // we skip this gate step

                found = true; // gate was found in the circuit

                std::vector<idx> ctrl = gate_step.ctrl_;
                std::vector<idx> target = gate_step.target_;
                std::vector<idx> ctrl_target;
                ctrl_target.reserve(ctrl.size() + target.size());
                ctrl_target.insert(ctrl_target.end(), ctrl.begin(), ctrl.end());
                ctrl_target.insert(ctrl_target.end(), target.begin(),
                                   target.end());

                idx max_height = 0;

                if (is_cCTRL(gate_step)) {
                    // compute the "height" of the to-be-placed gate
                    for (auto&& i : ctrl)
                        if (heights[i] > max_height)
                            max_height = heights[i];
                    for (auto&& i : target)
                        if (heights[nc_ + i] > max_height)
                            max_height = heights[nc_ + i];
                    // apply classical ctrl
                    for (auto&& i : ctrl)
                        heights[i] = max_height + 1;
                    // apply gate
                    for (auto&& i : target)
                        heights[nc_ + i] = max_height + 1;
                } else {
                    // compute the "height" of the to-be-placed gate
                    for (auto&& i : ctrl_target)
                        if (heights[nc_ + i] > max_height)
                            max_height = heights[nc_ + i];
                    // apply (ctrl) gate
                    for (auto&& i : ctrl_target)
                        heights[nc_ + i] = max_height + 1;
                }
            } // end if (step.type_ == StepType::GATE)
        }     // end for

        return found ? *std::max_element(std::begin(heights), std::end(heights))
                     : 0;
    }

    /**
     * \brief Quantum circuit description total gate depth
     *
     * \return Total gate depth
     */
    idx get_gate_depth() const {
        cmat all_gates(1, 1);
        all_gates << static_cast<double>(std::hash<std::string>{}("all gates"));
        return get_gate_depth(all_gates);
    }

    /**
     * \brief Quantum circuit description measurement depth
     *
     * \param V Orthonormal basis or rank-1 projectors specified by the
     * columns of matrix \a V
     * \return Measurement depth
     */
    idx get_measurement_depth(const cmat& V) const {
        bool found = false;
        std::vector<idx> heights(nc_ + nq_, 0);

        cmat all_gates(1, 1);
        all_gates << static_cast<double>(std::hash<std::string>{}("all gates"));
        bool compute_total_depth = false;
        if (V.rows() == 1 && V.cols() == 1)
            if (V == all_gates)
                compute_total_depth = true;

        std::size_t hashV = hash_eigen(V);

        // iterate over all steps in the circuit
        for (auto&& step : *this) {
            // measurements
            if (step.type_ == StepType::MEASUREMENT) {
                MeasureStep measure_step = *step.measurements_ip_;

                if (measure_step.mats_hash_[0] != hashV && !compute_total_depth)
                    continue; // we skip this measurement step

                found = true; // measurement was found in the circuit

                std::vector<idx> target = measure_step.target_;
                idx c_reg = measure_step.c_reg_;

                idx max_height = 0;
                switch (measure_step.measurement_type_) {
                    case MeasureType::NONE:
                    case MeasureType::MEASURE_Z:
                    case MeasureType::MEASURE_Z_MANY:
                    case MeasureType::MEASURE_V:
                    case MeasureType::MEASURE_V_MANY:
                    case MeasureType::MEASURE_Z_ND:
                    case MeasureType::MEASURE_Z_MANY_ND:
                    case MeasureType::MEASURE_V_ND:
                    case MeasureType::MEASURE_V_MANY_ND:
                        // compute the "height" of the to-be-placed measurement
                        if (heights[c_reg] > max_height)
                            max_height = heights[c_reg];
                        for (auto&& i : target)
                            if (heights[nc_ + i] > max_height)
                                max_height = heights[nc_ + i];
                        // apply measurement
                        heights[c_reg] = max_height + 1;
                        for (auto&& i : target)
                            heights[nc_ + i] = max_height + 1;
                        break;
                    case MeasureType::RESET:
                    case MeasureType::RESET_MANY:
                    case MeasureType::DISCARD:
                    case MeasureType::DISCARD_MANY:
                        for (auto&& i : target)
                            if (heights[nc_ + i] > max_height)
                                max_height = heights[nc_ + i];
                        // apply reset/discard
                        for (auto&& i : target)
                            heights[nc_ + i] = max_height + 1;
                        break;
                } // end switch
            }     // if (step.type_ == StepType::MEASUREMENT) }
        }         // end for

        return found ? *std::max_element(std::begin(heights), std::end(heights))
                     : 0;
    }

    /**
     * \brief Quantum circuit description total measurement depth
     *
     * \return Total measurement depth
     */
    idx get_measurement_depth() const {
        cmat all_gates(1, 1);
        all_gates << static_cast<double>(std::hash<std::string>{}("all gates"));
        return get_measurement_depth(all_gates);
    }

    // computes the depth greedily, measuring the "height" (depth) of the
    // "pieces" (gates) placed in a Tetris-like style
    /**
     * \brief Quantum circuit description total depth
     *
     * \return Gate/measurement total depth
     */
    idx get_depth() const { return get_gate_depth() + get_measurement_depth(); }

    /**
     * \brief Quantum circuit description measurement count
     *
     * \param V Orthonormal basis or rank-1 projectors specified by the
     * columns of matrix \a V
     * \return Measurement count
     */
    idx get_measurement_count(const cmat& V) const {
        // EXCEPTION CHECKS

        idx result = 0;
        std::size_t hashV = hash_eigen(V);
        // basis matrix hash not found in the hash table
        try {
            result = measurement_count_.at(hashV);
        } catch (...) {
            return 0;
        }

        return result;
    }

    /**
     * \brief Quantum circuit description total measurement count
     *
     * \return Total measurement count
     */
    idx get_measurement_count() const {
        idx result = 0;

        for (auto&& elem : measurement_count_)
            result += elem.second;

        return result;
    }

    /**
     * \brief Quantum circuit total steps count, i.e., the sum of gate count and
     * measurement count
     *
     * \return Total (gates + measurements) count
     */
    idx get_step_count() const noexcept { return step_types_.size(); }

    /**
     * \brief No-op count
     *
     * \return No-op count
     */
    idx get_nop_count() const {
        return std::count_if(
            std::begin(step_types_), std::end(step_types_),
            [](const StepType& s) { return s == StepType::NOP; });
    }

    /**
     * \brief Quantum circuit resources
     *
     * \return Instance of \a qpp::QCircuit::Resources
     */
    Resources get_resources() const {
        Resources result;
        result.nq = get_nq();
        result.nc = get_nc();
        result.d = get_d();
        result.name = get_name();
        result.step_count = get_step_count();
        result.gate_count = get_gate_count();
        result.gate_depth = get_gate_depth();
        result.measurement_count = get_measurement_count();
        result.measurement_depth = get_measurement_depth();
        result.total_depth = get_depth();

        return result;
    }

    // end getters

    // setters
    /**
     * \brief Sets name for the quantum circuit description
     *
     * \param name Quantum circuit description name
     * \return Reference to the current instance
     */
    QCircuit& set_name(const std::string& name) {
        name_ = name;
        return *this;
    }
    // end setters

    /**
     * \brief Adds \a n additional qudits before qudit \a pos
     *
     * \note Qudits with indexes greater or equal than the newly inserted ones
     * have their indexes automatically incremented
     *
     * \param n Number of qudits
     * \param pos Qudit index
     * \return Reference to the current instance
     */
    QCircuit& add_qudit(idx n, idx pos) {
        // EXCEPTION CHECKS

        if (pos > nq_)
            throw exception::OutOfRange("qpp::QCircuit::add_qudit()", "pos");
        // END EXCEPTION CHECKS

        nq_ += n;

        // update gate indexes
        for (auto& gate : gates_) {
            // update ctrl indexes
            if (is_CTRL(gate)) {
                for (auto& elem : gate.ctrl_) {
                    if (elem >= pos)
                        elem += n;
                }
            }

            // update target indexes
            for (auto& elem : gate.target_) {
                if (elem >= pos)
                    elem += n;
            }
        }

        // update measurement indexes
        for (auto& measurement : measurements_) {
            for (auto& elem : measurement.target_) {
                if (elem >= pos)
                    elem += n;
            }
        }

        // update the destructively measured qudits
        measured_.insert(
            std::next(std::begin(measured_), static_cast<std::ptrdiff_t>(pos)),
            n, false);

        // update the non-destructively measured qudits
        measured_nd_.insert(std::next(std::begin(measured_nd_),
                                      static_cast<std::ptrdiff_t>(pos)),
                            n, false);

        // update (enlarge) the clean qudits vector
        clean_qudits_.insert(std::next(std::begin(clean_qudits_),
                                       static_cast<std::ptrdiff_t>(pos)),
                             n, true);

        return *this;
    }

    /**
     * \brief Adds \a n additional qudits after the last qudit
     *
     * \param n Number of qudits
     * \return Reference to the current instance
     */
    QCircuit& add_qudit(idx n = 1) { return add_qudit(n, nq_); }

    /**
     * \brief Adds \a n additional classical dits before dit \a pos
     *
     * \note Classical dits with indexes greater or equal than the newly
     * inserted ones have their indexes automatically incremented
     *
     * \param n Number of classical dits
     * \param pos Classical dit index
     * \return Reference to the current instance
     */
    QCircuit& add_dit(idx n, idx pos) {
        // EXCEPTION CHECKS

        if (pos > nc_)
            throw exception::OutOfRange("qpp::QCircuit::add_dit()", "pos");
        // END EXCEPTION CHECKS

        nc_ += n;

        // update gate indexes
        for (auto& gate : gates_) {
            // update cctrl indexes
            if (is_cCTRL(gate)) {
                for (auto& elem : gate.ctrl_) {
                    if (elem >= pos)
                        elem += n;
                }
            }
        }

        // update measurement indexes
        for (auto& measurement : measurements_) {
            if (measurement.c_reg_ >= pos)
                measurement.c_reg_ += n;
        }

        // update (enlarge) the clean dits vector
        clean_dits_.insert(std::next(std::begin(clean_dits_),
                                     static_cast<std::ptrdiff_t>(pos)),
                           n, true);

        // update (enlarge) the measurement dits vector
        measurement_dits_.insert(std::next(std::begin(measurement_dits_),
                                           static_cast<std::ptrdiff_t>(pos)),
                                 n, false);

        return *this;
    }

    /**
     * \brief Adds \a n additional classical dits after the last dit
     *
     * \param n Number of classical dits
     * \return Reference to the current instance
     */
    QCircuit& add_dit(idx n = 1) { return add_dit(n, nc_); }

    /**
     * \brief Applies the single qudit gate \a U on single qudit \a i
     *
     * \param U Single qudit quantum gate
     * \param i Qudit index
     * \param name Optional gate name
     * \return Reference to the current instance
     * \return Reference to the current instance
     */
    QCircuit& gate(const cmat& U, idx i, std::string name = {}) {
        // EXCEPTION CHECKS

        std::string context{"Step " + std::to_string(get_step_count())};

        // check valid target
        if (i >= nq_)
            throw exception::OutOfRange("qpp::QCircuit::gate()",
                                        context + ": i");
        // check not measured before
        if (get_measured(i))
            throw exception::QuditAlreadyMeasured("qpp::QCircuit::gate()",
                                                  context + ": i");

        // check square matrix for the gate
        if (!internal::check_square_mat(U))
            throw exception::MatrixNotSquare("qpp::QCircuit::gate()",
                                             context + ": U");
        // check correct dimension
        if (static_cast<idx>(U.rows()) != d_)
            throw exception::MatrixMismatchSubsys("qpp::QCircuit::gate()",
                                                  context + ": U");
        // END EXCEPTION CHECKS

        if (name.empty())
            name = qpp::Gates::get_no_thread_local_instance().get_name(U);
        std::size_t hashU = hash_eigen(U);
        add_hash_(hashU, U);
        gates_.emplace_back(GateType::SINGLE, hashU, std::vector<idx>{},
                            std::vector<idx>{i}, std::vector<idx>{}, name);
        step_types_.emplace_back(StepType::GATE);
        ++gate_count_[hashU];

        clean_qudits_[i] = false;

        return *this;
    }

    /**
     * \brief Applies the two qudit gate \a U on qudits \a i and \a j
     *
     * \param U Two qudit quantum gate
     * \param i Qudit index
     * \param j Qudit index
     * \param name Optional gate name
     * \return Reference to the current instance
     */
    QCircuit& gate(const cmat& U, idx i, idx j, std::string name = {}) {
        // EXCEPTION CHECKS

        std::string context{"Step " + std::to_string(get_step_count())};

        // check valid target
        if (i >= nq_ || j >= nq_ || i == j)
            throw exception::OutOfRange("qpp::QCircuit::gate()",
                                        context + ": i/j");
        if (get_measured(i) || get_measured(j))
            throw exception::QuditAlreadyMeasured("qpp::QCircuit::gate()",
                                                  context + ": i/j");

        // check square matrix for the gate
        if (!internal::check_square_mat(U))
            throw exception::MatrixNotSquare("qpp::QCircuit::gate()",
                                             context + ": U");
        // check correct dimension
        if (static_cast<idx>(U.rows()) != d_ * d_)
            throw exception::MatrixMismatchSubsys("qpp::QCircuit::gate()",
                                                  context + ": U");
        // END EXCEPTION CHECKS

        if (name.empty())
            name = qpp::Gates::get_no_thread_local_instance().get_name(U);
        std::size_t hashU = hash_eigen(U);
        add_hash_(hashU, U);
        gates_.emplace_back(GateType::TWO, hashU, std::vector<idx>{},
                            std::vector<idx>{i, j}, std::vector<idx>{}, name);
        step_types_.emplace_back(StepType::GATE);
        ++gate_count_[hashU];

        clean_qudits_[i] = clean_qudits_[j] = false;

        return *this;
    }

    /**
     * \brief Applies the three qudit gate \a U on qudits \a i, \a j and \a
     * k
     *
     * \param U Three qudit quantum gate
     * \param i Qudit index
     * \param j Qudit index
     * \param k Qudit index
     * \param name Optional gate name
     * \return Reference to the current instance
     */
    QCircuit& gate(const cmat& U, idx i, idx j, idx k, std::string name = {}) {
        // EXCEPTION CHECKS

        std::string context{"Step " + std::to_string(get_step_count())};

        // check valid target
        if (i >= nq_ || j >= nq_ || k >= nq_ || (i == j) || (i == k) ||
            (j == k))
            throw exception::OutOfRange("qpp::QCircuit::gate()",
                                        context + ": i/j/k");
        if (get_measured(i) || get_measured(j) || get_measured(k))
            throw exception::QuditAlreadyMeasured("qpp::QCircuit::gate()",
                                                  context + ": i/j/k");

        // check square matrix for the gate
        if (!internal::check_square_mat(U))
            throw exception::MatrixNotSquare("qpp::QCircuit::gate()",
                                             context + ": U");
        // check correct dimension
        if (static_cast<idx>(U.rows()) != d_ * d_ * d_)
            throw exception::MatrixMismatchSubsys("qpp::QCircuit::gate()",
                                                  context + ": U");
        // END EXCEPTION CHECKS

        if (name.empty())
            name = qpp::Gates::get_no_thread_local_instance().get_name(U);
        std::size_t hashU = hash_eigen(U);
        add_hash_(hashU, U);
        gates_.emplace_back(GateType::THREE, hashU, std::vector<idx>{},
                            std::vector<idx>{i, j, k}, std::vector<idx>{},
                            name);
        step_types_.emplace_back(StepType::GATE);
        ++gate_count_[hashU];

        clean_qudits_[i] = clean_qudits_[j] = clean_qudits_[k] = false;

        return *this;
    }

    /**
     * \brief Applies the single qudit gate \a U on every qudit listed in
     * \a target
     *
     * \param U Single qudit quantum gate
     * \param target Target qudit indexes; the gate \a U is applied on every one
     * of them
     * \param name Optional gate name
     * \return Reference to the current instance
     */
    QCircuit& gate_fan(const cmat& U, const std::vector<idx>& target,
                       std::string name = {}) {
        // EXCEPTION CHECKS

        std::string context{"Step " + std::to_string(get_step_count())};

        // check valid target
        if (target.empty())
            throw exception::ZeroSize("qpp::QCircuit::gate_fan()",
                                      context + ": target");
        for (auto&& elem : target) {
            if (elem >= nq_)
                throw exception::OutOfRange("qpp::QCircuit::gate_fan()",
                                            context + ": target");
            // check target was not measured before
            if (get_measured(elem))
                throw exception::QuditAlreadyMeasured(
                    "qpp::QCircuit::gate_fan()", context + ": target");
        }
        // check no duplicates target
        if (!internal::check_no_duplicates(target))
            throw exception::Duplicates("qpp::QCircuit::gate_fan()",
                                        context + ": target");

        // check square matrix for the gate
        if (!internal::check_square_mat(U))
            throw exception::MatrixNotSquare("qpp::QCircuit::gate_fan()",
                                             context + ": U");
        // check correct dimension
        if (static_cast<idx>(U.rows()) != d_)
            throw exception::MatrixMismatchSubsys("qpp::QCircuit::gate_fan()",
                                                  context + ": U");
        // END EXCEPTION CHECKS

        if (name.empty())
            name = qpp::Gates::get_no_thread_local_instance().get_name(U);
        std::size_t hashU = hash_eigen(U);
        add_hash_(hashU, U);
        gates_.emplace_back(GateType::FAN, hashU, std::vector<idx>{}, target,
                            std::vector<idx>{}, name);
        step_types_.emplace_back(StepType::GATE);
        gate_count_[hashU] += target.size();

        for (auto&& elem : target) {
            clean_qudits_[elem] = false;
        }

        return *this;
    }

    // std::initializer_list overload, avoids ambiguity for 2-element lists,
    // see
    // https://stackoverflow.com/questions/26750039/ambiguity-when-using-initializer-list-as-parameter
    /**
     * \brief Applies the single qudit gate \a U on every qudit listed in
     * \a target
     *
     * \param U Single qudit quantum gate
     * \param target Target qudit indexes; the gate \a U is applied on every
     * one of them
     * \param name Optional gate name
     * \return Reference to the current instance
     */
    QCircuit& gate_fan(const cmat& U, const std::initializer_list<idx>& target,
                       std::string name = {}) {
        return gate_fan(U, std::vector<idx>(target), std::move(name));
    }

    /**
     * \brief Applies the single qudit gate \a U on all of the remaining
     * non-measured qudits
     *
     * \param U Single qudit quantum gate
     * \param name Optional gate name
     * \return Reference to the current instance
     */
    QCircuit& gate_fan(const cmat& U, std::string name = {}) {
        // EXCEPTION CHECKS

        std::string context{"Step " + std::to_string(get_step_count())};

        // check non-empty target
        std::vector<idx> target = get_non_measured();
        if (target.empty())
            throw exception::QuditAlreadyMeasured(
                "qpp::QCircuit::gate_fan()",
                context + ": all qudits have been already measured");
        // END EXCEPTION CHECKS

        if (name.empty())
            name = qpp::Gates::get_no_thread_local_instance().get_name(U);

        return gate_fan(U, target, name);
    }

    /**
     * \brief Jointly applies the multiple-qudit gate \a U on the qudit indexes
     * specified by \a target
     *
     * \param U Multiple qudit quantum gate
     * \param target Subsystem indexes where the gate \a U is applied
     * \param name Optional gate name
     * \return Reference to the current instance
     */
    QCircuit& gate_joint(const cmat& U, const std::vector<idx>& target,
                         std::string name = {}) {
        // EXCEPTION CHECKS

        idx n = static_cast<idx>(target.size());
        idx D = static_cast<idx>(std::llround(std::pow(d_, n)));

        std::string context{"Step " + std::to_string(get_step_count())};

        // check valid target
        if (target.empty())
            throw exception::ZeroSize("qpp::QCircuit::gate_joint()",
                                      context + ": target");
        for (auto&& elem : target) {
            if (elem >= nq_)
                throw exception::OutOfRange("qpp::QCircuit::gate_joint()",
                                            context + ": target");
            // check target was not measured before
            if (get_measured(elem))
                throw exception::QuditAlreadyMeasured(
                    "qpp::QCircuit::gate_joint()", context + ": target");
        }
        // check no duplicates target
        if (!internal::check_no_duplicates(target))
            throw exception::Duplicates("qpp::QCircuit::gate_joint()",
                                        context + ": target");

        // check square matrix for the gate
        if (!internal::check_square_mat(U))
            throw exception::MatrixNotSquare("qpp::QCircuit::gate_joint()",
                                             context + ": U");
        // check correct dimension
        if (static_cast<idx>(U.rows()) != D)
            throw exception::MatrixMismatchSubsys("qpp::QCircuit::gate_joint()",
                                                  context + ": U");
        // END EXCEPTION CHECKS

        if (name.empty())
            name = qpp::Gates::get_no_thread_local_instance().get_name(U);
        std::size_t hashU = hash_eigen(U);
        add_hash_(hashU, U);
        gates_.emplace_back(GateType::JOINT, hashU, std::vector<idx>{}, target,
                            std::vector<idx>{}, name);
        step_types_.emplace_back(StepType::GATE);
        ++gate_count_[hashU];

        for (auto&& elem : target) {
            clean_qudits_[elem] = false;
        }

        return *this;
    }

    /**
     * \brief Applies the quantum Fourier transform (as a series of gates)
     * on the qudit indexes specified by \a target
     *
     * \param target Subsystem indexes where the quantum Fourier transform
     * is applied
     * \param swap Swaps the qubits at the end (true by default)
     * \return Reference to the current instance
     */
    QCircuit& QFT(const std::vector<idx>& target, bool swap = true) {
        // EXCEPTION CHECKS

        std::string context{"Step " + std::to_string(get_step_count())};

        // check valid target
        if (target.empty())
            throw exception::ZeroSize("qpp::QCircuit::QFT()",
                                      context + ": target");
        for (auto&& elem : target) {
            if (elem >= nq_)
                throw exception::OutOfRange("qpp::QCircuit::QFT()",
                                            context + ": target");
            // check target was not measured before
            if (get_measured(elem))
                throw exception::QuditAlreadyMeasured("qpp::QCircuit::QFT()",
                                                      context + ": target");
        }
        // check no duplicates target
        if (!internal::check_no_duplicates(target))
            throw exception::Duplicates("qpp::QCircuit::QFT()",
                                        context + ": target");
        // END EXCEPTION CHECKS

        idx n_subsys = target.size();
        if (d_ == 2) // qubits
        {
            for (idx i = 0; i < n_subsys; ++i) {
                // apply Hadamard on qubit i
                gate(Gates::get_no_thread_local_instance().H, target[i]);
                // apply controlled rotations
                for (idx j = 2; j <= n_subsys - i; ++j) {
                    // construct Rj
                    cmat Rj(2, 2);
                    Rj << 1, 0, 0, std::exp(2.0 * pi * 1_i / std::pow(2, j));
                    CTRL(Rj, target[i + j - 1], target[i], {},
                         "CTRL-R" + std::to_string(j));
                }
            }
            if (swap) {
                // we have the qubits in reversed order, we must swap them
                for (idx i = 0; i < n_subsys / 2; ++i) {
                    gate(Gates::get_no_thread_local_instance().SWAP, target[i],
                         target[n_subsys - i - 1]);
                }
            }

        } else { // qudits
            for (idx i = 0; i < n_subsys; ++i) {
                // apply qudit Fourier on qudit i
                gate(Gates::get_no_thread_local_instance().Fd(d_), target[i],
                     "Fd");
                // apply controlled rotations
                for (idx j = 2; j <= n_subsys - i; ++j) {
                    // construct Rj
                    cmat Rj = cmat::Zero(d_, d_);
                    for (idx m = 0; m < d_; ++m) {
                        Rj(m, m) = std::exp(2.0 * pi * static_cast<double>(m) *
                                            1_i / std::pow(d_, j));
                    }
                    CTRL(Rj, target[i + j - 1], target[i], {},
                         "CTRL-R" + std::to_string(j) + "d");
                }
            }
            if (swap) {
                // we have the qudits in reversed order, we must swap them
                for (idx i = 0; i < n_subsys / 2; ++i) {
                    gate(Gates::get_no_thread_local_instance().SWAPd(d_),
                         target[i], target[n_subsys - i - 1], "SWAPd");
                }
            }
        }

        return *this;
    }

    // std::initializer_list overload, avoids ambiguity for {idx} -> bool
    /**
     * \brief Applies the quantum Fourier transform (as a series of gates)
     * on the qudit indexes specified by \a target
     *
     * \param target Subsystem indexes where the quantum Fourier transform
     * is applied
     * \param swap Swaps the qubits at the end (true by default)
     * \return Reference to the current instance
     */
    QCircuit& QFT(const std::initializer_list<idx>& target, bool swap = true) {
        return QFT(std::vector<idx>(target), swap);
    }

    /**
     * \brief Applies the quantum Fourier transform (as a series of gates)
     * on all of remaining non-measured qudits
     *
     * \param swap Swaps the qubits at the end (true by default)
     * \return Reference to the current instance
     */
    QCircuit& QFT(bool swap = true) { return QFT(get_non_measured(), swap); }

    /**
     * \brief Applies the inverse quantum Fourier transform (as a series of
     * gates) on the qudit indexes specified by \a target
     *
     * \param target Subsystem indexes where the inverse quantum Fourier
     * transform is applied
     * \param swap Swaps the qubits at the end (true by default)
     * \return Reference to the current instance
     */
    QCircuit& TFQ(const std::vector<idx>& target,
                  [[maybe_unused]] bool swap = true) {
        // EXCEPTION CHECKS

        std::string context{"Step " + std::to_string(get_step_count())};

        // check valid target
        if (target.empty())
            throw exception::ZeroSize("qpp::QCircuit::TFQ()",
                                      context + ": target");
        for (auto&& elem : target) {
            if (elem >= nq_)
                throw exception::OutOfRange("qpp::QCircuit::TFQ()",
                                            context + ": target");
            // check target was not measured before
            if (get_measured(elem))
                throw exception::QuditAlreadyMeasured("qpp::QCircuit::TFQ()",
                                                      context + ": target");
        }
        // check no duplicates target
        if (!internal::check_no_duplicates(target))
            throw exception::Duplicates("qpp::QCircuit::TFQ()",
                                        context + ": target");
        // END EXCEPTION CHECKS

        idx n_subsys = target.size();
        if (d_ == 2) // qubits
        {
            if (swap) {
                // we have the qubits in reversed order, we must swap them
                for (idx i = n_subsys / 2; i-- > 0;) {
                    gate(Gates::get_no_thread_local_instance().SWAP, target[i],
                         target[n_subsys - i - 1]);
                }
            }
            for (idx i = n_subsys; i-- > 0;) {
                // apply controlled rotations
                for (idx j = n_subsys - i + 1; j-- > 2;) {
                    // construct Rj
                    cmat Rj(2, 2);
                    Rj << 1, 0, 0, std::exp(-2.0 * pi * 1_i / std::pow(2, j));
                    CTRL(Rj, target[i + j - 1], target[i], {},
                         "CTRL-R" + std::to_string(j) + "+");
                }
                // apply Hadamard on qubit i
                gate(Gates::get_no_thread_local_instance().H, target[i]);
            }
        } else { // qudits
            if (swap) {
                // we have the qudits in reversed order, we must swap them
                for (idx i = n_subsys / 2; i-- > 0;) {
                    gate(Gates::get_no_thread_local_instance().SWAPd(d_),
                         target[i], target[n_subsys - i - 1], "SWAPd");
                }
            }
            for (idx i = n_subsys; i-- > 0;) {
                // apply controlled rotations
                for (idx j = n_subsys - i + 1; j-- > 2;) {
                    // construct Rj
                    cmat Rj = cmat::Zero(d_, d_);
                    for (idx m = 0; m < d_; ++m) {
                        Rj(m, m) = std::exp(-2.0 * pi * static_cast<double>(m) *
                                            1_i / std::pow(d_, j));
                    }
                    CTRL(Rj, target[i + j - 1], target[i], {},
                         "CTRL-R" + std::to_string(j) + "d+");
                }
                // apply qudit Fourier on qudit i
                gate(qpp::adjoint(Gates::get_no_thread_local_instance().Fd(d_)),
                     target[i], "Fd+");
            }
        }

        return *this;
    }

    // std::initializer_list overload, avoids ambiguity for {idx} -> bool
    /**
     * \brief Applies the inverse quantum Fourier transform (as a series of
     * gates) on the qudit indexes specified by \a target
     *
     * \param target Subsystem indexes where the inverse quantum Fourier
     * transform is applied
     * \param swap Swaps the qubits at the end (true by default)
     * \return Reference to the current instance
     */
    QCircuit& TFQ(const std::initializer_list<idx>& target, bool swap = true) {
        return TFQ(std::vector<idx>(target), swap);
    }

    /**
     * \brief Applies the inverse quantum Fourier transform (as a series of
     * gates) on all of remaining non-measured qudits
     *
     * \param swap Swaps the qubits at the end (true by default)
     * \return Reference to the current instance
     */
    QCircuit& TFQ(bool swap = true) { return TFQ(get_non_measured(), swap); }

    // single ctrl single target
    /**
     * \brief Applies the single qudit controlled gate \a U with control
     * qudit \a ctrl and target qudit \a target, i.e., CTRL-U.
     *
     * \param U Single qudit quantum gate
     * \param ctrl Control qudit index
     * \param target Target qudit index
     * \param shift Performs the control as if the \a ctrl qudit state was
     * \f$X\f$-incremented by \a shift
     * \param name Optional gate name
     * \return Reference to the current instance
     */
    QCircuit& CTRL(const cmat& U, idx ctrl, idx target, idx shift = 0,
                   std::string name = {}) {
        // EXCEPTION CHECKS

        std::string context{"Step " + std::to_string(get_step_count())};

        // check valid ctrl and target
        if (ctrl >= nq_ || target >= nq_ || ctrl == target)
            throw exception::OutOfRange("qpp::QCircuit::CTRL()",
                                        context + ": ctrl/target");
        if (get_measured(ctrl) || get_measured(target))
            throw exception::QuditAlreadyMeasured("qpp::QCircuit::CTRL()",
                                                  context + ": ctrl/target");

        // check square matrix for the gate
        if (!internal::check_square_mat(U))
            throw exception::MatrixNotSquare("qpp::QCircuit::CTRL()",
                                             context + ": U");
        // check correct dimension
        if (static_cast<idx>(U.rows()) != d_)
            throw exception::MatrixMismatchSubsys("qpp::QCircuit::CTRL()",
                                                  context + ": U");

        // check shift
        if (shift >= d_)
            throw exception::OutOfRange("qpp::QCircuit::CTRL()",
                                        context + ": shift");
        // END EXCEPTION CHECKS

        if (name.empty()) {
            std::string gate_name =
                qpp::Gates::get_no_thread_local_instance().get_name(U);
            name = gate_name.empty() ? "CTRL" : "CTRL-" + gate_name;
        }
        std::size_t hashU = hash_eigen(U);
        add_hash_(hashU, U);
        gates_.emplace_back(GateType::SINGLE_CTRL_SINGLE_TARGET, hashU,
                            std::vector<idx>{ctrl}, std::vector<idx>{target},
                            std::vector<idx>{shift}, name);
        step_types_.emplace_back(StepType::GATE);
        ++gate_count_[hashU];

        clean_qudits_[ctrl] = false;
        clean_qudits_[target] = false;

        return *this;
    }

    // single ctrl multiple target
    /**
     * \brief Applies the single qudit controlled gate \a U with control
     * qudit \a ctrl on every qudit listed in \a target, i.e., CTRL-U-U-...-U.
     *
     * \param U Single qudit quantum gate
     * \param ctrl Control qudit index
     * \param target Target qudit indexes; the gate \a U is applied on every
     * one of them depending on the values of the control qudits
     * \param shift Performs the control as if the \a ctrl qudit state was
     * \f$X\f$-incremented by \a shift
     * \param name Optional gate name
     * \return Reference to the current instance
     */
    QCircuit& CTRL(const cmat& U, idx ctrl, const std::vector<idx>& target,
                   idx shift = 0, std::string name = {}) {
        // EXCEPTION CHECKS

        std::string context{"Step " + std::to_string(get_step_count())};

        // check valid ctrl
        if (ctrl >= nq_)
            throw exception::OutOfRange("qpp::QCircuit::CTRL()",
                                        context + ": ctrl");
        if (get_measured(ctrl))
            throw exception::QuditAlreadyMeasured("qpp::QCircuit::CTRL()",
                                                  context + ": ctrl");

        // check valid target
        if (target.empty())
            throw exception::ZeroSize("qpp::QCircuit::CTRL()",
                                      context + ": target");
        for (auto&& elem : target) {
            if (elem >= nq_)
                throw exception::OutOfRange("qpp::QCircuit::CTRL()",
                                            context + ": target");
            // check target was not measured before
            if (get_measured(elem))
                throw exception::QuditAlreadyMeasured("qpp::QCircuit::CTRL()",
                                                      context + ": target");
        }
        // check no duplicates target
        if (!internal::check_no_duplicates(target))
            throw exception::Duplicates("qpp::QCircuit::CTRL()",
                                        context + ": target");

        // check ctrl and target don't share common elements
        for (auto&& elem : target)
            if (elem == ctrl)
                throw exception::OutOfRange("qpp::QCircuit::CTRL()",
                                            context + ": ctrl/target");

        // check square matrix for the gate
        if (!internal::check_square_mat(U))
            throw exception::MatrixNotSquare("qpp::QCircuit::CTRL()",
                                             context + ": U");
        // check correct dimension
        if (static_cast<idx>(U.rows()) != d_)
            throw exception::MatrixMismatchSubsys("qpp::QCircuit::CTRL()",
                                                  context + ": U");

        // check shift
        if (shift >= d_)
            throw exception::OutOfRange("qpp::QCircuit::CTRL()",
                                        context + ": shift");
        // END EXCEPTION CHECKS

        if (name.empty()) {
            std::string gate_name =
                qpp::Gates::get_no_thread_local_instance().get_name(U);
            name = gate_name.empty() ? "CTRL" : "CTRL-" + gate_name;
        }
        std::size_t hashU = hash_eigen(U);
        add_hash_(hashU, U);
        gates_.emplace_back(GateType::SINGLE_CTRL_MULTIPLE_TARGET, hashU,
                            std::vector<idx>{ctrl}, target,
                            std::vector<idx>{shift}, name);
        step_types_.emplace_back(StepType::GATE);
        ++gate_count_[hashU];

        clean_qudits_[ctrl] = false;
        for (auto&& elem : target) {
            clean_qudits_[elem] = false;
        }

        return *this;
    }

    // multiple ctrl single target
    /**
     * \brief Applies the single qudit controlled gate \a U with multiple
     * control qudits listed in \a ctrl on the target qudit \a target, i.e.,
     * CTRL-CTRL-...-CTRL-U.
     *
     * \param U Single qudit quantum gate
     * \param ctrl Control qudit indexes
     * \param target Target qudit index
     * \param shift Performs the control as if the \a ctrl qudit states were
     * \f$X\f$-incremented component-wise by \a shift. If non-empty
     * (default), the size of \a shift must be the same as the size of \a
     * ctrl.
     * \param name Optional gate name
     * \return Reference to the current instance
     */
    QCircuit& CTRL(const cmat& U, const std::vector<idx>& ctrl, idx target,
                   const std::vector<idx>& shift = {}, std::string name = {}) {
        // EXCEPTION CHECKS

        std::string context{"Step " + std::to_string(get_step_count())};

        // check valid ctrl
        for (auto&& elem : ctrl) {
            if (elem >= nq_)
                throw exception::OutOfRange("qpp::QCircuit::CTRL()",
                                            context + ": ctrl");
            // check ctrl was not measured before
            if (get_measured(elem))
                throw exception::QuditAlreadyMeasured("qpp::QCircuit::CTRL()",
                                                      context + ": ctrl");
        }
        // check no duplicates ctrl
        if (!internal::check_no_duplicates(ctrl))
            throw exception::Duplicates("qpp::QCircuit::CTRL()",
                                        context + ": ctrl");

        // check valid target
        if (target >= nq_)
            throw exception::OutOfRange("qpp::QCircuit::CTRL()",
                                        context + ": target");
        if (get_measured(target))
            throw exception::QuditAlreadyMeasured("qpp::QCircuit::CTRL()",
                                                  context + ": target");

        // check ctrl and target don't share common elements
        for (auto&& elem : ctrl)
            if (elem == target)
                throw exception::OutOfRange("qpp::QCircuit::CTRL()",
                                            context + ": ctrl/target");

        // check square matrix for the gate
        if (!internal::check_square_mat(U))
            throw exception::MatrixNotSquare("qpp::QCircuit::CTRL()",
                                             context + ": U");
        // check correct dimension
        if (static_cast<idx>(U.rows()) != d_)
            throw exception::MatrixMismatchSubsys("qpp::QCircuit::CTRL()",
                                                  context + ": U");

        // check shift
        if (!shift.empty() && (shift.size() != ctrl.size()))
            throw exception::SizeMismatch("qpp::QCircuit::CTRL()",
                                          context + ": ctrl/shift");
        if (!shift.empty())
            for (auto&& elem : shift)
                if (elem >= d_)
                    throw exception::OutOfRange("qpp::QCircuit::CTRL()",
                                                context + ": shift");
        // END EXCEPTION CHECKS

        if (name.empty()) {
            std::string gate_name =
                qpp::Gates::get_no_thread_local_instance().get_name(U);
            name = gate_name.empty() ? "CTRL" : "CTRL-" + gate_name;
        }
        std::size_t hashU = hash_eigen(U);
        add_hash_(hashU, U);
        gates_.emplace_back(GateType::MULTIPLE_CTRL_SINGLE_TARGET, hashU, ctrl,
                            std::vector<idx>{target}, shift, name);
        step_types_.emplace_back(StepType::GATE);
        ++gate_count_[hashU];

        for (auto&& elem : ctrl) {
            clean_qudits_[elem] = false;
        }
        clean_qudits_[target] = false;

        return *this;
    }

    // multiple ctrl multiple target
    /**
     * \brief Applies the single qudit controlled gate \a U with multiple
     * control qudits listed in \a ctrl on every qudit listed in \a target,
     * i.e., CTRL-CTRL-...-CTRL-U-U-...-U.
     *
     * \param U Single qudit quantum gate
     * \param ctrl Control qudit indexes
     * \param target Target qudit indexes; the gate \a U is applied on every
     * one of them depending on the values of the control qudits
     * \param shift Performs the control as if the \a ctrl qudit states were
     * \f$X\f$-incremented component-wise by \a shift. If non-empty
     * (default), the size of \a shift must be the same as the size of \a
     * ctrl.
     * \param name Optional gate name
     * \return Reference to the current instance
     */
    QCircuit& CTRL(const cmat& U, const std::vector<idx>& ctrl,
                   const std::vector<idx>& target,
                   const std::vector<idx>& shift = {}, std::string name = {}) {
        // EXCEPTION CHECKS

        std::string context{"Step " + std::to_string(get_step_count())};

        // check valid ctrl
        for (auto&& elem : ctrl) {
            if (elem >= nq_)
                throw exception::OutOfRange("qpp::QCircuit::CTRL()",
                                            context + ": ctrl");
            // check ctrl was not measured before
            if (get_measured(elem))
                throw exception::QuditAlreadyMeasured("qpp::QCircuit::CTRL()",
                                                      context + ": ctrl");
        }
        // check no duplicates ctrl
        if (!internal::check_no_duplicates(ctrl))
            throw exception::Duplicates("qpp::QCircuit::CTRL()",
                                        context + ": ctrl");

        // check valid target
        if (target.empty())
            throw exception::ZeroSize("qpp::QCircuit::CTRL()",
                                      context + ": target");
        for (auto&& elem : target) {
            if (elem >= nq_)
                throw exception::OutOfRange("qpp::QCircuit::CTRL()",
                                            context + ": target");
            // check target was not measured before
            if (get_measured(elem))
                throw exception::QuditAlreadyMeasured("qpp::QCircuit::CTRL()",
                                                      context + ": target");
        }
        // check no duplicates target
        if (!internal::check_no_duplicates(target))
            throw exception::Duplicates("qpp::QCircuit::CTRL()",
                                        context + ": target");

        // check ctrl and target don't share common elements
        for (auto&& elem_ctrl : ctrl)
            for (auto&& elem_target : target)
                if (elem_ctrl == elem_target)
                    throw exception::OutOfRange("qpp::QCircuit::CTRL()",
                                                context + ": ctrl/target");

        // check square matrix for the gate
        if (!internal::check_square_mat(U))
            throw exception::MatrixNotSquare("qpp::QCircuit::CTRL()",
                                             context + ": U");
        // check correct dimension
        if (static_cast<idx>(U.rows()) != d_)
            throw exception::MatrixMismatchSubsys("qpp::QCircuit::CTRL()",
                                                  context + ": U");

        // check shift
        if (!shift.empty() && (shift.size() != ctrl.size()))
            throw exception::SizeMismatch("qpp::QCircuit::CTRL()",
                                          context + ": ctrl/shift");
        if (!shift.empty())
            for (auto&& elem : shift)
                if (elem >= d_)
                    throw exception::OutOfRange("qpp::QCircuit::CTRL()",
                                                context + ": shift");

        // END EXCEPTION CHECKS

        if (name.empty()) {
            std::string gate_name =
                qpp::Gates::get_no_thread_local_instance().get_name(U);
            name = gate_name.empty() ? "CTRL" : "CTRL-" + gate_name;
        }
        std::size_t hashU = hash_eigen(U);
        add_hash_(hashU, U);
        gates_.emplace_back(GateType::MULTIPLE_CTRL_MULTIPLE_TARGET, hashU,
                            ctrl, target, shift, name);
        step_types_.emplace_back(StepType::GATE);
        ++gate_count_[hashU];

        for (auto&& elem : ctrl) {
            clean_qudits_[elem] = false;
        }
        for (auto&& elem : target) {
            clean_qudits_[elem] = false;
        }

        return *this;
    }

    // multiple control composed target
    /**
     * \brief Jointly applies the multiple-qudit controlled gate \a U with
     * multiple control qudits listed in \a ctrl on the qudit indexes specified
     * by \a target, i.e., CTRL-CTRL-...-CTRL-U_{joint}.
     *
     * \param U Multiple-qudit quantum gate
     * \param ctrl Control qudit indexes
     * \param target Target qudit indexes where the gate \a U is applied
     * depending on the values of the control qudits
     * \param shift Performs the control as if the \a ctrl qudit states were
     * \f$X\f$-incremented component-wise by \a shift. If non-empty
     * (default), the size of \a shift must be the same as the size of \a
     * ctrl.
     * \param name Optional gate name
     * \return Reference to the current instance
     */
    QCircuit& CTRL_joint(const cmat& U, const std::vector<idx>& ctrl,
                         const std::vector<idx>& target,
                         const std::vector<idx>& shift = {},
                         std::string name = {}) {
        // EXCEPTION CHECKS

        std::string context{"Step " + std::to_string(get_step_count())};

        idx D_target =
            static_cast<idx>(std::llround(std::pow(d_, target.size())));
        // check valid ctrl
        for (auto&& elem : ctrl) {
            if (elem >= nq_)
                throw exception::OutOfRange("qpp::QCircuit::CTRL_joint()",
                                            context + ": ctrl");
            // check ctrl was not measured before
            if (get_measured(elem))
                throw exception::QuditAlreadyMeasured(
                    "qpp::QCircuit::CTRL_joint()", context + ": ctrl");
        }
        // check no duplicates ctrl
        if (!internal::check_no_duplicates(ctrl))
            throw exception::Duplicates("qpp::QCircuit::CTRL_joint()",
                                        context + ": ctrl");

        // check valid target
        if (target.empty())
            throw exception::ZeroSize("qpp::QCircuit::CTRL_joint()",
                                      context + ": target");
        for (auto&& elem : target) {
            if (elem >= nq_)
                throw exception::OutOfRange("qpp::QCircuit::CTRL_joint()",
                                            context + ": target");
            // check target was not measured before
            if (get_measured(elem))
                throw exception::QuditAlreadyMeasured(
                    "qpp::QCircuit::CTRL_joint()", context + ": target");
        }

        // check ctrl and target don't share common elements
        for (auto&& elem_ctrl : ctrl)
            for (auto&& elem_target : target)
                if (elem_ctrl == elem_target)
                    throw exception::OutOfRange("qpp::QCircuit::CTRL_joint()",
                                                context + ": ctrl/target");

        // check square matrix for the gate
        if (!internal::check_square_mat(U))
            throw exception::MatrixNotSquare("qpp::QCircuit::CTRL_joint()",
                                             context + ": U");
        // check correct dimension
        if (static_cast<idx>(U.rows()) != D_target)
            throw exception::MatrixMismatchSubsys("qpp::QCircuit::CTRL_joint()",
                                                  context + ": U");

        // check shift
        if (!shift.empty() && (shift.size() != ctrl.size()))
            throw exception::SizeMismatch("qpp::QCircuit::CTRL_joint()",
                                          context + ": ctrl/shift");
        if (!shift.empty())
            for (auto&& elem : shift)
                if (elem >= d_)
                    throw exception::OutOfRange("qpp::QCircuit::CTRL_joint()",
                                                context + ": shift");
        // END EXCEPTION CHECKS

        if (name.empty()) {
            std::string gate_name =
                qpp::Gates::get_no_thread_local_instance().get_name(U);
            name = gate_name.empty() ? "CTRL" : "CTRL-" + gate_name;
        }
        std::size_t hashU = hash_eigen(U);
        add_hash_(hashU, U);
        gates_.emplace_back(GateType::JOINT_CTRL, hashU, ctrl, target, shift,
                            name);
        step_types_.emplace_back(StepType::GATE);
        ++gate_count_[hashU];

        for (auto&& elem : ctrl) {
            clean_qudits_[elem] = false;
        }
        for (auto&& elem : target) {
            clean_qudits_[elem] = false;
        }

        return *this;
    }

    // single ctrl single target
    /**
     * \brief Applies the single qubit controlled gate \a U with classical
     * control dit \a ctrl and target qudit \a target, i.e., cCTRL-U.
     *
     * \param U Single qudit quantum gate
     * \param ctrl_dit Classical control dit index
     * \param target Target qudit index
     * \param shift Performs the control as if the \a ctrl_dit classical dit
     * was incremented by \a shift
     * \param name Optional gate name
     * \return Reference to the current instance
     */
    QCircuit& cCTRL(const cmat& U, idx ctrl_dit, idx target, idx shift = 0,
                    std::string name = {}) {
        // EXCEPTION CHECKS

        std::string context{"Step " + std::to_string(get_step_count())};

        // check valid ctrl_dit and target
        if (ctrl_dit >= nc_ || target >= nq_)
            throw exception::OutOfRange("qpp::QCircuit::cCTRL()",
                                        context + ": ctrl/target");
        if (get_measured(target))
            throw exception::QuditAlreadyMeasured("qpp::QCircuit::cCTRL()",
                                                  context + ": target");

        // check square matrix for the gate
        if (!internal::check_square_mat(U))
            throw exception::MatrixNotSquare("qpp::QCircuit::cCTRL()",
                                             context + ": U");
        // check correct dimension
        if (static_cast<idx>(U.rows()) != d_)
            throw exception::MatrixMismatchSubsys("qpp::QCircuit::cCTRL()",
                                                  context + ": U");

        // check shift
        if (shift >= d_)
            throw exception::OutOfRange("qpp::QCircuit::cCTRL()",
                                        context + ": shift");
        // END EXCEPTION CHECKS

        if (name.empty()) {
            std::string gate_name =
                qpp::Gates::get_no_thread_local_instance().get_name(U);
            name = gate_name.empty() ? "cCTRL" : "cCTRL-" + gate_name;
        }

        std::size_t hashU = hash_eigen(U);
        add_hash_(hashU, U);
        gates_.emplace_back(GateType::SINGLE_cCTRL_SINGLE_TARGET, hashU,
                            std::vector<idx>{ctrl_dit},
                            std::vector<idx>{target}, std::vector<idx>{shift},
                            name);
        step_types_.emplace_back(StepType::GATE);
        ++gate_count_[hashU];

        clean_dits_[ctrl_dit] = false;
        clean_qudits_[target] = false;

        return *this;
    }

    // single ctrl multiple targets
    /**
     * \brief Applies the single qudit controlled gate \a U with classical
     * control dit \a ctrl on every qudit listed in \a target, i.e.,
     * cCTRL-U-U-...-U.
     *
     * \param U Single qudit quantum gate
     * \param ctrl_dit Classical control dit index
     * \param target Target qudit indexes; the gate \a U is applied on every
     * one of them depending on the values of the classical control dits
     * \param shift Performs the control as if the \a ctrl_dit classical dit
     * was incremented by \a shift
     * \param name Optional gate name
     * \return Reference to the current instance
     */
    QCircuit& cCTRL(const cmat& U, idx ctrl_dit, const std::vector<idx>& target,
                    idx shift = 0, std::string name = {}) {
        // EXCEPTION CHECKS

        std::string context{"Step " + std::to_string(get_step_count())};

        // check valid ctrl_dit
        if (ctrl_dit >= nc_)
            throw exception::OutOfRange("qpp::QCircuit::cCTRL()",
                                        context + ": ctrl_dit");

        // check valid target
        if (target.empty())
            throw exception::ZeroSize("qpp::QCircuit::cCTRL()",
                                      context + ": target");
        for (auto&& elem : target) {
            if (elem >= nq_)
                throw exception::OutOfRange("qpp::QCircuit::cCTRL()",
                                            context + ": target");
            // check target was not measured before
            if (get_measured(elem))
                throw exception::QuditAlreadyMeasured("qpp::QCircuit::cCTRL()",
                                                      context + ": target");
        }
        // check no duplicates target
        if (!internal::check_no_duplicates(target))
            throw exception::Duplicates("qpp::QCircuit::cCTRL()",
                                        context + ": target");

        // check square matrix for the gate
        if (!internal::check_square_mat(U))
            throw exception::MatrixNotSquare("qpp::QCircuit::cCTRL()",
                                             context + ": U");
        // check correct dimension
        if (static_cast<idx>(U.rows()) != d_)
            throw exception::MatrixMismatchSubsys("qpp::QCircuit::cCTRL()",
                                                  context + ": U");

        // check shift
        if (shift >= d_)
            throw exception::OutOfRange("qpp::QCircuit::cCTRL()",
                                        context + ": shift");
        // END EXCEPTION CHECKS

        if (name.empty()) {
            std::string gate_name =
                qpp::Gates::get_no_thread_local_instance().get_name(U);
            name = gate_name.empty() ? "cCTRL" : "cCTRL-" + gate_name;
        }
        std::size_t hashU = hash_eigen(U);
        add_hash_(hashU, U);
        gates_.emplace_back(GateType::SINGLE_cCTRL_MULTIPLE_TARGET, hashU,
                            std::vector<idx>{ctrl_dit}, target,
                            std::vector<idx>{shift}, name);
        step_types_.emplace_back(StepType::GATE);
        ++gate_count_[hashU];

        clean_dits_[ctrl_dit] = false;
        for (auto&& elem : target) {
            clean_qudits_[elem] = false;
        }

        return *this;
    }

    // multiple ctrl single target
    /**
     * \brief Applies the single qudit controlled gate \a U with multiple
     * classical control dits listed in \a ctrl on the target qudit \a
     * target, i.e., cCTRL-cCTRL-...-CTRL-U.
     *
     * \param U Single qudit quantum gate
     * \param ctrl_dits Classical control dits indexes
     * \param target Target qudit index
     * \param shift Performs the control as if the \a ctrl_dits classical
     * dits were incremented component-wise by \a shift. If non-empty
     * (default), the size of \a shift must be the same as the size of \a
     * ctrl_dits.
     * \param name Optional gate name
     * \return Reference to the current instance
     */
    QCircuit& cCTRL(const cmat& U, const std::vector<idx>& ctrl_dits,
                    idx target, const std::vector<idx>& shift = {},
                    std::string name = {}) {
        // EXCEPTION CHECKS

        std::string context{"Step " + std::to_string(get_step_count())};

        // check valid ctrl_dits
        for (auto&& elem : ctrl_dits) {
            if (elem >= nc_)
                throw exception::OutOfRange("qpp::QCircuit::cCTRL()",
                                            context + ": ctrl_dits");
        }
        // check no duplicates ctrl_dits
        if (!internal::check_no_duplicates(ctrl_dits))
            throw exception::Duplicates("qpp::QCircuit::cCTRL()",
                                        context + ": ctrl_dits");

        // check valid target
        if (target >= nq_)
            throw exception::OutOfRange("qpp::QCircuit::cCTRL()",
                                        context + ": target");
        // check target was not measured before
        if (get_measured(target))
            throw exception::QuditAlreadyMeasured("qpp::QCircuit::cCTRL()",
                                                  context + ": target");

        // check square matrix for the gate
        if (!internal::check_square_mat(U))
            throw exception::MatrixNotSquare("qpp::QCircuit::cCTRL()",
                                             context + ": U");
        // check correct dimension
        if (static_cast<idx>(U.rows()) != d_)
            throw exception::MatrixMismatchSubsys("qpp::QCircuit::cCTRL()",
                                                  context + ": U");

        // check shift
        if (!shift.empty() && (shift.size() != ctrl_dits.size()))
            throw exception::SizeMismatch("qpp::QCircuit::cCTRL()",
                                          context + ": ctrl_dits/shift");
        if (!shift.empty())
            for (auto&& elem : shift)
                if (elem >= d_)
                    throw exception::OutOfRange("qpp::QCircuit::cCTRL()",
                                                context + ": shift");
        // END EXCEPTION CHECKS

        if (name.empty()) {
            std::string gate_name =
                qpp::Gates::get_no_thread_local_instance().get_name(U);
            name = gate_name.empty() ? "cCTRL" : "cCTRL-" + gate_name;
        }
        std::size_t hashU = hash_eigen(U);
        add_hash_(hashU, U);
        gates_.emplace_back(GateType::MULTIPLE_cCTRL_SINGLE_TARGET, hashU,
                            ctrl_dits, std::vector<idx>{target}, shift, name);
        step_types_.emplace_back(StepType::GATE);
        ++gate_count_[hashU];

        for (auto&& elem : ctrl_dits) {
            clean_dits_[elem] = false;
        }
        clean_qudits_[target] = false;

        return *this;
    }

    // multiple ctrl multiple targets
    /**
     * \brief Applies the single qudit controlled gate \a U with multiple
     * classical control dits listed in \a ctrl on every qudit listed in
     * \a target, i.e., cCTRL-cCTRL-...-cCTRL-U-U-...-U.
     *
     * \param U Single qudit quantum gate
     * \param ctrl_dits Classical control dits indexes
     * \param target Target qudit indexes; the gate \a U is applied on every
     * one of them depending on the values of the classical control dits
     * \param shift Performs the control as if the \a ctrl_dits classical
     * dits were incremented component-wise by \a shift. If non-empty
     * (default), the size of \a shift must be the same as the size of \a
     * ctrl_dits.
     * \param name Optional gate name
     * \return Reference to the current instance
     */
    QCircuit& cCTRL(const cmat& U, const std::vector<idx>& ctrl_dits,
                    const std::vector<idx>& target,
                    const std::vector<idx>& shift = {}, std::string name = {}) {
        // EXCEPTION CHECKS

        std::string context{"Step " + std::to_string(get_step_count())};

        // check valid ctrl_dits
        for (auto&& elem : ctrl_dits) {
            if (elem >= nc_)
                throw exception::OutOfRange("qpp::QCircuit::cCTRL()",
                                            context + ": ctrl_dits");
        }
        // check no duplicates ctrl_dits
        if (!internal::check_no_duplicates(ctrl_dits))
            throw exception::Duplicates("qpp::QCircuit::cCTRL()",
                                        context + ": ctrl_dits");

        // check valid target
        if (target.empty())
            throw exception::ZeroSize("qpp::QCircuit::cCTRL()",
                                      context + ": target");
        for (auto&& elem : target) {
            if (elem >= nq_)
                throw exception::OutOfRange("qpp::QCircuit::cCTRL()",
                                            context + ": target");
            // check target was not measured before
            if (get_measured(elem))
                throw exception::QuditAlreadyMeasured("qpp::QCircuit::cCTRL()",
                                                      context + ": target");
        }
        // check no duplicates target
        if (!internal::check_no_duplicates(target))
            throw exception::Duplicates("qpp::QCircuit::cCTRL()",
                                        context + ": target");

        // check square matrix for the gate
        if (!internal::check_square_mat(U))
            throw exception::MatrixNotSquare("qpp::QCircuit::cCTRL()",
                                             context + ": U");
        // check correct dimension
        if (static_cast<idx>(U.rows()) != d_)
            throw exception::MatrixMismatchSubsys("qpp::QCircuit::cCTRL()",
                                                  context + ": U");

        // check shift
        if (!shift.empty() && (shift.size() != ctrl_dits.size()))
            throw exception::SizeMismatch("qpp::QCircuit::cCTRL()",
                                          context + ": ctrl_dits/shift");
        if (!shift.empty())
            for (auto&& elem : shift)
                if (elem >= d_)
                    throw exception::OutOfRange("qpp::QCircuit::cCTRL()",
                                                context + ": shift");
        // END EXCEPTION CHECKS

        if (name.empty()) {
            std::string gate_name =
                qpp::Gates::get_no_thread_local_instance().get_name(U);
            name = gate_name.empty() ? "cCTRL" : "cCTRL-" + gate_name;
        }
        std::size_t hashU = hash_eigen(U);
        add_hash_(hashU, U);
        gates_.emplace_back(GateType::MULTIPLE_cCTRL_MULTIPLE_TARGET, hashU,
                            ctrl_dits, std::vector<idx>{target}, shift, name);
        step_types_.emplace_back(StepType::GATE);
        ++gate_count_[hashU];

        for (auto&& elem : ctrl_dits) {
            clean_dits_[elem] = false;
        }
        for (auto&& elem : target) {
            clean_qudits_[elem] = false;
        }

        return *this;
    }

    // multiple classical control composed target
    /**
     * \brief Jointly applies the multiple-qudit controlled gate \a U with
     * multiple classical control dits listed in \a ctrl on the qudit indexes
     * specified by \a target, i.e., cCTRL-cCTRL-...-cCTRL-U_{joint}.
     *
     * \param U Multiple-qudit quantum gate
     * \param ctrl_dits Classical control dits indexes
     * \param target Target qudit indexes where the gate \a U is applied
     * depending on the values of the classical control dits
     * \param shift Performs the control as if the \a ctrl_dits classical
     * dits were incremented component-wise by \a shift. If non-empty (default),
     * the size of \a shift must be the same as the size of \a ctrl_dits.
     * \param name Optional gate name
     * \return Reference to the current instance
     */
    QCircuit& cCTRL_joint(const cmat& U, const std::vector<idx>& ctrl_dits,
                          const std::vector<idx>& target,
                          const std::vector<idx>& shift = {},
                          std::string name = {}) {
        // EXCEPTION CHECKS

        std::string context{"Step " + std::to_string(get_step_count())};

        idx D_target =
            static_cast<idx>(std::llround(std::pow(d_, target.size())));
        // check valid ctrl_dits
        for (auto&& elem : ctrl_dits) {
            if (elem >= nc_)
                throw exception::OutOfRange("qpp::QCircuit::cCTRL_joint()",
                                            context + ": ctrl_dits");
        }
        // check no duplicates ctrl_dits
        if (!internal::check_no_duplicates(ctrl_dits))
            throw exception::Duplicates("qpp::QCircuit::cCTRL_joint()",
                                        context + ": ctrl_dits");

        // check valid target
        if (target.empty())
            throw exception::ZeroSize("qpp::QCircuit::cCTRL_joint()",
                                      context + ": target");
        for (auto&& elem : target) {
            if (elem >= nq_)
                throw exception::OutOfRange("qpp::QCircuit::cCTRL_joint()",
                                            context + ": target");
            // check target was not measured before
            if (get_measured(elem))
                throw exception::QuditAlreadyMeasured(
                    "qpp::QCircuit::cCTRL_joint()", context + ": target");
        }
        // check no duplicates target
        if (!internal::check_no_duplicates(target))
            throw exception::Duplicates("qpp::QCircuit::cCTRL_joint()",
                                        context + ": target");

        // check square matrix for the gate
        if (!internal::check_square_mat(U))
            throw exception::MatrixNotSquare("qpp::QCircuit::cCTRL_joint()",
                                             context + ": U");
        // check correct dimension
        if (static_cast<idx>(U.rows()) != D_target)
            throw exception::MatrixMismatchSubsys(
                "qpp::QCircuit::cCTRL_joint()", context + ": U");

        // check shift
        if (!shift.empty() && (shift.size() != ctrl_dits.size()))
            throw exception::SizeMismatch("qpp::QCircuit::cCTRL()",
                                          context + ": ctrl_dits/shift");
        if (!shift.empty())
            for (auto&& elem : shift)
                if (elem >= d_)
                    throw exception::OutOfRange("qpp::QCircuit::cCTRL()",
                                                context + ": shift");
        // END EXCEPTION CHECKS

        if (name.empty()) {
            std::string gate_name =
                qpp::Gates::get_no_thread_local_instance().get_name(U);
            name = gate_name.empty() ? "cCTRL" : "cCTRL-" + gate_name;
        }
        std::size_t hashU = hash_eigen(U);
        add_hash_(hashU, U);
        gates_.emplace_back(GateType::JOINT_cCTRL, hashU, ctrl_dits, target,
                            shift, name);
        step_types_.emplace_back(StepType::GATE);
        ++gate_count_[hashU];

        for (auto&& elem : ctrl_dits) {
            clean_dits_[elem] = false;
        }
        for (auto&& elem : target) {
            clean_qudits_[elem] = false;
        }

        return *this;
    }

    // Z measurement of single qudit
    /**
     * \brief Measurement of single qudit in the computational basis
     * (Z-basis)
     *
     * \param target Target qudit index that is measured
     * \param c_reg Classical register where the value of the measurement is
     * being stored
     * \param destructive Destructive measurement, true by default
     * \param name Optional measurement name, default is "mZ"
     * \return Reference to the current instance
     */
    QCircuit& measureZ(idx target, idx c_reg, bool destructive = true,
                       std::string name = {}) {
        // EXCEPTION CHECKS

        std::string context{"Step " + std::to_string(get_step_count())};

        // measuring non-existing qudit
        if (target >= nq_)
            throw exception::OutOfRange("qpp::QCircuit::measureZ()",
                                        context + ": target");
        // trying to put the result into a non-existing classical slot
        if (c_reg >= nc_)
            throw exception::OutOfRange("qpp::QCircuit::measureZ()",
                                        context + ": c_reg");
        // qudit was measured before
        if (get_measured(target))
            throw exception::QuditAlreadyMeasured("qpp:QCircuit::measureZ()",
                                                  context + ": target");
        // END EXCEPTION CHECKS

        if (name.empty())
            name = "mZ";

        std::size_t m_hash = hash_eigen(Gates::get_instance().Zd(d_));

        if (destructive) {
            measured_[target] = true;
            measurements_.emplace_back(MeasureType::MEASURE_Z,
                                       std::vector<std::size_t>{m_hash},
                                       std::vector<idx>{target}, c_reg, name);
        } else {
            measured_nd_[target] = true;
            measurements_.emplace_back(MeasureType::MEASURE_Z_ND,
                                       std::vector<std::size_t>{m_hash},
                                       std::vector<idx>{target}, c_reg, name);
        }
        step_types_.emplace_back(StepType::MEASUREMENT);
        ++measurement_count_[m_hash];

        clean_qudits_[target] = false;

        clean_dits_[c_reg] = false;
        measurement_dits_[c_reg] = true;

        return *this;
    }

    // Z measurement of multiple qudits
    /**
     * \brief Measurement of multiple qudits in the computational basis
     * (Z-basis)
     *
     * \param target Target qudit indexes that are measured
     * \param c_reg Classical register where the value of the measurement is
     * being stored, as a decimal representation of the string representing the
     * measurement results, with the most significant dit on the left
     * (corresponding to the first/top qudit that is being measured, i.e.,
     * target[0]); that is, big-endian order.
     * \param destructive Destructive measurement, true by default
     * \param name Optional measurement name, default is "mZ"
     * \return Reference to the current instance
     */
    QCircuit& measureZ(const std::vector<idx>& target, idx c_reg,
                       bool destructive = true, std::string name = {}) {
        // EXCEPTION CHECKS

        std::string context{"Step " + std::to_string(get_step_count())};

        // check valid target
        if (target.empty())
            throw exception::ZeroSize("qpp::QCircuit::measureZ()",
                                      context + ": target");
        for (auto&& elem : target) {
            // measuring non-existing qudit
            if (elem >= nq_)
                throw exception::OutOfRange("qpp::QCircuit::measureZ()",
                                            context + ": target");
            // qudit was measured before
            if (get_measured(elem))
                throw exception::QuditAlreadyMeasured(
                    "qpp:QCircuit::measureZ()", context + ": target");
        }
        // trying to put the result into a non-existing classical slot
        if (c_reg >= nc_)
            throw exception::OutOfRange("qpp::QCircuit::measureZ()",
                                        context + ": c_reg");
        // END EXCEPTION CHECKS

        if (name.empty())
            name = "mZ";

        std::size_t m_hash = hash_eigen(Gates::get_instance().Zd(d_));

        if (destructive) {
            for (auto&& elem : target) {
                measured_[elem] = true;
            }
            measurements_.emplace_back(MeasureType::MEASURE_Z_MANY,
                                       std::vector<std::size_t>{m_hash}, target,
                                       c_reg, name);
        } else {
            for (auto&& elem : target) {
                measured_nd_[elem] = true;
            }
            measurements_.emplace_back(MeasureType::MEASURE_Z_MANY_ND,
                                       std::vector<std::size_t>{m_hash}, target,
                                       c_reg, name);
        }
        step_types_.emplace_back(StepType::MEASUREMENT);
        ++measurement_count_[m_hash];

        for (auto&& elem : target) {
            clean_qudits_[elem] = false;
        }

        clean_dits_[c_reg] = false;
        measurement_dits_[c_reg] = true;

        return *this;
    }

    // measurement of single qudit in the orthonormal basis or rank-1
    // projectors specified by the columns of matrix V
    /**
     * \brief Measurement of single qudit in the orthonormal basis or rank-1
     * projectors specified by the columns of matrix \a V
     *
     * \param V Orthonormal basis or rank-1 projectors specified by the
     * columns of matrix \a V
     * \param target Target qudit index that is measured
     * \param c_reg Classical register where the value of the measurement is
     * stored
     * \param destructive Destructive measurement, true by default
     * \param name Optional measurement name
     * \return Reference to the current instance
     */
    QCircuit& measureV(const cmat& V, idx target, idx c_reg,
                       bool destructive = true, std::string name = {}) {
        // EXCEPTION CHECKS

        std::string context{"Step " + std::to_string(get_step_count())};

        // measuring non-existing qudit
        if (target >= nq_)
            throw exception::OutOfRange("qpp::QCircuit::measureV()",
                                        context + ": target");
        // trying to put the result into a non-existing classical slot
        if (c_reg >= nc_)
            throw exception::OutOfRange("qpp::QCircuit::measureV()",
                                        context + ": c_reg");
        // qudit was measured before
        if (get_measured(target))
            throw exception::QuditAlreadyMeasured("qpp:QCircuit::measureV()",
                                                  context + ": target");
        // END EXCEPTION CHECKS

        if (name.empty())
            name = "m" + qpp::Gates::get_no_thread_local_instance().get_name(V);

        std::size_t hashV = hash_eigen(V);
        add_hash_(hashV, V);

        if (destructive) {
            measured_[target] = true;
            measurements_.emplace_back(MeasureType::MEASURE_V,
                                       std::vector<std::size_t>{hashV},
                                       std::vector<idx>{target}, c_reg, name);
        } else {
            measured_nd_[target] = true;
            measurements_.emplace_back(MeasureType::MEASURE_V_ND,
                                       std::vector<std::size_t>{hashV},
                                       std::vector<idx>{target}, c_reg, name);
        }
        step_types_.emplace_back(StepType::MEASUREMENT);
        ++measurement_count_[hashV];

        clean_qudits_[target] = false;

        clean_dits_[c_reg] = false;
        measurement_dits_[c_reg] = true;

        return *this;
    }

    // measurement of multiple qudits in the orthonormal basis or rank-1
    // projectors specified by the columns of matrix V
    /**
     * \brief Joint measurement of multiple qudits in the orthonormal basis or
     * rank-1 projectors specified by the columns of matrix \a V
     *
     * \param V Orthonormal basis or rank-1 projectors specified by the columns
     * of matrix \a V
     * \param target Target qudit indexes that are jointly measured
     * \param c_reg Classical register where the value of the measurement is
     * stored
     * \param destructive Destructive measurement, true by default
     * \param name Optional measurement name
     * \return Reference to the current instance
     */
    QCircuit& measureV(const cmat& V, const std::vector<idx>& target, idx c_reg,
                       bool destructive = true, std::string name = {}) {
        // EXCEPTION CHECKS

        std::string context{"Step " + std::to_string(get_step_count())};

        // check valid target
        if (target.empty())
            throw exception::ZeroSize("qpp::QCircuit::measureV()",
                                      context + ": target");
        for (auto&& elem : target) {
            if (elem >= nq_)
                throw exception::OutOfRange("qpp::QCircuit::measureV()",
                                            context + ": target");
            // check target was not measured before
            if (get_measured(elem))
                throw exception::QuditAlreadyMeasured(
                    "qpp::QCircuit::measureV()", context + ": target");
        }
        // check no duplicates target
        if (!internal::check_no_duplicates(target))
            throw exception::Duplicates("qpp::QCircuit::measureV()",
                                        context + ": target");

        // trying to put the result into a non-existing classical slot
        if (c_reg >= nc_)
            throw exception::OutOfRange("qpp::QCircuit::measureV()",
                                        context + ": c_reg");
        // qudit was measured before
        for (auto&& elem : target) {
            if (get_measured(elem))
                throw exception::QuditAlreadyMeasured(
                    "qpp::QCircuit::measureV()", context + ": target");
        }
        // END EXCEPTION CHECKS

        if (name.empty())
            name = "m" + qpp::Gates::get_no_thread_local_instance().get_name(V);

        std::size_t hashV = hash_eigen(V);
        add_hash_(hashV, V);

        if (destructive) {
            for (auto&& elem : target) {
                measured_[elem] = true;
            }
            measurements_.emplace_back(MeasureType::MEASURE_V_MANY,
                                       std::vector<std::size_t>{hashV}, target,
                                       c_reg, name);
        } else {
            for (auto&& elem : target) {
                measured_nd_[elem] = true;
            }
            measurements_.emplace_back(MeasureType::MEASURE_V_MANY_ND,
                                       std::vector<std::size_t>{hashV}, target,
                                       c_reg, name);
        }
        step_types_.emplace_back(StepType::MEASUREMENT);
        ++measurement_count_[hashV];

        for (auto&& elem : target) {
            clean_qudits_[elem] = false;
        }

        clean_dits_[c_reg] = false;
        measurement_dits_[c_reg] = true;

        return *this;
    }

    /**
     * \brief Discards single qudit by measuring it destructively in the
     * computational basis (Z-basis) and discarding the measurement result
     *
     * \param target Target qudit index that is discarded
     * \param name Optional discard operation name, default is "discard"
     * \return Reference to the current instance
     */
    QCircuit& discard(idx target, std::string name = {}) {
        // EXCEPTION CHECKS

        std::string context{"Step " + std::to_string(get_step_count())};

        // discarding non-existing qudit
        if (target >= nq_)
            throw exception::OutOfRange("qpp::QCircuit::discard()",
                                        context + ": target");
        // qudit was measured before
        if (get_measured(target))
            throw exception::QuditAlreadyMeasured("qpp:QCircuit::discard()",
                                                  context + ": target");
        // END EXCEPTION CHECKS

        if (name.empty())
            name = "discard";

        std::size_t m_hash = hash_eigen(cmat::Zero(d_, d_));

        measured_[target] = true;
        measurements_.emplace_back(MeasureType::DISCARD,
                                   std::vector<std::size_t>{m_hash},
                                   std::vector<idx>{target}, -1, name);

        step_types_.emplace_back(StepType::MEASUREMENT);
        ++measurement_count_[m_hash];

        clean_qudits_[target] = false;

        return *this;
    }

    /**
     * \brief Discards multiple qudits by measuring them destructively in the
     * computational basis (Z-basis) and discarding the measurement result
     *
     * \param target Target qudit indexes that are discarded
     * \param name Optional discard operation name, default is "discard"
     * \return Reference to the current instance
     */
    QCircuit& discard(const std::vector<idx>& target, std::string name = {}) {
        // EXCEPTION CHECKS

        std::string context{"Step " + std::to_string(get_step_count())};

        // check valid target
        if (target.empty())
            throw exception::ZeroSize("qpp::QCircuit::discard()",
                                      context + ": target");
        for (auto&& elem : target) {
            // discarding non-existing qudit
            if (elem >= nq_)
                throw exception::OutOfRange("qpp::QCircuit::discard()",
                                            context + ": target");
            // qudit was measured before
            if (get_measured(elem))
                throw exception::QuditAlreadyMeasured("qpp:QCircuit::discard()",
                                                      context + ": target");
        }
        // END EXCEPTION CHECKS

        if (name.empty())
            name = "discard";

        std::size_t m_hash = hash_eigen(cmat::Zero(d_, d_));

        for (auto&& elem : target) {
            measured_[elem] = true;
        }
        measurements_.emplace_back(MeasureType::DISCARD_MANY,
                                   std::vector<std::size_t>{m_hash}, target, -1,
                                   name);
        step_types_.emplace_back(StepType::MEASUREMENT);
        ++measurement_count_[m_hash];

        for (auto&& elem : target) {
            clean_qudits_[elem] = false;
        }

        return *this;
    }

    /**
     * \brief No operation (no-op)
     *
     * \note If the underlying step is executed on a noisy engine, then the
     * noise acts before it
     *
     * \return Reference to the current instance
     */
    QCircuit& nop() {
        step_types_.emplace_back(StepType::NOP);

        return *this;
    }

    // reset single qudit
    /**
     * \brief Resets single qudit by first measuring it non-destructively in
     * the computational basis and discarding the measurement result,
     * followed by shifting it back to the \f$|0\rangle\f$ state
     *
     * \param target Target qudit index that is reset
     * \param name Optional reset operation name, default is "reset"
     * \return Reference to the current instance
     */
    QCircuit& reset(idx target, std::string name = {}) {
        // EXCEPTION CHECKS

        std::string context{"Step " + std::to_string(get_step_count())};

        // resetting non-existing qudit
        if (target >= nq_)
            throw exception::OutOfRange("qpp::QCircuit::reset()",
                                        context + ": target");
        // qudit was measured before
        if (get_measured(target))
            throw exception::QuditAlreadyMeasured("qpp:QCircuit::reset()",
                                                  context + ": target");
        // END EXCEPTION CHECKS

        if (name.empty())
            name = "reset";

        std::size_t m_hash = hash_eigen(mprj({0}, d_));

        measurements_.emplace_back(MeasureType::RESET,
                                   std::vector<std::size_t>{m_hash},
                                   std::vector<idx>{target}, -1, name);
        step_types_.emplace_back(StepType::MEASUREMENT);
        ++measurement_count_[m_hash];

        return *this;
    }

    // reset multiple qudits
    /**
     * \brief Resets multiple qudits by first measuring them
     * non-destructively in the computational basis and discarding the
     * measurement results, followed by shifting them back to the
     * \f$|0\cdots 0\rangle\f$ state
     *
     * \param target Target qudit indexes that are reset
     * \param name Optional measurement name, default is "reset"
     * \return Reference to the current instance
     */
    QCircuit& reset(const std::vector<idx>& target, std::string name = {}) {
        // EXCEPTION CHECKS

        std::string context{"Step " + std::to_string(get_step_count())};

        // check valid target
        if (target.empty())
            throw exception::ZeroSize("qpp::QCircuit::reset()",
                                      context + ": target");
        for (auto&& elem : target) {
            // resetting non-existing qudit
            if (elem >= nq_)
                throw exception::OutOfRange("qpp::QCircuit::reset()",
                                            context + ": target");
            // qudit was measured before
            if (get_measured(elem))
                throw exception::QuditAlreadyMeasured("qpp:QCircuit::reset()",
                                                      context + ": target");
        }
        // END EXCEPTION CHECKS

        if (name.empty())
            name = "reset";

        std::size_t m_hash = hash_eigen(mprj({0}, d_));

        measurements_.emplace_back(MeasureType::RESET_MANY,
                                   std::vector<std::size_t>{m_hash}, target, -1,
                                   name);
        step_types_.emplace_back(StepType::MEASUREMENT);
        ++measurement_count_[m_hash];

        return *this;
    }

    /**
     * \brief Replicates the circuit, in place
     * \note The circuit should not contain any measurements when invoking this
     * member function
     *
     * \param n Number of repetitions. If \a n == 1, returns the original
     * circuit.
     * \return Reference to the current instance
     */
    QCircuit& replicate(idx n) {
        // EXCEPTION CHECKS

        if (n == 0)
            throw exception::OutOfRange("qpp::QCircuit::replicate()", "n");
        if (!get_measured().empty())
            throw exception::QuditAlreadyMeasured("qpp::QCircuit::replicate()",
                                                  "n");
        if (n == 1)
            return *this;
        // END EXCEPTION CHECKS

        auto gates_copy = gates_;
        idx gates_size = gates_.size();

        auto step_types_copy = step_types_;
        idx step_types_size = step_types_.size();

        gates_.resize(gates_.size() * n);
        step_types_.resize(step_types_size * n);

        for (idx i = 0; i < n - 1; ++i) {
            std::copy(std::begin(gates_copy), std::end(gates_copy),
                      std::next(std::begin(gates_), static_cast<std::ptrdiff_t>(
                                                        (i + 1) * gates_size)));
            std::copy(std::begin(step_types_copy), std::end(step_types_copy),
                      std::next(std::begin(step_types_),
                                static_cast<std::ptrdiff_t>((i + 1) *
                                                            step_types_size)));
        }

        for (auto& elem : gate_count_)
            elem.second *= n;

        return *this;
    }

    /**
     * \brief Matches a quantum circuit description to the current quantum
     * circuit description, with the to-be-matched quantum circuit description
     * placed at the right (end) of the current quantum circuit description
     * \see qpp::QCircuit::match_circuit_left() and qpp::QCircuit::add_circuit()
     *
     * \note The matched quantum circuit description cannot be larger than the
     * current quantum circuit description, i.e., all qudit indexes of the added
     * quantum circuit description must match with qudits from the current
     * quantum circuit description (and those matched of the latter must contain
     * no measurements)
     *
     * \note The classical dits are not relabeled
     *
     * \param other Quantum circuit description
     * \param target Qudit indexes of the current circuit description where the
     * qudits of \a other are being matched, i.e., the first/top qudit of
     * \a other quantum circuit description is matched with the target[0] qudit
     * of the current circuit description, and so on
     * \param pos_dit The first classical dit of \a other is inserted before
     * the \a pos_dit classical dit index of the current quantum circuit
     * description (in the classical dits array), the rest following in order.
     * By default, insertion is performed at the end.
     * \return Reference to the current instance
     */
    QCircuit& match_circuit_right(QCircuit other,
                                  const std::vector<idx>& target,
                                  idx pos_dit = -1) {
        // EXCEPTION CHECKS

        // check equal dimensions
        if (other.d_ != d_)
            throw exception::DimsNotEqual(
                "qpp::QCircuit::match_circuit_right()", "other");
        // check classical dits
        if (pos_dit == static_cast<idx>(-1))
            pos_dit = nc_;
        else if (pos_dit > nc_)
            throw exception::OutOfRange("qpp::QCircuit::match_circuit_right()",
                                        "pos_dit");
        // check valid target
        if (target.size() != other.nq_)
            throw exception::OutOfRange("qpp::QCircuit::match_circuit_right()",
                                        "target");
        if (target.size() > nq_)
            throw exception::OutOfRange("qpp::QCircuit::match_circuit_right()",
                                        "target");
        if (!internal::check_no_duplicates(target))
            throw exception::Duplicates("qpp::QCircuit::match_circuit_right()",
                                        "target");
        for (auto&& qudit : target) {
            if (qudit >= nq_)
                throw exception::OutOfRange(
                    "qpp::QCircuit::match_circuit_right()", "target");
        }
        // check matching qudits (in the current instance) were not already
        // measured destructively
        for (auto&& qudit : target) {
            if (get_measured(qudit)) {
                throw exception::QuditAlreadyMeasured(
                    "qpp::QCircuit::match_circuit_right()", "target");
            }
        }
        // END EXCEPTION CHECKS

        // STEP 0: insert classical dits from the to-be-matched circuit
        add_dit(other.nc_, pos_dit);

        // STEP 1: update [c]ctrl and target indexes of other
        for (auto& gate : other.gates_) {
            // update the cctrl indexes
            if (is_cCTRL(gate)) {
                for (auto& dit : gate.ctrl_) {
                    dit += pos_dit;
                }
            }
            // update the ctrl indexes
            if (is_CTRL(gate)) {
                for (auto& pos : gate.ctrl_) {
                    pos = target[pos];
                }
            }
            // update the target indexes
            for (auto& pos : gate.target_) {
                pos = target[pos];
            }
        } // end for other.gates_

        // update measurement indexes of other
        for (auto& measurement : other.measurements_) {
            measurement.c_reg_ += pos_dit;
            for (auto& pos : measurement.target_) {
                pos = target[pos];
            }
        } // end for other.measurements_

        // TODO check this

        // replace the corresponding elements of measured_, measured_nd_,
        // clean_qudits_, clean_dits_, and measurement_dits_ with the ones of
        // other
        for (idx i = 0; i < other.measured_.size(); ++i)
            if (other.measured_[i])
                measured_[target[i]] = true;
        for (idx i = 0; i < other.measured_nd_.size(); ++i)
            if (other.measured_nd_[i])
                measured_nd_[target[i]] = true;
        for (idx i = 0; i < other.clean_qudits_.size(); ++i)
            if (!other.clean_qudits_[i])
                clean_qudits_[target[i]] = false;

        for (idx i = 0; i < other.clean_dits_.size(); ++i)
            if (!other.clean_dits_[i])
                clean_dits_[target[i]] = false;
        for (idx i = 0; i < other.measurement_dits_.size(); ++i)
            if (other.measurement_dits_[i])
                measurement_dits_[target[i]] = true;

        // STEP 2: append the copy of other to the current instance
        // append gate steps vector
        gates_.insert(std::end(gates_), std::begin(other.gates_),
                      std::end(other.gates_));

        // append measurement steps vector
        measurements_.insert(std::end(measurements_),
                             std::begin(other.measurements_),
                             std::end(other.measurements_));

        // append step types vector
        step_types_.insert(std::end(step_types_), std::begin(other.step_types_),
                           std::end(other.step_types_));

        // STEP 3: modify gate counts, hash tables etc accordingly
        // update matrix hash table
        for (auto& elem : other.cmat_hash_tbl_)
            cmat_hash_tbl_[elem.first] = elem.second;
        // update gate counts
        for (auto& elem : other.gate_count_)
            gate_count_[elem.first] += elem.second;
        // update measurement counts
        for (auto& elem : other.measurement_count_)
            measurement_count_[elem.first] += elem.second;

        return *this;
    }

    /**
     * \brief Matches a quantum circuit description to the current quantum
     * circuit description, with the to-be-matched quantum circuit description
     * placed at the left (beginning) of the current quantum circuit description
     * \see qpp::QCircuit::match_circuit_right() and
     * qpp::QCircuit::add_circuit()
     *
     * \note The matched quantum circuit description cannot contain measurements
     * and cannot be larger than the current quantum circuit description, i.e.,
     * all qudit indexes of the added quantum circuit description must match
     * with qudits from the current quantum circuit description (and those
     * matched of the latter must contain no measurements)
     * \note The classical dits are not relabeled
     *
     * \param other Quantum circuit description
     * \param target Qudit indexes of the current circuit description where the
     * qudits of \a other are being matched, i.e., the first/top qudit of
     * \a other quantum circuit description is matched with the target[0] qudit
     * of the current circuit description, and so on
     * \param pos_dit The first classical dit of \a other is inserted before
     * the \a pos_dit classical dit index of the current quantum circuit
     * description (in the classical dits array), the rest following in order.
     * By default, insertion is performed at the end.
     * \return Reference to the current instance
     */
    QCircuit& match_circuit_left(QCircuit other, const std::vector<idx>& target,
                                 idx pos_dit = -1) {
        // EXCEPTION CHECKS

        // check equal dimensions
        if (other.d_ != d_)
            throw exception::DimsNotEqual("qpp::QCircuit::match_circuit_left()",
                                          "other");
        // check classical dits
        if (pos_dit == static_cast<idx>(-1))
            pos_dit = nc_;
        else if (pos_dit > nc_)
            throw exception::OutOfRange("qpp::QCircuit::match_circuit_left()",
                                        "pos_dit");
        // check no measurement for the matched circuit
        if (!other.get_measured().empty())
            throw exception::QuditAlreadyMeasured(
                "qpp::QCircuit::match_circuit_left()", "other");
        // check valid target
        if (target.size() != other.nq_)
            throw exception::OutOfRange("qpp::QCircuit::match_circuit_left()",
                                        "target");
        if (target.size() > nq_)
            throw exception::OutOfRange("qpp::QCircuit::match_circuit_left()",
                                        "target");
        if (!internal::check_no_duplicates(target))
            throw exception::Duplicates("qpp::QCircuit::match_circuit_left()",
                                        "target");
        for (auto&& qudit : target) {
            if (qudit >= nq_)
                throw exception::OutOfRange(
                    "qpp::QCircuit::match_circuit_left()", "target");
        }
        // check matching qudits (in the current instance) were not already
        // measured destructively
        for (auto&& qudit : target) {
            if (get_measured(qudit)) {
                throw exception::QuditAlreadyMeasured(
                    "qpp::QCircuit::match_circuit_left()", "target");
            }
        }
        // END EXCEPTION CHECKS

        // STEP 0: insert classical dits from the to-be-matched circuit
        add_dit(other.nc_, pos_dit);

        // STEP 1: update [c]ctrl and target indexes of other
        for (auto& gate : other.gates_) {
            // update the cctrl indexes
            if (is_cCTRL(gate)) {
                for (auto& dit : gate.ctrl_) {
                    dit += pos_dit;
                }
            }
            // update the ctrl indexes
            if (is_CTRL(gate)) {
                for (auto& pos : gate.ctrl_) {
                    pos = target[pos];
                }
            }
            // update the target indexes
            for (auto& pos : gate.target_) {
                pos = target[pos];
            }
        } // end for other.gates_

        // update measurement indexes of other
        for (auto& measurement : other.measurements_) {
            measurement.c_reg_ += pos_dit;
            for (auto& pos : measurement.target_) {
                pos = target[pos];
            }
        } // end for other.measurements_

        // TODO check this

        // replace the corresponding elements of measured_, measured_nd_,
        // clean_qudits_, clean_dits_, and measurement_dits_ with the ones of
        // other
        for (idx i = 0; i < other.measured_.size(); ++i)
            if (other.measured_[i])
                measured_[target[i]] = true;
        for (idx i = 0; i < other.measured_nd_.size(); ++i)
            if (other.measured_nd_[i])
                measured_nd_[target[i]] = true;
        for (idx i = 0; i < other.clean_qudits_.size(); ++i)
            if (!other.clean_qudits_[i])
                clean_qudits_[target[i]] = false;

        for (idx i = 0; i < other.clean_dits_.size(); ++i)
            if (!other.clean_dits_[i])
                clean_dits_[target[i]] = false;
        for (idx i = 0; i < other.measurement_dits_.size(); ++i)
            if (other.measurement_dits_[i])
                measurement_dits_[target[i]] = true;

        // STEP 2: append the copy of other to the current instance
        // append gate steps vector
        gates_.insert(std::begin(gates_), std::begin(other.gates_),
                      std::end(other.gates_));

        // append measurement steps vector
        measurements_.insert(std::begin(measurements_),
                             std::begin(other.measurements_),
                             std::end(other.measurements_));

        // append step types vector
        step_types_.insert(std::begin(step_types_),
                           std::begin(other.step_types_),
                           std::end(other.step_types_));

        // STEP 3: modify gate counts, hash tables etc accordingly
        // update matrix hash table
        for (auto& elem : other.cmat_hash_tbl_)
            cmat_hash_tbl_[elem.first] = elem.second;
        // update gate counts
        for (auto& elem : other.gate_count_)
            gate_count_[elem.first] += elem.second;
        // update measurement counts
        for (auto& elem : other.measurement_count_)
            measurement_count_[elem.first] += elem.second;

        return *this;
    }

    /**
     * \brief Appends (glues) a quantum circuit description to the end of the
     * current one
     * \see qpp::QCircuit::match_circuit_left() and
     * qpp::QCircuit::match_circuit_right()
     *
     * \note If the qudit indexes of the added quantum circuit description
     * do not totally overlap with the indexes of the current quantum
     * circuit description, then the required number of additional qudits
     * are automatically added to the current quantum circuit description
     *
     * \param other Quantum circuit description
     * \param pos_qudit The index of the first/top qudit of \a other quantum
     * circuit description relative to the index of the first/top qudit of the
     * current quantum circuit description, with the rest following in
     * order. If negative or greater than the total number of qudits of the
     * current quantum circuit description, then the required number of
     * additional qudits are automatically added to the current quantum
     * circuit description.
     * \param pos_dit The first classical dit of \a other is inserted before
     * the \a pos_dit classical dit index of the current quantum circuit
     * description (in the classical dits array), the rest following in order.
     * By default, insertion is performed at the end.
     * \return Reference to the current instance
     */
    QCircuit& add_circuit(QCircuit other, bigint pos_qudit, idx pos_dit = -1) {
        // EXCEPTION CHECKS

        // check equal dimensions
        if (other.d_ != d_)
            throw exception::DimsNotEqual("qpp::QCircuit::add_circuit()",
                                          "other");
        // check classical dits
        if (pos_dit == static_cast<idx>(-1))
            pos_dit = nc_;
        else if (pos_dit > nc_)
            throw exception::OutOfRange("qpp::QCircuit::add_circuit()",
                                        "pos_dit");
        // check that overlapping qudits (in the current instance) were not
        // already destructively measured
        if (pos_qudit < 0 &&
            (pos_qudit + static_cast<bigint>(other.nq_)) >= 0) {
            for (idx i = 0;
                 i < std::min(static_cast<idx>(pos_qudit +
                                               static_cast<bigint>(other.nq_)),
                              nq_);
                 ++i)
                if (get_measured(i))
                    throw exception::QuditAlreadyMeasured(
                        "qpp::QCircuit::add_circuit()",
                        "Current qpp::QCircuit instance");
        }
        if (pos_qudit >= 0 && static_cast<idx>(pos_qudit) < nq_) {
            for (idx i = 0;
                 i < std::min(static_cast<idx>(nq_ - pos_qudit), other.nq_);
                 ++i)
                if (get_measured(pos_qudit + i))
                    throw exception::QuditAlreadyMeasured(
                        "qpp::QCircuit::add_circuit()",
                        "Current qpp::QCircuit instance");
        }
        // END EXCEPTION CHECKS

        // STEP 0: add additional qudits (if needed) and classical dits from the
        // to-be-matched circuit
        if (pos_qudit < 0) {
            // add qudits before beginning
            idx extra_qudits = std::abs(pos_qudit);
            add_qudit(extra_qudits, 0);
        } else if (pos_qudit >= 0) {
            // add qudits after beginning
            idx tmp = pos_qudit + other.nq_;
            if (tmp > nq_) {
                idx extra_qudits = tmp - nq_;
                add_qudit(extra_qudits);
            }
        }
        add_dit(other.nc_, pos_dit);

        // STEP 1: update [c]ctrl and target indexes of other
        for (auto& gate : other.gates_) {
            // update the cctrl indexes
            if (is_cCTRL(gate)) {
                for (auto& pos : gate.ctrl_) {
                    pos += pos_dit;
                }
            }
            // update the ctrl indexes
            if (is_CTRL(gate) && pos_qudit >= 0) {
                for (auto& pos : gate.ctrl_) {
                    pos += pos_qudit;
                }
            }

            // update the target indexes
            if (pos_qudit >= 0) {
                for (auto& pos : gate.target_) {
                    pos += pos_qudit;
                }
            }
        } // end for other.gates_

        // update measurement indexes of other
        for (auto& measurement : other.measurements_) {
            measurement.c_reg_ += pos_dit;
            if (pos_qudit >= 0) {
                for (auto& pos : measurement.target_) {
                    pos += pos_qudit;
                }
            }
        } // end for other.measurements_

        // STEP 2
        // replace the corresponding elements of measured_, measured_nd_, and
        // clean_qudits_ with the ones of other
        if (pos_qudit < 0) {
            std::copy_if(std::begin(other.measured_), std::end(other.measured_),
                         std::begin(measured_), [](bool val) { return val; });
            std::copy_if(std::begin(other.measured_nd_),
                         std::end(other.measured_nd_), std::begin(measured_nd_),
                         [](bool val) { return val; });
            std::copy_if(
                std::begin(other.clean_qudits_), std::end(other.clean_qudits_),
                std::begin(clean_qudits_), [](bool val) { return !val; });
        } else {
            std::copy_if(std::begin(other.measured_), std::end(other.measured_),
                         std::next(std::begin(measured_), pos_qudit),
                         [](bool val) { return val; });
            std::copy_if(std::begin(other.measured_nd_),
                         std::end(other.measured_nd_),
                         std::next(std::begin(measured_nd_), pos_qudit),
                         [](bool val) { return val; });
            std::copy_if(std::begin(other.clean_qudits_),
                         std::end(other.clean_qudits_),
                         std::next(std::begin(clean_qudits_), pos_qudit),
                         [](bool val) { return !val; });
        }

        // STEP 3
        // replace the corresponding elements of clean_dits_ and
        // measurement_dits_ with the ones of other
        std::copy(std::begin(other.clean_dits_), std::end(other.clean_dits_),
                  std::next(std::begin(clean_dits_),
                            static_cast<std::ptrdiff_t>(pos_dit)));
        std::copy(std::begin(other.measurement_dits_),
                  std::end(other.measurement_dits_),
                  std::next(std::begin(measurement_dits_),
                            static_cast<std::ptrdiff_t>(pos_dit)));

        // STEP 4: append the copy of other to the current instance
        // append gate steps vector
        gates_.insert(std::end(gates_), std::begin(other.gates_),
                      std::end(other.gates_));

        // append measurement steps vector
        measurements_.insert(std::end(measurements_),
                             std::begin(other.measurements_),
                             std::end(other.measurements_));

        // append step types vector
        step_types_.insert(std::end(step_types_), std::begin(other.step_types_),
                           std::end(other.step_types_));

        // STEP 5: modify gate counts, hash tables etc accordingly
        // update matrix hash table
        for (auto& elem : other.cmat_hash_tbl_)
            cmat_hash_tbl_[elem.first] = elem.second;
        // update gate counts
        for (auto& elem : other.gate_count_)
            gate_count_[elem.first] += elem.second;
        // update measurement counts
        for (auto& elem : other.measurement_count_)
            measurement_count_[elem.first] += elem.second;

        return *this;
    }

    /**
     * \brief Kronecker product with another quantum circuit description, in
     * place
     *
     * \param qc Quantum circuit description
     * \return Reference to the current instance
     */
    QCircuit& kron(QCircuit qc) {
        add_circuit(std::move(qc), static_cast<bigint>(nq_));

        return *this;
    }

    /**
     * \brief Adjoint quantum circuit description, in place
     *
     * \return Reference to the current instance
     */
    QCircuit& adjoint() {
        // EXCEPTION CHECKS

        if (!get_measured().empty())
            throw exception::QuditAlreadyMeasured(
                "qpp::QCircuit::adjoint()", "Current qpp::QCircuit instance");
        // END EXCEPTION CHECKS

        auto htbl = cmat_hash_tbl_; // copy the gate hash table of other
        cmat_hash_tbl_.clear();

        std::reverse(std::begin(gates_), std::end(gates_));
        std::reverse(std::begin(step_types_), std::end(step_types_));

        for (auto& elem : gates_) {
            // get the gate and its corresponding hash
            std::size_t hashU = elem.gate_hash_;
            cmat U = htbl[hashU];

            // compute the adjoints
            cmat Udagger = qpp::adjoint(U);
            std::size_t hashUdagger = hash_eigen(Udagger);

            // modify and add hash
            elem.gate_hash_ = hashUdagger;
            if (!elem.name_.empty())
                elem.name_ += "+";
            add_hash_(hashUdagger, Udagger);
        }

        return *this;
    }

    /**
     * \brief Checks whether a qudit in the circuit was used before or not
     * \see qpp::QCircuit::get_clean_qudits(), qpp::QCircuit::get_dirty_qudits()
     *
     * \param i Qudit index
     * \return True if the qudit \a i was used before (by a gate and/or
     * measurement, either destructive or non-destructive), false otherwise
     */
    bool is_clean_qudit(idx i) const {
        // EXCEPTION CHECKS

        // check valid target
        if (i >= nq_)
            throw exception::OutOfRange("qpp::QCircuit::is_clean_qudit()", "i");
        // END EXCEPTION CHECKS

        return clean_qudits_[i];
    }

    /**
     * \brief Checks whether a classical dit in the circuit was used before
     * or not
     * \see qpp::QCircuit::get_clean_dits(), qpp::QCircuit::get_dirty_dits()
     *
     * \param i Classical dit index
     * \return True if the classical dit \a i was used before (by a cCTRL
     * gate and/or measurement, either destructive or non-destructive), false
     * otherwise
     */
    bool is_clean_dit(idx i) const {
        // EXCEPTION CHECKS

        // check valid target
        if (i >= nc_)
            throw exception::OutOfRange("qpp::QCircuit::is_clean_dit()", "i");
        // END EXCEPTION CHECKS

        return clean_dits_[i];
    }

    /**
     * \brief Checks whether a classical dit in the circuit was used to store
     * the result of a measurement (either destructive or non-destructive)
     * \see qpp::QCircuit::get_measurement_dits()
     *
     * \param i Classical dit index
     * \return True if the classical dit \a i was used before to store the
     * result of a measurement, false otherwise
     */
    bool is_measurement_dit(idx i) const {
        // EXCEPTION CHECKS

        // check valid target
        if (i >= nc_)
            throw exception::OutOfRange("qpp::QCircuit::is_measurement_dit()",
                                        "i");
        // END EXCEPTION CHECKS

        return measurement_dits_[i];
    }

    /**
     * \brief Vector of clean qudits
     * \see qpp::QCircuit::is_clean_qudit(), qpp::QCircuit::get_dirty_qudits()
     *
     * \return Vector of clean qudits
     */
    std::vector<idx> get_clean_qudits() const {
        std::vector<idx> result;
        for (idx i = 0; i < nq_; ++i)
            if (is_clean_qudit(i))
                result.emplace_back(i);

        return result;
    }

    /**
     * \brief Vector of dirty qudits
     * \see qpp::QCircuit::is_clean_qudit(), qpp::QCircuit::get_clean_qudits()
     *
     * \return Vector of dirty qudits
     */
    std::vector<idx> get_dirty_qudits() const {
        return complement(get_clean_qudits(), get_nq());
    }

    /**
     * \brief Vector of clean classical dits
     * \see qpp::QCircuit::is_clean_dit(), qpp::QCircuit::get_dirty_dits()
     *
     * \return Vector of clean classical dits
     */
    std::vector<idx> get_clean_dits() const {
        std::vector<idx> result;
        for (idx i = 0; i < nc_; ++i)
            if (is_clean_dit(i))
                result.emplace_back(i);

        return result;
    }

    /**
     * \brief Vector of dirty classical dits
     * \see qpp::QCircuit::is_clean_dit(), qpp::QCircuit::get_clean_dits()
     *
     * \return Vector of dirty classical dits
     */
    std::vector<idx> get_dirty_dits() const {
        return complement(get_clean_dits(), get_nc());
    }

    /**
     * \brief Removes clean qudit from the quantum circuit description and
     * relabels the rest of the qudits accordingly
     * \see qpp::QCircuit::is_clean_qudit(), qpp::QCircuit::compress()
     *
     * \param target Target clean qudit index that is removed
     * \return Reference to the current instance
     */
    QCircuit& remove_clean_qudit(idx target) {
        // EXCEPTION CHECKS

        // check valid target and clean qudit
        if (target >= nq_ || !is_clean_qudit(target))
            throw exception::OutOfRange("qpp::QCircuit::remove_clean_qudit()",
                                        "target");
        // END EXCEPTION CHECKS

        for (auto&& gate : gates_) {
            if (is_CTRL(gate)) {
                for (idx& pos : gate.ctrl_) {
                    if (pos > target)
                        --pos;
                }
            }
            for (idx& pos : gate.target_) {
                if (pos > target)
                    --pos;
            }
        }

        for (auto&& measurement : measurements_) {
            for (idx& pos : measurement.target_) {
                if (pos > target)
                    --pos;
            }
        }

        clean_qudits_.erase(std::next(std::begin(clean_qudits_),
                                      static_cast<std::ptrdiff_t>(target)));

        --nq_;

        return *this;
    }

    /**
     * \brief Removes clean classical dit from the quantum circuit
     * description and relabels the rest of the classical dits accordingly
     * \see qpp::QCircuit::is_clean_dit(), qpp::QCircuit::compress()
     *
     * \param target Target clean classical dit index that is removed
     * \return Reference to the current instance
     */
    QCircuit& remove_clean_dit(idx target) {
        // EXCEPTION CHECKS

        // check valid target and clean dit
        if (target >= nc_ || !is_clean_dit(target))
            throw exception::OutOfRange("qpp::QCircuit::remove_clean_dit()",
                                        "target");
        // END EXCEPTION CHECKS

        for (auto&& gate : gates_) {
            if (is_cCTRL(gate)) {
                for (idx& pos : gate.ctrl_) {
                    if (pos > target)
                        --pos;
                }
            }
        }

        for (auto&& measurement : measurements_) {
            if (measurement.c_reg_ > target) {
                --measurement.c_reg_;
            }
        }

        clean_dits_.erase(std::next(std::begin(clean_dits_),
                                    static_cast<std::ptrdiff_t>(target)));

        --nc_;

        return *this;
    }

    /**
     * \brief Removes list of clean qudits from the quantum circuit
     * description and relabels the rest of the qudits accordingly
     * \see qpp::QCircuit::is_clean_qudit(), qpp::QCircuit::compress()
     *
     * \param target Target clean qudit indexes that are removed
     * \return Reference to the current instance
     */
    QCircuit& remove_clean_qudits(std::vector<idx> target) {
        // EXCEPTION CHECKS

        // check valid target
        for (auto&& pos : target) {
            // removing non-existing or non-clean qudit
            if (pos >= nq_ || !is_clean_qudit(pos))
                throw exception::OutOfRange(
                    "qpp::QCircuit::remove_clean_qudits()", "target");
        }
        // END EXCEPTION CHECKS

        // sort the target
        std::sort(std::begin(target), std::end(target));
        idx dirty = 0;

        for (auto&& pos : target) {
            remove_clean_qudit(pos - dirty++);
        }

        return *this;
    }

    /**
     * \brief Removes list of clean classical dits from the quantum circuit
     * description and relabels the rest of the classical dits accordingly
     * \see qpp::QCircuit::is_clean_dit(), qpp::QCircuit::compress()
     *
     * \param target Target clean classical dit indexes that are removed
     * \return Reference to the current instance
     */
    QCircuit& remove_clean_dits(std::vector<idx> target) {
        // EXCEPTION CHECKS

        // check valid target
        for (auto&& pos : target) {
            // removing non-existing or non-clean dit
            if (pos >= nc_ || !is_clean_dit(pos))
                throw exception::OutOfRange(
                    "qpp::QCircuit::remove_clean_dits()", "target");
        }
        // END EXCEPTION CHECKS

        // sort the target
        std::sort(std::begin(target), std::end(target));
        idx dirty = 0;

        for (auto&& pos : target) {
            remove_clean_dit(pos - dirty++);
        }

        return *this;
    }

    /**
     * \brief Removes all clean qudits form the quantum circuit description
     * and relabels the rest of the qudits accordingly
     * \see qpp::QCircuit::remove_clean_qudits(),
     * qpp::QCircuit::remove_clean_dits()
     *
     * \param compress_dits If true, removes clean classical dits. Set to false
     * by default.
     * \return Reference to the current instance
     */
    QCircuit& compress(bool compress_dits = false) {
        remove_clean_qudits(get_clean_qudits());
        if (compress_dits)
            remove_clean_dits(get_clean_dits());

        return *this;
    }

    /**
     * \brief Equality operator
     * \note Ignores names (e.g., circuit names, gate names etc.) and does
     * not perform any circuit simplifications, in other words the circuits
     * have to have the exact same number of qubits/classical dits and the
     * exact same gates/measurements placed in the exact same order. For
     * example, the circuit \f$X_1 Z_2\f$ is considered different from
     * \f$Z_2 X_1\f$, although logically they are the same.
     *
     * \param rhs Quantum circuit description against which the equality is
     * being tested
     * \return True if the quantum circuit descriptions are equal, false
     * otherwise
     */
    bool operator==(const QCircuit& rhs) const noexcept {
        return std::tie(rhs.step_types_, rhs.nq_, rhs.nc_, rhs.measurements_,
                        rhs.measured_, rhs.gates_, rhs.d_,
                        rhs.cmat_hash_tbl_) ==
               std::tie(step_types_, nq_, nc_, measurements_, measured_, gates_,
                        d_, cmat_hash_tbl_);
    }

    /**
     * \brief Inequality operator
     *
     * \param rhs Quantum circuit description against which the inequality
     * is being tested
     * \return True if the quantum circuit descriptions are not equal, false
     * otherwise
     */
    bool operator!=(const QCircuit& rhs) const noexcept {
        return !(*this == rhs);
    }

    /**
     * \brief qpp::IJSON::to_JSON() override
     *
     * Displays the quantum circuit description in JSON format
     *
     * \param enclosed_in_curly_brackets If true, encloses the result in
     * curly brackets
     * \return String containing the JSON representation of the quantum circuit
     * description
     */
    std::string to_JSON(bool enclosed_in_curly_brackets = true) const override {
        std::string result;

        if (enclosed_in_curly_brackets)
            result += "{";

        result += R"("name": ")" + name_ + "\", ";

        std::string sep;
        std::ostringstream ss;
        result += "\"steps\": [";
        for (auto&& elem : *this) {
            result += sep;
            sep = ", ";
            result += "{\"step\": " + std::to_string(elem.ip_) + ", ";
            result += "\"type\": ";
            // gate step
            if (elem.type_ == StepType::GATE) {
                idx pos = std::distance(std::begin(elem.value_type_qc_->gates_),
                                        elem.gates_ip_);
                ss.str("");
                ss.clear();
                ss << gates_[pos].gate_type_;
                result += '\"' + ss.str() + "\", ";
                if (!gates_[pos].ctrl_.empty()) {
                    ss.str("");
                    ss.clear();
                    ss << disp(gates_[pos].ctrl_, ", ");
                    if (gates_[pos].gate_type_ >=
                        GateType::SINGLE_cCTRL_SINGLE_TARGET)
                        result += "\"c_ctrl\": " + ss.str() + ", ";
                    else
                        result += "\"ctrl\": " + ss.str() + ", ";
                }
                ss.str("");
                ss.clear();
                ss << disp(gates_[pos].target_, ", ");
                result += "\"target\": " + ss.str() + ", ";

                if (!gates_[pos].shift_.empty()) {
                    ss.str("");
                    ss.clear();
                    ss << disp(gates_[pos].shift_, ", ");
                    result += "\"shift\": " + ss.str() + ", ";
                }

                result += "\"name\": ";
                result += '\"' + gates_[pos].name_ + "\"}";
            }
            // measurement step
            else if (elem.type_ == StepType::MEASUREMENT) {
                idx pos = std::distance(
                    std::begin(elem.value_type_qc_->measurements_),
                    elem.measurements_ip_);
                ss.str("");
                ss.clear();
                ss << measurements_[pos].measurement_type_;
                result += '\"' + ss.str() + "\", ";
                ss.str("");
                ss.clear();
                ss << disp(measurements_[pos].target_, ", ");
                result += "\"target\": " + ss.str() + ", ";

                if (measurements_[pos].measurement_type_ !=
                        MeasureType::RESET &&
                    measurements_[pos].measurement_type_ !=
                        MeasureType::RESET_MANY &&
                    measurements_[pos].measurement_type_ !=
                        MeasureType::DISCARD &&
                    measurements_[pos].measurement_type_ !=
                        MeasureType::DISCARD_MANY)
                    result += "\"c_reg\": " +
                              std::to_string(measurements_[pos].c_reg_) + ", ";

                result += "\"name\": ";
                result += '\"' + measurements_[pos].name_ + "\"}";
            }
            // no-op
            else if (elem.type_ == StepType::NOP) {
                result += std::string{"\"NOP\""} + "}";
            }
            // otherwise
            else {
            }
        }                // end for
        result += "], "; // end steps

        result += get_resources().to_JSON(false) + ", ";

        ss.str("");
        ss.clear();
        ss << disp(get_measured(), ", ");
        result += "\"measured/discarded (destructively)\": " + ss.str() + ", ";

        ss.str("");
        ss.clear();
        ss << disp(get_measured_nd(), ", ");
        result += "\"measured (non-destructive)\": " + ss.str() + ", ";

        ss.str("");
        ss.clear();
        ss << disp(get_measurement_dits(), ", ");
        result += "\"measurement dits\": " + ss.str();

        if (enclosed_in_curly_brackets)
            result += "}";

        return result;
    } /* to_JSON() */

  private:
    /**
     * \brief qpp::IDisplay::display() override
     *
     * Writes to the output stream a textual representation of the quantum
     * circuit
     *
     * \param os Output stream passed by reference
     * \return Reference to the output stream
     */
    std::ostream& display(std::ostream& os) const override {
        os << "[QCircuit ";
        os << "nq: " << nq_ << ", nc: " << nc_ << ", d: " << d_;
        if (!name_.empty())
            os << ", name: \"" << name_ << '"';
        os << "]\n";

        std::string sep{};
        for (auto&& elem : *this) {
            os << sep << elem;
            sep = '\n';
        }

        /* os << "\n$";
            os << "\nmeasured/discarded (destructive): "
               << disp(get_measured(), ", ");
            os << "\nmeasured (non-destructive): " << disp(get_measured_nd(), ",
            "); os << "\nmeasurement dits: " << disp(get_measurement_dits(), ",
            "); */

        return os;
    }
}; /* class QCircuit */

// free functions

/**
 * \brief Appends (glues) a quantum circuit description to another one
 * \see qpp::match_circuit_left() and qpp::match_circuit_right()
 *
 * \note If qudit indexes of the second quantum circuit description do
 * not totally overlap with the indexes of the first quantum circuit
 * description, then the required number of additional qudits are
 * automatically added to the output quantum circuit description
 *
 * \param qc1 Quantum circuit description
 * \param qc2 Quantum circuit description
 * \param pos_qudit The index of the first/top qudit of \a qc2 quantum
 * circuit description relative to the index of the first/top qudit of the
 * \a qc1 quantum circuit description, with the rest following in order.
 * If negative or greater than the total number of qudits of \a qc1,
 * then the required number of additional qudits are automatically added
 * to the output quantum circuit description.
 * \param pos_dit The first classical dit of \a qc2 quantum circuit
 * description is inserted before the \a pos_dit classical dit index of
 * \a qc1 quantum circuit description (in the classical dits array), the
 * rest following in order. By default, insertion is performed at the end.
 * \return Combined quantum circuit description, with \a qc2 added at the
 * end of \a qc1
 */
inline QCircuit add_circuit(QCircuit qc1, const QCircuit& qc2, bigint pos_qudit,
                            idx pos_dit = -1) {
    return qc1.add_circuit(qc2, pos_qudit, pos_dit);
}

/**
 * \brief Adjoint quantum circuit description
 *
 * \param qc Quantum circuit description
 * \return Adjoint quantum circuit description
 */
inline QCircuit adjoint(QCircuit qc) {
    // EXCEPTION CHECKS

    if (!qc.get_measured().empty())
        throw exception::QuditAlreadyMeasured("qpp::adjoint()", "qc");
    // END EXCEPTION CHECKS

    return qc.adjoint();
}

/**
 * \brief Kronecker product between two quantum circuit descriptions
 *
 * \param qc1 Quantum circuit description
 * \param qc2 Quantum circuit description
 * \return Quantum circuit description of the Kronecker product of \a qc1 with
 * \a qc2
 */
inline QCircuit kron(QCircuit qc1, const QCircuit& qc2) {
    return qc1.kron(qc2);
}

/**
 * \brief Matches a quantum circuit description \a qc2 to another quantum
 * circuit description \a qc1, with the \a qc2 quantum circuit description
 * placed at the left (beginning) of the first quantum circuit description
 * \see qpp::match_circuit_right() and qpp::add_circuit()
 *
 * \note The matched quantum circuit description \a qc2 cannot be larger than
 * the \a qc1 quantum circuit description, i.e., all qudit indexes of the added
 * quantum circuit description must match with qudits from the \a qc1 quantum
 * circuit description (and those matched of the latter must contain no
 * measurements)
 *
 * \note The classical dits are not relabeled
 *
 * \param qc1 Quantum circuit description
 * \param qc2 Quantum circuit description
 * \param target Qudit indexes of the \a qc1 circuit description where the
 * qudits of \a qc2 are being matched, i.e., the first/top qudit of
 * \a qc2 quantum circuit description is matched with the target[0] qudit
 * of the \a qc1 circuit description, and so on
 * \param pos_dit The first classical dit of \a qc2 is inserted before
 * the \a pos_dit classical dit index of the \a qc1 quantum circuit
 * description (in the classical dits array), the rest following in order.
 * By default, insertion is performed at the end.
 * \return Combined quantum circuit description
 */
inline QCircuit match_circuit_left(QCircuit qc1, const QCircuit& qc2,
                                   const std::vector<idx>& target,
                                   idx pos_dit = -1) {
    return qc1.match_circuit_left(qc2, target, pos_dit);
}

/**
 * \brief Matches a quantum circuit description \a qc2 to another quantum
 * circuit description \a qc1, with the \a qc2 quantum circuit description
 * placed at the right (end) of the first quantum circuit description
 * \see qpp::match_circuit_right() and qpp::add_circuit()
 * \see qpp::match_circuit_left() and qpp::add_circuit()
 *
 * \note The matched quantum circuit description \a qc 2cannot be larger than
 * the \a qc1 quantum circuit description, i.e., all qudit indexes of the added
 * quantum circuit description must match with qudits from the \a qc1 quantum
 * circuit description (and those matched of the latter must contain no
 * measurements)
 *
 * \note The classical dits are not relabeled
 *
 * \param qc1 Quantum circuit description
 * \param qc2 Quantum circuit description
 * \param target Qudit indexes of the \a qc1 circuit description where the
 * qudits of \a qc2 are being matched, i.e., the first/top qudit of
 * \a qc2 quantum circuit description is matched with the target[0] qudit
 * of the \a qc1 circuit description, and so on
 * \param pos_dit The first classical dit of \a qc2 is inserted before
 * the \a pos_dit classical dit index of the \a qc1 quantum circuit
 * description (in the classical dits array), the rest following in order.
 * By default, insertion is performed at the end.
 * \return Combined quantum circuit description
 */
inline QCircuit match_circuit_right(QCircuit qc1, const QCircuit& qc2,
                                    const std::vector<idx>& target,
                                    idx pos_dit = -1) {
    return qc1.match_circuit_right(qc2, target, pos_dit);
}

/**
 * \brief Replicates a quantum circuit description
 * \note The circuit should not contain any measurements when invoking this
 * function
 *
 * \param qc Quantum circuit description
 * \param n Number of repetitions. If \a n == 1, returns the original circuit.
 * \return Replicated quantum circuit description
 */
inline QCircuit replicate(QCircuit qc, idx n) {
    // EXCEPTION CHECKS

    if (n == 0)
        throw exception::OutOfRange("qpp::replicate()", "n");
    if (!qc.get_measured().empty())
        throw exception::QuditAlreadyMeasured("qpp::replicate()", "qc");
    if (n == 1)
        return qc;
    // END EXCEPTION CHECKS

    return qc.replicate(n);
}

/**
 * \brief Random quantum circuit description generator for fixed gate count
 *
 * \param nq Number of qudits
 * \param num_steps Number of gates (steps) in the circuit
 * \param p_two Probability of applying a two qudit gate. If the two qudit
 * gate set has more than one element, then the gate is chosen at random
 * from the set.
 * \param one_qudit_gate_set Set of one qudit gates (optional, must be specified
 * for \a d > 2)
 * \param two_qudit_gate_set Set of two qudit gates (optional, must be specified
 * for \a d > 2);
 * \param d Subsystem dimensions (optional, default is qubit, i.e., \a d = 2)
 * \param one_qudit_gate_names One qudit gate names (optional)
 * \param two_qudit_gate_names Two qudit gate names (optional)
 * \return Instance of random qpp::QCircuit for fixed gate count
 */
inline QCircuit random_circuit_count(
    idx nq, idx num_steps, double p_two,
    std::vector<cmat> one_qudit_gate_set = {},
    std::vector<cmat> two_qudit_gate_set = {}, idx d = 2,
    const std::vector<std::string>& one_qudit_gate_names = {},
    const std::vector<std::string>& two_qudit_gate_names = {}) {
    // EXCEPTION CHECKS

    // check valid dimension
    if (d < 2)
        throw exception::DimsInvalid("qpp::random_circuit_count()", "d");
    // check valid probabilities
    if (p_two < 0 || p_two > 1)
        throw exception::OutOfRange("qpp::random_circuit_count()", "p_two");
    if (nq < 1 || ((p_two > 0) && (nq == 1)))
        throw exception::OutOfRange("qpp::random_circuit_count()", "nq/p_two");
    // check gate sets are not empty for d > 2
    if (d > 2) {
        if (one_qudit_gate_set.empty())
            throw exception::ZeroSize("qpp::random_circuit_count()",
                                      "one_qudit_gate_set");
        if (p_two > 0 && one_qudit_gate_set.empty())
            throw exception::ZeroSize("qpp::random_circuit_count()",
                                      "two_qudit_gate_set");
    }
    // check gate name sizes
    if (!one_qudit_gate_names.empty() &&
        (one_qudit_gate_names.size() != one_qudit_gate_set.size()))
        throw exception::SizeMismatch("qpp::random_circuit_count()",
                                      "one_qudit_gate_names");
    if (!two_qudit_gate_names.empty() &&
        (two_qudit_gate_names.size() != two_qudit_gate_set.size()))
        throw exception::SizeMismatch("qpp::random_circuit_count()",
                                      "two_qudit_gate_names");
    // check one qudit gate sizes
    for (auto&& gate : one_qudit_gate_set) {
        if (!internal::check_square_mat(gate))
            throw exception::MatrixNotSquare("qpp::random_circuit_count()",
                                             "one_qudit_gate_set");
        if (static_cast<idx>(gate.rows()) != d)
            throw exception::MatrixMismatchSubsys("qpp::random_circuit_count()",
                                                  "one_qudit_gate_set");
    }
    // check two qudit gate sizes
    for (auto&& gate : two_qudit_gate_set) {
        if (!internal::check_square_mat(gate))
            throw exception::MatrixNotSquare("qpp::random_circuit_count()",
                                             "two_qudit_gate_set");
        if (static_cast<idx>(gate.rows()) != d * d)
            throw exception::MatrixMismatchSubsys("qpp::random_circuit_count()",
                                                  "two_qudit_gate_set");
    }
    // END EXCEPTION CHECKS

    if (d == 2 && one_qudit_gate_set.empty())
        one_qudit_gate_set = std::vector{
            Gates::get_instance().X, Gates::get_instance().Y,
            Gates::get_instance().Z, Gates::get_instance().H,
            Gates::get_instance().S, adjoint(Gates::get_instance().S),
            Gates::get_instance().T, adjoint(Gates::get_instance().T)};

    if (d == 2 && two_qudit_gate_set.empty())
        two_qudit_gate_set = std::vector{Gates::get_instance().CNOT};

    double p_one = 1 - p_two;
    QCircuit qc{nq, 0, d};
    for (idx i = 0; i < num_steps; ++i) {
        bool is_one_qudit_gate = bernoulli(p_one);
        if (is_one_qudit_gate) {
            idx q = randidx(0, nq - 1);
            idx gate = randidx(0, one_qudit_gate_set.size() - 1);
            if (one_qudit_gate_names.empty())
                qc.gate(one_qudit_gate_set[gate], q);
            else
                qc.gate(one_qudit_gate_set[gate], q,
                        one_qudit_gate_names[gate]);
        } else {
            idx ctrl = randidx(0, nq - 1);
            idx target = randidx(0, nq - 1);
            while (ctrl == target)
                target = randidx(0, nq - 1);
            idx gate = randidx(0, two_qudit_gate_set.size() - 1);
            if (two_qudit_gate_names.empty())
                qc.gate(two_qudit_gate_set[gate], ctrl, target);
            else
                qc.gate(two_qudit_gate_set[gate], ctrl, target,
                        two_qudit_gate_names[gate]);
        }
    }

    return qc;
}

/**
 * \brief Random quantum circuit description generator for fixed gate depth
 *
 * \param nq Number of qudits
 * \param depth Circuit depth
 * \param p_two Probability of applying a two qudit gate. If the two qudit
 * gate set has more than one element, then the gate is chosen at random
 * from the set.
 * \param gate_depth If non-empty, depth is calculated with respect to this
 * particular gate (optional, empty by default, so by default depth is
 * calculated with respect to all gates in the circuit)
 * \param one_qudit_gate_set Set of one qudit gates (optional, must be specified
 * for \a d > 2)
 * \param two_qudit_gate_set Set of two qudit gates (optional, must be specified
 * for \a d > 2);
 * \param d Subsystem dimensions (optional, default is qubit, i.e., \a d = 2)
 * \param one_qudit_gate_names One qudit gate names (optional)
 * \param two_qudit_gate_names Two qudit gate names (optional)
 * \return Instance of random qpp::QCircuit for fixed circuit gate depth
 */
inline QCircuit random_circuit_depth(
    idx nq, idx depth, double p_two, const cmat& gate_depth = {},
    std::vector<cmat> one_qudit_gate_set = {},
    std::vector<cmat> two_qudit_gate_set = {}, idx d = 2,
    const std::vector<std::string>& one_qudit_gate_names = {},
    const std::vector<std::string>& two_qudit_gate_names = {}) {
    // EXCEPTION CHECKS

    // check valid dimension
    if (d < 2)
        throw exception::DimsInvalid("qpp::random_circuit_depth()", "d");
    // check valid probabilities
    if (p_two < 0 || p_two > 1)
        throw exception::OutOfRange("qpp::random_circuit_depth()", "p_two");
    if (nq < 1 || ((p_two > 0) && (nq == 1)))
        throw exception::OutOfRange("qpp::random_circuit_depth()", "nq/p_two");
    // check gate sets are not empty for d > 2
    if (d > 2) {
        if (one_qudit_gate_set.empty())
            throw exception::ZeroSize("qpp::random_circuit_depth()",
                                      "one_qudit_gate_set");
        if (p_two > 0 && one_qudit_gate_set.empty())
            throw exception::ZeroSize("qpp::random_circuit_depth()",
                                      "two_qudit_gate_set");
    }
    // check gate name sizes
    if (!one_qudit_gate_names.empty() &&
        (one_qudit_gate_names.size() != one_qudit_gate_set.size()))
        throw exception::SizeMismatch("qpp::random_circuit_depth()",
                                      "one_qudit_gate_names");
    if (!two_qudit_gate_names.empty() &&
        (two_qudit_gate_names.size() != two_qudit_gate_set.size()))
        throw exception::SizeMismatch("qpp::random_circuit_depth()",
                                      "two_qudit_gate_names");
    // check one qudit gate sizes
    for (auto&& gate : one_qudit_gate_set) {
        if (!internal::check_square_mat(gate))
            throw exception::MatrixNotSquare("qpp::random_circuit_depth()",
                                             "one_qudit_gate_set");
        if (static_cast<idx>(gate.rows()) != d)
            throw exception::MatrixMismatchSubsys("qpp::random_circuit_depth()",
                                                  "one_qudit_gate_set");
    }
    // check two qudit gate sizes
    for (auto&& gate : two_qudit_gate_set) {
        if (!internal::check_square_mat(gate))
            throw exception::MatrixNotSquare("qpp::random_circuit_depth()",
                                             "two_qudit_gate_set");
        if (static_cast<idx>(gate.rows()) != d * d)
            throw exception::MatrixMismatchSubsys("qpp::random_circuit_depth()",
                                                  "two_qudit_gate_set");
    }
    // END EXCEPTION CHECKS

    if (d == 2 && one_qudit_gate_set.empty())
        one_qudit_gate_set = std::vector{
            Gates::get_instance().X, Gates::get_instance().Y,
            Gates::get_instance().Z, Gates::get_instance().H,
            Gates::get_instance().S, adjoint(Gates::get_instance().S),
            Gates::get_instance().T, adjoint(Gates::get_instance().T)};

    if (d == 2 && two_qudit_gate_set.empty())
        two_qudit_gate_set = std::vector{Gates::get_instance().CNOT};

    double p_one = 1 - p_two;
    QCircuit qc{nq, 0, d};
    idx current_depth = 0;
    bool gate_has_value = gate_depth.size() != 0;
    while (current_depth < depth) {
        bool is_one_qudit_gate = bernoulli(p_one);
        if (is_one_qudit_gate) {
            idx q = randidx(0, nq - 1);
            idx gate = randidx(0, one_qudit_gate_set.size() - 1);
            if (one_qudit_gate_names.empty())
                qc.gate(one_qudit_gate_set[gate], q);
            else
                qc.gate(one_qudit_gate_set[gate], q,
                        one_qudit_gate_names[gate]);
        } else {
            idx ctrl = randidx(0, nq - 1);
            idx target = randidx(0, nq - 1);
            while (ctrl == target)
                target = randidx(0, nq - 1);
            idx gate = randidx(0, two_qudit_gate_set.size() - 1);
            if (two_qudit_gate_names.empty())
                qc.gate(two_qudit_gate_set[gate], ctrl, target);
            else
                qc.gate(two_qudit_gate_set[gate], ctrl, target,
                        two_qudit_gate_names[gate]);
        }
        current_depth = gate_has_value ? qc.get_gate_depth(gate_depth)
                                       : qc.get_gate_depth();
    }

    return qc;
}

} /* namespace qpp */

#endif /* CLASSES_CIRCUITS_CIRCUITS_HPP_ */
