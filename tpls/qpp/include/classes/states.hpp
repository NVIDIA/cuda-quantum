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
 * \file classes/states.hpp
 * \brief Quantum states
 */

#ifndef CLASSES_STATES_HPP_
#define CLASSES_STATES_HPP_

namespace qpp {
/**
 * \class qpp::States
 * \brief const Singleton class that implements most commonly used states
 */
class States final : public internal::Singleton<const States> // const Singleton
{
    friend class internal::Singleton<const States>;

  public:
    // Pauli eigen-states
    ket x0{ket::Zero(2)}; ///< Pauli Sigma-X 0-eigenstate |+>
    ket x1{ket::Zero(2)}; ///< Pauli Sigma-X 1-eigenstate |->
    ket y0{ket::Zero(2)}; ///< Pauli Sigma-Y 0-eigenstate |y+>
    ket y1{ket::Zero(2)}; ///< Pauli Sigma-Y 1-eigenstate |y->
    ket z0{ket::Zero(2)}; ///< Pauli Sigma-Z 0-eigenstate |0>
    ket z1{ket::Zero(2)}; ///< Pauli Sigma-Z 1-eigenstate |1>

    // projectors onto Pauli eigen-states
    cmat px0{cmat::Zero(2, 2)};
    ///< Projector onto the Pauli Sigma-X 0-eigenstate |+><+|
    cmat px1{cmat::Zero(2, 2)};
    ///< Projector onto the Pauli Sigma-X 1-eigenstate |-><-|
    cmat py0{cmat::Zero(2, 2)};
    ///< Projector onto the Pauli Sigma-Y 0-eigenstate |y+><y+|
    cmat py1{cmat::Zero(2, 2)};
    ///< Projector onto the Pauli Sigma-Y 1-eigenstate |y-><y-|
    cmat pz0{cmat::Zero(2, 2)};
    ///< Projector onto the Pauli Sigma-Z 0-eigenstate |0><0|
    cmat pz1{cmat::Zero(2, 2)};
    ///< Projector onto the Pauli Sigma-Z 1-eigenstate |1><1|

    // Bell states
    ket b00{ket::Zero(4)};
    ///< Bell-00 state, as described in Nielsen and Chuang
    ket b01{ket::Zero(4)};
    ///< Bell-01 state, as described in Nielsen and Chuang
    ket b10{ket::Zero(4)};
    ///< Bell-10 state, as described in Nielsen and Chuang
    ket b11{ket::Zero(4)};
    ///< Bell-11 state, as described in Nielsen and Chuang

    // projectors onto Bell states
    cmat pb00{cmat::Zero(4, 4)}; ///< Projector onto the Bell-00 state
    cmat pb01{cmat::Zero(4, 4)}; ///< Projector onto the Bell-01 state
    cmat pb10{cmat::Zero(4, 4)}; ///< Projector onto the Bell-10 state
    cmat pb11{cmat::Zero(4, 4)}; ///< Projector onto the Bell-11 state

    // W and GHZ states
    ket GHZ{ket::Zero(8)}; ///< GHZ state
    ket W{ket::Zero(8)};   ///< W state

    // projectors onto GHZ and W
    cmat pGHZ{cmat::Zero(8, 8)}; ///< Projector onto the GHZ state
    cmat pW{cmat::Zero(8, 8)};   ///< Projector onto the W state

    /**
     * \brief Maximally entangled state of 2 qudits
     *
     * \param d Subsystem dimensions
     * \return Maximally entangled state
     * \f$\frac{1}{\sqrt{d}}\sum_{j=0}^{d-1}|jj\rangle\f$ of 2 qudits
     */
    ket mes(idx d = 2) const {
        // EXCEPTION CHECKS

        // check valid dims
        if (d == 0)
            throw exception::DimsInvalid("qpp::States::mes()", "d");
        // END EXCEPTION CHECKS

        ket psi = mket({0, 0}, {d, d});
        for (idx i = 1; i < d; ++i) {
            psi += mket({i, i}, {d, d});
        }

        return psi / std::sqrt(d);
    }

    /**
     * \brief Zero state of \a n qudits
     * \see qpp::States::one()
     *
     * \param n Positive integer, 1 by default
     * \param d Subsystem dimensions
     * \return Zero state \f$|0\rangle^{\otimes n}\f$ of \a n qudits
     */
    ket zero(idx n = 1, idx d = 2) const {
        // EXCEPTION CHECKS

        // check out of range
        if (n == 0)
            throw exception::OutOfRange("qpp::States::zero()", "n");
        // check valid dims
        if (d == 0)
            throw exception::DimsInvalid("qpp::States::zero()", "d");
        // END EXCEPTION CHECKS

        idx D = static_cast<idx>(std::llround(std::pow(d, n)));
        ket result = ket::Zero(D);
        result(0) = 1;

        return result;
    }

    /**
     * \brief One state of \a n qudits
     * \see qpp::States::zero()
     *
     * \param n Positive integer, 1 by default
     * \param d Subsystem dimensions
     * \return One state \f$|1\rangle^{\otimes n}\f$ of \a n qudits
     */
    ket one(idx n = 1, idx d = 2) const {
        // EXCEPTION CHECKS

        // check out of range
        if (n == 0)
            throw exception::OutOfRange("qpp::States::one()", "n");
        // check valid dims
        if (d == 0)
            throw exception::DimsInvalid("qpp::States::one()", "d");
        // END EXCEPTION CHECKS

        ket result = ket::Zero(static_cast<ket::Index>(std::pow(d, n)));
        result(multiidx2n(std::vector<idx>(n, 1), std::vector<idx>(n, d))) = 1;

        return result;
    }

    /**
     * \brief \f$|j\rangle^{\otimes n}\f$ state of \a n qudits
     * \see qpp::States::j()
     *
     * \param j Non-negative integer
     * \param n Positive integer, 1 by default
     * \param d Subsystem dimensions
     * \return \f$|j\rangle^{\otimes n}\f$ state of \a n qudits
     */
    ket jn(idx j, idx n = 1, idx d = 2) const {
        // EXCEPTION CHECKS

        // check out of range
        if (n == 0)
            throw exception::OutOfRange("qpp::States::jn()", "n");
        // check valid subsystem
        if (j >= d)
            throw exception::SubsysMismatchDims("qpp::States::jn()", "d/j");
        // check valid dims
        if (d == 0)
            throw exception::DimsInvalid("qpp::States::jn()", "d");
        // END EXCEPTION CHECKS

        ket result = ket::Zero(static_cast<ket::Index>(std::pow(d, n)));
        result(multiidx2n(std::vector<idx>(n, j), std::vector<idx>(n, d))) = 1;

        return result;
    }

    /**
     * \brief \f$|j\rangle\f$ computational basis state of a single qudit
     * \see qpp::States::jn()
     *
     * \param j Non-negative integer
     * \param D System dimension
     * \return \f$|j\rangle\f$ computational basis state of a single qudit
     */
    ket j(idx j, idx D = 2) const {
        // EXCEPTION CHECKS

        // check valid subsystem
        if (j >= D)
            throw exception::SubsysMismatchDims("qpp::States::j()", "D/j");
        // check valid dims
        if (D == 0)
            throw exception::DimsInvalid("qpp::States::j()", "D");
        // END EXCEPTION CHECKS

        ket result = ket::Zero(D);
        result(j) = 1;

        return result;
    }

    /**
     * \brief Plus state of \a n qubits
     * \see qpp::States::minus()
     *
     * \param n Positive integer, 1 by default
     * \return Plus state \f$|+\rangle^{\otimes n}\f$ of \a n qubits
     */
    ket plus(idx n = 1) const {
        // EXCEPTION CHECKS

        // check out of range
        if (n == 0)
            throw exception::OutOfRange("qpp::States::plus()", "n");
        // END EXCEPTION CHECKS

        idx D = static_cast<idx>(std::llround(std::pow(2, n)));
        ket result = ket::Ones(D);

        return result / std::sqrt(D);
    }

    /**
     * \brief Minus state of \a n qubits
     * \see qpp::States::plus()
     *
     * \param n Positive integer, 1 by default
     * \return Minus state \f$|-\rangle^{\otimes n}\f$ of \a n qubits
     */
    ket minus(idx n = 1) const {
        // EXCEPTION CHECKS

        // check out of range
        if (n == 0)
            throw exception::OutOfRange("qpp::States::minus()", "n");
        // END EXCEPTION CHECKS

        return kronpow(x1, n);
    }

  private:
    /**
     * Initialize the states
     */
    States() {
        // initialize
        x0 << 1 / std::sqrt(2.), 1 / std::sqrt(2.);
        x1 << 1 / std::sqrt(2.), -1 / std::sqrt(2.);
        y0 << 1 / std::sqrt(2.), 1_i / std::sqrt(2.);
        y1 << 1 / std::sqrt(2.), -1_i / std::sqrt(2.);
        z0 << 1, 0;
        z1 << 0, 1;
        px0 = x0 * x0.adjoint();
        px1 = x1 * x1.adjoint();
        py0 = y0 * y0.adjoint();
        py1 = y1 * y1.adjoint();
        pz0 = z0 * z0.adjoint();
        pz1 = z1 * z1.adjoint();

        // Bell states, as described in Nielsen and Chuang
        // |ij> -> |b_{ij}> by the CNOT*(H x Id) circuit

        b00 << 1 / std::sqrt(2.), 0, 0, 1 / std::sqrt(2.);
        // (|00> + |11>) / sqrt(2)
        b01 << 0, 1 / std::sqrt(2.), 1 / std::sqrt(2.), 0;
        // (|01> + |10>) / sqrt(2)
        b10 << 1 / std::sqrt(2.), 0, 0, -1 / std::sqrt(2.);
        // (|00> - |11>) / sqrt(2)
        b11 << 0, 1 / std::sqrt(2.), -1 / std::sqrt(2.), 0;
        // (|01> - |10>) / sqrt(2)

        pb00 = b00 * b00.adjoint();
        pb01 = b01 * b01.adjoint();
        pb10 = b10 * b10.adjoint();
        pb11 = b11 * b11.adjoint();

        GHZ << 1, 0, 0, 0, 0, 0, 0, 1;
        GHZ = GHZ / std::sqrt(2.);
        W << 0, 1, 1, 0, 1, 0, 0, 0;
        W = W / std::sqrt(3.);

        pGHZ = GHZ * GHZ.adjoint();
        pW = W * W.adjoint();
    }

    /**
     * \brief Default destructor
     */
    ~States() override = default;
}; /* class States */

} /* namespace qpp */

#endif /* CLASSES_STATES_HPP_ */
