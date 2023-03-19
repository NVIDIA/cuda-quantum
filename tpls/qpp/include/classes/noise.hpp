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
 * \file classes/noise.hpp
 * \brief Noise models
 */

#ifndef CLASSES_NOISE_HPP_
#define CLASSES_NOISE_HPP_

namespace qpp {
/**
 * \class qpp::NoiseType
 * \brief Contains template tags used to specify the noise type
 */
struct NoiseType {
    /**
     * \class qpp::NoiseType::StateDependent
     * \brief Template tag, used whenever the noise is state-dependent
     */
    struct StateDependent;

    /**
     * \class qpp::NoiseType::StateIndependent
     * \brief Template tag, used whenever the noise is state-independent
     */
    struct StateIndependent;
};
/**
 * \class qpp::NoiseBase
 * \brief Base class for all noise models, derive your particular noise model
 */
template <class T>
class NoiseBase {
  public:
    using noise_type = T;
    static_assert(std::is_same<NoiseType::StateDependent, noise_type>::value ||
                  std::is_same<NoiseType::StateIndependent, noise_type>::value);

  protected:
    const std::vector<cmat> Ks_;        ///< Kraus operators
    mutable std::vector<double> probs_; ///< probabilities
    mutable idx D_{};                   ///< qudit dimension

    mutable idx i_{}; ///< index of the last occurring noise element
    mutable bool generated_{false}; ///< set to true after compute_state_() is
                                    ///< invoked, or if the noise is
                                    ///< state-independent

    /**
     * \brief Compute probability outcomes for StateDependent noise type,
     * otherwise returns without performing any operation (no-op)
     *
     * \param state State vector or density matrix
     * \param target Target qudit indexes where the noise is applied
     * \param caller Optional caller name
     */
    void compute_probs_(const cmat& state, const std::vector<idx>& target,
                        const std::string& caller = {}) const {
        if (!std::is_same<NoiseType::StateDependent, noise_type>::value)
            return; // no-op

        // minimal EXCEPTION CHECKS

        if (!internal::check_nonzero_size(state))
            throw exception::ZeroSize(caller,
                                      "qpp::NoiseBase::compute_probs_()/state");
        // END EXCEPTION CHECKS

        cmat rho_i;
        idx n = internal::get_num_subsys(state.rows(), D_);

        for (idx i = 0; i < Ks_.size(); ++i) {
            rho_i = ptrace(state, complement(target, n), D_);
            probs_[i] = trace(Ks_[i] * rho_i * adjoint(Ks_[i])).real();
        }
    } /* compute_probs_() */

    /**
     * \brief Compute the resulting state after the noise was applied
     *
     * \param state State vector or density matrix
     * \param target Target qudit indexes where the noise is applied
     * \param caller Optional caller name
     * \return Resulting state after the noise was applied
     */
    cmat compute_state_(const cmat& state, const std::vector<idx>& target,
                        const std::string& caller = {}) const {
        cmat result;
        idx D = static_cast<idx>(state.rows());

        //************ ket ************//
        if (internal::check_cvector(state)) {
            result.resize(D, 1);
        }
        //************ density matrix ************//
        else if (internal::check_square_mat(state)) {
            result.resize(D, D);
        }
        //************ Exception: not ket nor density matrix ************//
        else
            throw exception::MatrixNotSquareNorCvector(
                caller, "qpp::NoiseBase::compute_state_()/state");

        // now do the actual noise generation
        assert(probs_ != decltype(probs_)(probs_.size(), 0)); // not all zeros
        std::discrete_distribution<idx> dd{std::begin(probs_),
                                           std::end(probs_)};
        auto& gen = RandomDevices::get_instance().get_prng();
        i_ = dd(gen);
        result = apply(state, Ks_[i_], target, D_);
        generated_ = true;

        return normalize(result);
    } /* compute_state_() */

  public:
    /**
     * \brief Constructs a noise instance for StateDependent noise type
     *
     * \note SFINAEd-out for StateIndependent noise
     *
     * \param Ks Vector of noise (Kraus) operators that specify the noise
     */
    template <typename U = noise_type>
    explicit NoiseBase(
        const std::vector<cmat>& Ks,
        typename std::enable_if<
            std::is_same<NoiseType::StateDependent, U>::value>::type* = nullptr)
        : Ks_{Ks}, probs_(Ks.size()) {
        // EXCEPTION CHECKS

        if (Ks.empty())
            throw exception::ZeroSize("qpp::NoiseBase::NoiseBase()", "Ks");
        if (!internal::check_nonzero_size(Ks[0]))
            throw exception::ZeroSize("qpp::NoiseBase::NoiseBase()", "Ks[0]");
        if (!internal::check_square_mat(Ks[0]))
            throw exception::MatrixNotSquare("qpp::NoiseBase::NoiseBase()",
                                             "Ks[0]");
        for (auto&& elem : Ks)
            if (elem.rows() != Ks[0].rows() || elem.cols() != Ks[0].rows())
                throw exception::DimsNotEqual("qpp::NoiseBase::NoiseBase()",
                                              "K");
        // END EXCEPTION CHECKS

        D_ = Ks[0].rows(); // set the local dimension
    }

    /**
     * \brief Constructs a noise instance for StateIndependent noise type
     *
     * \note SFINAEd-out for StateDependent noise
     *
     * \param Ks Vector of noise (Kraus) operators that specify the noise
     * \param probs Vector of probabilities corresponding to each Kraus operator
     */
    template <typename U = noise_type>
    explicit NoiseBase(
        const std::vector<cmat>& Ks, const std::vector<double>& probs,
        typename std::enable_if<std::is_same<NoiseType::StateIndependent,
                                             U>::value>::type* = nullptr)
        : Ks_{Ks}, probs_(probs) {
        // EXCEPTION CHECKS

        if (Ks.empty())
            throw exception::ZeroSize("qpp::NoiseBase::NoiseBase()", "Ks");
        if (Ks.size() != probs.size())
            throw exception::SizeMismatch("qpp::NoiseBase::NoiseBase",
                                          "Ks/probs");
        if (!internal::check_nonzero_size(Ks[0]))
            throw exception::ZeroSize("qpp::NoiseBase::NoiseBase()", "Ks[0]");
        if (!internal::check_square_mat(Ks[0]))
            throw exception::MatrixNotSquare("qpp::NoiseBase::NoiseBase()",
                                             "Ks[0]");
        for (auto&& elem : Ks)
            if (elem.rows() != Ks[0].rows() || elem.cols() != Ks[0].rows())
                throw exception::DimsNotEqual("qpp::NoiseBase::NoiseBase()",
                                              "K");
        for (auto&& elem : probs)
            if (elem < 0 || elem > 1)
                throw exception::OutOfRange("qpp::NoiseBase::NoiseBase",
                                            "probs");
        // END EXCEPTION CHECKS

        D_ = Ks[0].rows(); // set the local dimension
        probs_ = probs;
    }

    /**
     * \brief Default virtual destructor
     */
    virtual ~NoiseBase() = default;

    // getters
    /**
     * \brief Qudit dimension
     *
     * \return Qudit dimension
     */
    idx get_d() const noexcept { return D_; };

    /**
     * \brief Vector of noise operators
     *
     * \return Vector of noise operators
     */
    std::vector<cmat> get_Ks() const { return Ks_; }

    /**
     * \brief Vector of probabilities corresponding to each noise operator
     *
     * \return Probability vector
     */
    std::vector<double> get_probs() const {
        if (generated_ ||
            std::is_same<NoiseType::StateIndependent, noise_type>::value) {
            return probs_;
        } else
            throw exception::CustomException(
                "qpp::NoiseBase::get_probs()",
                "NoiseBase::operator() was not yet invoked");
    }

    /**
     * \brief Index of the last occurring noise element
     *
     * \return Index of the last occurring noise element
     */
    idx get_last_idx() const {
        if (generated_) {
            return i_;
        } else
            throw exception::CustomException(
                "qpp::NoiseBase::get_last_idx()",
                "NoiseBase::operator() was not yet invoked");
    }

    /**
     * \brief Probability of the last occurring noise element
     *
     * \return Probability of the last occurring noise element
     */
    double get_last_p() const {
        if (generated_) {
            return probs_[i_];
        } else
            throw exception::CustomException(
                "qpp::NoiseBase::get_last_p()",
                "NoiseBase::operator() was not yet invoked");
    }

    /**
     * \brief Last occurring noise element
     *
     * \return Last occurring noise element
     */
    cmat get_last_K() const {
        if (generated_) {
            return Ks_[i_];
        } else
            throw exception::CustomException(
                "qpp::NoiseBase::get_last_K()",
                "NoiseBase::operator() was not yet invoked");
    }
    // end getters

    /**
     * \brief Function invocation operator, applies the underlying noise model
     * on the state vector or density matrix \a state
     *
     * \param state State vector or density matrix
     * \return Resulting state vector or density matrix
     */
    virtual cmat operator()(const cmat& state) const {
        cmat result;
        compute_probs_(state, std::vector<idx>{0},
                       "qpp::NoiseBase::operator()");
        result = compute_state_(state, std::vector<idx>{0},
                                "qpp::NoiseBase::operator()");

        return result;
    }

    /**
     * \brief Function invocation operator, applies the underlying noise
     * model on qudit \a target of the multi-partite state vector or density
     * matrix \a state
     *
     * \param state Multi-partite state vector or density matrix
     * \param target Target qudit index where the noise is applied
     * \return Resulting state vector or density matrix
     */
    virtual cmat operator()(const cmat& state, idx target) const {
        cmat result;
        compute_probs_(state, std::vector<idx>{target},
                       "qpp::NoiseBase::operator()");
        result = compute_state_(state, std::vector<idx>{target},
                                "qpp::NoiseBase::operator()");

        return result;
    }

    /**
     * \brief Function invocation operator, applies the underlying correlated
     * noise model on qudits specified by \a target of the multi-partite state
     * vector or density matrix \a state
     *
     * \param state Multi-partite state vector or density matrix
     * \param target Target qudit indexes where the correlated noise is applied
     * \return Resulting state vector or density matrix
     */
    virtual cmat operator()(const cmat& state,
                            const std::vector<idx>& target) const {
        cmat result;
        compute_probs_(state, target, "qpp::NoiseBase::operator()");
        result = compute_state_(state, target, "qpp::NoiseBase::operator()");

        return result;
    }
}; /* class NoiseBase */

// qubit noise models

/**
 * \class qpp::QubitDepolarizingNoise
 * \brief Qubit depolarizing noise
 */
class QubitDepolarizingNoise : public NoiseBase<NoiseType::StateIndependent> {
  public:
    /**
     * \brief Qubit depolarizing noise constructor
     *
     * \param p Noise probability
     */
    explicit QubitDepolarizingNoise(double p)
        : NoiseBase({Gates::get_no_thread_local_instance().Id2,
                     Gates::get_no_thread_local_instance().X,
                     Gates::get_no_thread_local_instance().Y,
                     Gates::get_no_thread_local_instance().Z},
                    {1 - p, p / 3, p / 3, p / 3}) {
        // EXCEPTION CHECKS

        if (p < 0 || p > 1)
            throw exception::OutOfRange(
                "qpp::QubitDepolarizingNoise::QubitDepolarizingNoise()", "p");
        // END EXCEPTION CHECKS
    }
}; /* class QubitDepolarizingNoise */

/**
 * \class qpp::QubitPhaseFlipNoise
 * \brief Qubit phase flip (dephasing) noise
 */
class QubitPhaseFlipNoise : public NoiseBase<NoiseType::StateIndependent> {
  public:
    /**
     * \brief Qubit phase flip (dephasing) noise constructor
     *
     * \param p Noise probability
     */
    explicit QubitPhaseFlipNoise(double p)
        : NoiseBase({Gates::get_no_thread_local_instance().Id2,
                     Gates::get_no_thread_local_instance().Z},
                    {1 - p, p}) {
        // EXCEPTION CHECKS

        if (p < 0 || p > 1)
            throw exception::OutOfRange(
                "qpp::QubitPhaseFlipNoise::QubitPhaseFlipNoise()", "p");
        // END EXCEPTION CHECKS
    }
}; /* class QubitPhaseFlipNoise */

/**
 * \class qpp::QubitBitFlipNoise
 * \brief Qubit bit flip noise
 */
class QubitBitFlipNoise : public NoiseBase<NoiseType::StateIndependent> {
  public:
    /**
     * \brief Qubit bit flip noise constructor
     *
     * \param p Noise probability
     */
    explicit QubitBitFlipNoise(double p)
        : NoiseBase({Gates::get_no_thread_local_instance().Id2,
                     Gates::get_no_thread_local_instance().X},
                    {1 - p, p}) {
        // EXCEPTION CHECKS

        if (p < 0 || p > 1)
            throw exception::OutOfRange(
                "qpp::QubitBitFlipNoise::QubitBitFlipNoise()", "p");
        // END EXCEPTION CHECKS
    }
}; /* class QubitBitFlipNoise */

/**
 * \class qpp::QubitBitPhaseFlipNoise
 * \brief Qubit bit-phase flip (dephasing) noise
 */
class QubitBitPhaseFlipNoise : public NoiseBase<NoiseType::StateIndependent> {
  public:
    /**
     * \brief Qubit bit-phase flip noise constructor
     *
     * \param p Noise probability
     */
    explicit QubitBitPhaseFlipNoise(double p)
        : NoiseBase({Gates::get_no_thread_local_instance().Id2,
                     Gates::get_no_thread_local_instance().Y},
                    {1 - p, p}) {
        // EXCEPTION CHECKS

        if (p < 0 || p > 1)
            throw exception::OutOfRange(
                "qpp::QubitBitPhaseFlipNoise::QubitBitPhaseFlipNoise()", "p");
        // END EXCEPTION CHECKS
    }
}; /* class QubitBitPhaseFlipNoise */

/**
 * \class qpp::QubitAmplitudeDampingNoise
 * \brief Qubit amplitude damping noise, as described in Nielsen and Chuang
 */
class QubitAmplitudeDampingNoise : public NoiseBase<NoiseType::StateDependent> {
  public:
    /**
     * \brief Qubit amplitude damping noise constructor
     *
     * \param gamma Amplitude damping coefficient
     */
    explicit QubitAmplitudeDampingNoise(double gamma)
        : NoiseBase(std::vector<cmat>{
              ((cmat(2, 2)) << 1, 0, 0, std::sqrt(gamma)).finished(),
              ((cmat(2, 2)) << 0, std::sqrt(1 - gamma), 0, 0).finished()}) {
        // EXCEPTION CHECKS

        if (gamma < 0 || gamma > 1)
            throw exception::OutOfRange(
                "qpp::QubitAmplitudeDampingNoise::QubitAmplitudeDampingNoise()",
                "gamma");
        // END EXCEPTION CHECKS
    }
}; /* class QubitAmplitudeDampingNoise */

/**
 * \class qpp::QubitPhaseDampingNoise
 * \brief Qubit phase damping noise, as described in Nielsen and Chuang
 */
class QubitPhaseDampingNoise : public NoiseBase<NoiseType::StateDependent> {
  public:
    /**
     * \brief Qubit phase damping noise constructor
     *
     * \param lambda Phase damping coefficient
     */
    explicit QubitPhaseDampingNoise(double lambda)
        : NoiseBase(std::vector<cmat>{
              ((cmat(2, 2)) << 1, 0, 0, std::sqrt(1 - lambda)).finished(),
              ((cmat(2, 2)) << 0, 0, 0, std::sqrt(lambda)).finished()}) {
        // EXCEPTION CHECKS

        if (lambda < 0 || lambda > 1)
            throw exception::OutOfRange(
                "qpp::QubitPhaseDampingNoise::QubitPhaseDampingNoise()",
                "lambda");
        // END EXCEPTION CHECKS
    }
}; /* class QubitPhaseDampingNoise */

// qudit noise models

/**
 * \class qpp::QuditDepolarizingNoise
 * \brief Qudit depolarizing noise
 */
class QuditDepolarizingNoise : public NoiseBase<NoiseType::StateIndependent> {
    /**
     * \brief Constructs the vector of Kraus operators
     *
     * \param D Qudit dimension
     * \return Vector of Kraus operators representing the depolarizing noise
     */
    static std::vector<cmat> fill_Ks_(idx D) {
        std::vector<cmat> Ks(D * D);
        idx tmp = 0;
        for (idx i = 0; i < D; ++i)
            for (idx j = 0; j < D; ++j)
                Ks[tmp++] =
                    powm(Gates::get_no_thread_local_instance().Xd(D), i) *
                    powm(Gates::get_no_thread_local_instance().Zd(D), j);

        return Ks;
    }

    /**
     * \brief Fills the probability vector
     *
     * \param p Probability
     * \param D Qudit dimension
     * \return Probability vector
     */
    static std::vector<double> fill_probs_(double p, idx D) {
        std::vector<double> probs(D * D);
        probs[0] = 1 - p;
        for (idx i = 1; i < D * D; ++i)
            probs[i] =
                p / static_cast<double>(D - 1) * static_cast<double>(D - 1);

        return probs;
    }

  public:
    /**
     * \brief Qudit depolarizing noise constructor
     *
     * \param p Noise probability
     * \param D Qudit dimension
     */
    explicit QuditDepolarizingNoise(double p, idx D)
        : NoiseBase(fill_Ks_(D), fill_probs_(p, D)) {
        // EXCEPTION CHECKS

        if (D < 2)
            throw exception::OutOfRange(
                "qpp::QuditDepolarizingNoise::QuditDepolarizingNoise()", "D");
        if (p < 0 || p > 1)
            throw exception::OutOfRange(
                "qpp::QuditDepolarizingNoise::QuditDepolarizingNoise()", "p");
        // END EXCEPTION CHECKS
    }
}; /* class QuditDepolarizingNoise */

} /* namespace qpp */

#endif /* CLASSES_NOISE_HPP_ */
