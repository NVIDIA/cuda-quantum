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
 * \file experimental/experimental.hpp
 * \brief Experimental/test functions/classes
 */

#ifndef EXPERIMENTAL_EXPERIMENTAL_HPP_
#define EXPERIMENTAL_EXPERIMENTAL_HPP_

/**
 * \namespace qpp::experimental
 * \brief Experimental/test functions/classes, do not use or modify
 */
namespace qpp::experimental {
/**
 * \brief Sequentially measures the part \a target of the multi-partite state
 * vector or density matrix \a A in the computational basis
 * \see qpp::measure()
 *
 * \note If \a destructive is set to true (by default), the measurement is
 * destructive, i.e., the measured subsystems are traced away.
 *
 * \param A Eigen expression
 * \param target Subsystem indexes that are measured
 * \param dims Dimensions of the multi-partite system
 * \param destructive Destructive measurement, true by default
 * \return Tuple of: 1. Vector of outcome results of the measurement (ordered in
 * increasing order with respect to \a target, i.e. first measurement result
 * corresponds to the subsystem with the smallest index), 2. Outcome
 * probability, and 3. Post-measurement normalized state
 */
template <typename Derived>
[[qpp::critical]] std::tuple<std::vector<idx>, double, expr_t<Derived>>
measure_seq(const Eigen::MatrixBase<Derived>& A, std::vector<idx> target,
            const std::vector<idx>& dims, bool destructive = true) {
    // typename Eigen::MatrixBase<Derived>::EvalReturnType cA = A.derived();
    expr_t<Derived> cA = A.derived();

    // EXCEPTION CHECKS

    // check zero-size
    if (!internal::check_nonzero_size(cA))
        throw exception::ZeroSize("qpp::measure_seq()", "A");

    // check that dimension is valid
    if (!internal::check_dims(dims))
        throw exception::DimsInvalid("qpp::measure_seq()", "dims");

    // check valid state and matching dimensions
    if (internal::check_cvector(cA)) {
        if (!internal::check_dims_match_cvect(dims, cA))
            throw exception::DimsMismatchCvector("qpp::measure_seq()",
                                                 "A/dims");
    } else if (internal::check_square_mat(cA)) {
        if (!internal::check_dims_match_mat(dims, cA))
            throw exception::DimsMismatchMatrix("qpp::measure_seq()", "A/dims");
    } else
        throw exception::MatrixNotSquareNorCvector("qpp::measure_seq()", "A");

    // check that target is valid w.r.t. dims
    if (!internal::check_subsys_match_dims(target, dims))
        throw exception::SubsysMismatchDims("qpp::measure_seq()",
                                            "dims/target");
    // END EXCEPTION CHECKS

    idx n = dims.size();                     // number of subsystems
    idx D = prod(dims);                      // total dimension
    std::sort(target.begin(), target.end()); // sort target in increasing order
    std::vector<idx> subsys = complement(target, n); // complement of target

    std::vector<idx> dims_target(target.size());
    for (idx i = 0; i < target.size(); ++i) {
        dims_target[i] = dims[target[i]];
    }

    std::vector<idx> dims_subsys(subsys.size());
    for (idx i = 0; i < subsys.size(); ++i) {
        dims_subsys[i] = dims[subsys[i]];
    }
    idx Dsubsys = prod(dims_subsys);

    bool is_ket = internal::check_cvector(cA);
    std::vector<double> pbs;
    if (is_ket) {
        pbs = qpp::abssq(cA);
    } else {
        pbs.resize(D);
        for (idx i = 0; i < D; ++i)
            pbs[i] = std::real(cA(i, i));
    }

    // sample
    std::discrete_distribution dd(pbs.begin(), pbs.end());
    auto& gen = RandomDevices::get_instance().get_prng();
    idx sample_dec = dd(gen);
    auto sample_midx = n2multiidx(sample_dec, dims);

    // measurement result as a vector of dits
    std::vector<idx> measurement_midx(target.size(), 0);
    std::transform(target.begin(), target.end(), measurement_midx.begin(),
                   [&sample_midx = std::as_const(sample_midx)](idx pos) {
                       return sample_midx[pos];
                   });

    // checks whether a basis state fully overlap with the measurement result
    // given a target
    auto overlap = [](const std::vector<idx>& basis_midx,
                      const std::vector<idx>& measurement_midx,
                      const std::vector<idx>& target) {
        idx target_size = target.size();
        for (idx j = 0; j < target_size; ++j) {
            if (basis_midx[target[j]] != measurement_midx[j]) {
                return false;
            }
        }
        return true;
    };

    expr_t<Derived> out_state{};
    double prob = 0;
    //************ ket ************//
    if (is_ket) {
        // compute the probability of the outcome and the output state
        out_state = destructive ? ket::Zero(Dsubsys) : ket::Zero(D);
#ifdef HAS_OPENMP
// NOLINTNEXTLINE
#pragma omp parallel for
#endif // HAS_OPENMP
        for (idx i = 0; i < D; ++i) {
            if (pbs[i] == 0)
                continue;
            std::vector<idx> ket_midx = n2multiidx(i, dims);
            if (overlap(ket_midx, measurement_midx, target)) {
                ket current_ket;
                if (destructive) {
                    std::vector<idx> subsys_midx(subsys.size());
                    for (idx j = 0; j < subsys.size(); ++j) {
                        subsys_midx[j] = ket_midx[subsys[j]];
                    }
                    current_ket = mket(subsys_midx, dims_subsys) * cA(i);
                } else {
                    current_ket = mket(ket_midx, dims) * cA(i);
                }
#ifdef HAS_OPENMP
#pragma omp critical
#endif // HAS_OPENMP
                { out_state += current_ket; }
            } // end if(overlap_ket)
        }     // end for
        double norm_out_state = norm(out_state);
        prob = norm_out_state * norm_out_state;
        out_state = out_state / norm_out_state;
    } // end if(ket)
    //************ density matrix ************//
    else {
        // compute the probability of the outcome and the output state
        out_state =
            destructive ? cmat::Zero(Dsubsys, Dsubsys) : cmat::Zero(D, D);
#ifdef HAS_OPENMP
// NOLINTNEXTLINE
#pragma omp parallel for
#endif // HAS_OPENMP
        for (idx i = 0; i < D; ++i) {
            if (pbs[i] == 0)
                continue;
            std::vector<idx> ket_midx = n2multiidx(i, dims);
            if (overlap(ket_midx, measurement_midx, target)) {
                ket current_ket;
                if (destructive) {
                    std::vector<idx> subsys_midx(subsys.size());
                    for (idx k = 0; k < subsys.size(); ++k) {
                        subsys_midx[k] = ket_midx[subsys[k]];
                    }
                    current_ket = mket(subsys_midx, dims_subsys);
                } else {
                    current_ket = mket(ket_midx, dims);
                }
                // now run over all possible bras
                for (idx j = 0; j < D; ++j) {
                    bra current_bra = bra::Zero(Dsubsys);
                    if (cA(i, j) == cmat::Scalar{})
                        continue;
                    std::vector<idx> bra_midx = n2multiidx(j, dims);
                    if (overlap(bra_midx, measurement_midx, target)) {
                        if (destructive) {
                            std::vector<idx> subsys_midx(subsys.size());
                            for (idx k = 0; k < subsys.size(); ++k) {
                                subsys_midx[k] = bra_midx[subsys[k]];
                            }
                            current_bra =
                                adjoint(mket(subsys_midx, dims_subsys));
                        } else {
                            current_bra = adjoint(mket(bra_midx, dims));
                        }
#ifdef HAS_OPENMP
#pragma omp critical
#endif // HAS_OPENMP
                        { out_state += cA(i, j) * current_ket * current_bra; }
                    } // end if(overlap_bra)
                }     // end for(all bras)
            }         // end if(overlap_ket)
        }             // end for(all kets)
        prob = trace(out_state).real();
        out_state = out_state / prob;
    } // end else(density matrix)

    return std::make_tuple(measurement_midx, prob, cA);
}

/**
 * \brief Sequentially measures the part \a target of the multi-partite state
 * vector or density matrix \a A in the computational basis
 * \see qpp::measure()
 *
 * \note If \a destructive is set to true (by default), the measurement is
 * destructive, i.e., the measured subsystems are traced away.
 *
 * \param A Eigen expression
 * \param target Subsystem indexes that are measured
 * \param d Subsystem dimensions
 * \param destructive Destructive measurement, true by default
 * \return Tuple of: 1. Vector of outcome results of the measurement (ordered in
 * increasing order with respect to \a target, i.e. first measurement result
 * corresponds to the subsystem with the smallest index), 2. Outcome
 * probability, and 3. Post-measurement normalized state
 */
template <typename Derived>
std::tuple<std::vector<idx>, double, expr_t<Derived>>
measure_seq(const Eigen::MatrixBase<Derived>& A, const std::vector<idx>& target,
            idx d = 2, bool destructive = true) {
    const typename Eigen::MatrixBase<Derived>::EvalReturnType& rA = A.derived();

    // EXCEPTION CHECKS

    // check zero size
    if (!internal::check_nonzero_size(rA))
        throw exception::ZeroSize("qpp::measure_seq()", "A");

    // check valid dims
    if (d < 2)
        throw exception::DimsInvalid("qpp::measure_seq()", "d");
    // END EXCEPTION CHECKS

    idx n = internal::get_num_subsys(static_cast<idx>(rA.rows()), d);
    std::vector<idx> dims(n, d); // local dimensions vector

    return measure_seq(rA, target, dims, destructive);
}
} /* namespace qpp::experimental */

#endif /* EXPERIMENTAL_EXPERIMENTAL_HPP_ */
