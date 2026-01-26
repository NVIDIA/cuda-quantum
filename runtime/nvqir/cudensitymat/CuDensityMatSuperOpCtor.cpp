/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CuDensityMatErrorHandling.h"
#include "CuDensityMatOpConverter.h"
#include "CuDensityMatUtils.h"
#include "common/Logger.h"
#include <iostream>
#include <map>
#include <ranges>

namespace {
cudaq::product_op<cudaq::matrix_handler>
computeDagger(const cudaq::matrix_handler &op) {
  const std::string daggerOpName = op.to_string(false) + "_dagger";
  try {
    auto func = [op](const std::vector<int64_t> &dimensions,
                     const std::unordered_map<std::string, std::complex<double>>
                         &params) {
      cudaq::dimension_map dims;
      if (dimensions.size() != op.degrees().size())
        throw std::runtime_error("Dimension mismatched");

      for (std::size_t i = 0; i < dimensions.size(); ++i) {
        dims[op.degrees()[i]] = dimensions[i];
      }
      auto originalMat = op.to_matrix(dims, params);
      return originalMat.adjoint();
    };

    auto dia_func =
        [op](const std::vector<int64_t> &dimensions,
             const std::unordered_map<std::string, std::complex<double>>
                 &params) {
          cudaq::dimension_map dims;
          if (dimensions.size() != op.degrees().size())
            throw std::runtime_error("Dimension mismatched");

          for (std::size_t i = 0; i < dimensions.size(); ++i) {
            dims[op.degrees()[i]] = dimensions[i];
          }
          auto diaMat = op.to_diagonal_matrix(dims, params);
          for (auto &offset : diaMat.second)
            offset *= -1;
          for (auto &element : diaMat.first)
            element = std::conj(element);
          return diaMat;
        };
    cudaq::matrix_handler::define(daggerOpName, {-1}, std::move(func),
                                  std::move(dia_func));
  } catch (...) {
    // Nothing, this has been define
  }
  return cudaq::matrix_handler::instantiate(daggerOpName, op.degrees());
}

cudaq::scalar_operator computeDagger(const cudaq::scalar_operator &scalar) {
  if (scalar.is_constant()) {
    return cudaq::scalar_operator(std::conj(scalar.evaluate()));
  } else {
    return cudaq::scalar_operator(
        [scalar](
            const std::unordered_map<std::string, std::complex<double>> &params)
            -> std::complex<double> {
          return std::conj(scalar.evaluate(params));
        });
  }
}

cudaq::product_op<cudaq::matrix_handler>
computeDagger(const cudaq::product_op<cudaq::matrix_handler> &productOp) {
  std::vector<cudaq::product_op<cudaq::matrix_handler>> daggerOps;
  for (const auto &component : productOp) {
    if (const auto *elemOp =
            dynamic_cast<const cudaq::matrix_handler *>(&component)) {
      daggerOps.emplace_back(computeDagger(*elemOp));
    } else {
      throw std::runtime_error("Unhandled type!");
    }
  }
  std::reverse(daggerOps.begin(), daggerOps.end());

  if (daggerOps.empty())
    throw std::runtime_error("Empty product operator");
  cudaq::product_op<cudaq::matrix_handler> daggerProduct = daggerOps[0];
  for (std::size_t i = 1; i < daggerOps.size(); ++i) {
    daggerProduct *= daggerOps[i];
  }
  daggerProduct *= computeDagger(productOp.get_coefficient());
  return daggerProduct;
}

cudaq::sum_op<cudaq::matrix_handler>
computeDagger(const cudaq::sum_op<cudaq::matrix_handler> &sumOp) {
  cudaq::sum_op<cudaq::matrix_handler> daggerSum =
      cudaq::sum_op<cudaq::matrix_handler>::empty();
  for (const cudaq::product_op<cudaq::matrix_handler> &prodOp : sumOp)
    daggerSum += computeDagger(prodOp);

  return daggerSum;
}

} // namespace

std::vector<
    std::pair<std::vector<cudaq::scalar_operator>, cudensitymatOperatorTerm_t>>
cudaq::dynamics::CuDensityMatOpConverter::computeLindbladTerms(
    const std::vector<sum_op<cudaq::matrix_handler>> &batchedCollapseOps,
    const std::vector<int64_t> &modeExtents,
    const std::unordered_map<std::string, std::complex<double>> &parameters) {
  if (batchedCollapseOps.empty())
    return {};
  // Split the collapse operators into batched product terms.
  auto batchedCollapsedProdTerms = splitToBatch(batchedCollapseOps);
  std::vector<std::pair<std::vector<cudaq::scalar_operator>,
                        cudensitymatOperatorTerm_t>>
      lindbladTerms;

  for (const auto &collapseOp : batchedCollapsedProdTerms) {
    const auto allSameDegrees =
        std::all_of(collapseOp.begin(), collapseOp.end(),
                    [&](const product_op<matrix_handler> &prodOp) {
                      return prodOp.degrees() == collapseOp[0].degrees();
                    });
    if (!allSameDegrees) {
      throw std::invalid_argument("All product terms in a collapse operator "
                                  "must have the same degrees.");
    }
  }

  const auto batchedSize = batchedCollapsedProdTerms.size();
  const auto numberProductTerms = batchedCollapsedProdTerms[0].size();
  for (std::size_t leftProdTermIdx = 0; leftProdTermIdx < numberProductTerms;
       ++leftProdTermIdx) {
    for (std::size_t rightProdTermIdx = 0;
         rightProdTermIdx < numberProductTerms; ++rightProdTermIdx) {
      std::vector<product_op<matrix_handler>> l_ops;
      std::vector<product_op<matrix_handler>> r_ops;
      l_ops.reserve(batchedSize);
      r_ops.reserve(batchedSize);
      for (std::size_t i = 0; i < batchedSize; ++i) {
        l_ops.push_back(batchedCollapsedProdTerms[i][leftProdTermIdx]);
        r_ops.push_back(batchedCollapsedProdTerms[i][rightProdTermIdx]);
      }
      // L * rho * L_dagger
      {
        std::vector<scalar_operator> coeffs;
        coeffs.reserve(batchedSize);
        std::vector<cudensitymatElementaryOperator_t> elemOps;
        std::vector<std::vector<std::size_t>> allDegrees;
        std::vector<std::vector<int>> all_action_dual_modalities;
        for (std::size_t i = 0; i < batchedSize; ++i) {
          coeffs.push_back(l_ops[i].get_coefficient() *
                           computeDagger(r_ops[i].get_coefficient()));
        }
        const auto leftNumOps = l_ops[0].num_ops();
        for (std::size_t i = 0; i < leftNumOps; ++i) {
          std::vector<cudaq::matrix_handler> leftOpComponents;
          for (const auto &leftOp : l_ops) {
            const auto &component = leftOp[i];
            if (const auto *elemOp =
                    dynamic_cast<const cudaq::matrix_handler *>(&component)) {
              leftOpComponents.emplace_back(*elemOp);
            } else {
              // Catch anything that we don't know
              throw std::runtime_error("Unhandled type!");
            }
          }

          auto cudmElemOp = createElementaryOperator(leftOpComponents,
                                                     parameters, modeExtents);
          elemOps.emplace_back(cudmElemOp);
          allDegrees.emplace_back(l_ops[0][i].degrees());
          all_action_dual_modalities.emplace_back(
              std::vector<int>(l_ops[0][i].degrees().size(), 0));
        }
        auto ldags = r_ops;
        for (auto &ldag : ldags) {
          ldag = computeDagger(ldag);
        }
        const auto rightNumOps = ldags[0].num_ops();
        for (std::size_t i = 0; i < rightNumOps; ++i) {
          std::vector<cudaq::matrix_handler> rightOpComponents;
          for (const auto &rightOp : ldags) {
            const auto &component = rightOp[i];
            if (const auto *elemOp =
                    dynamic_cast<const cudaq::matrix_handler *>(&component)) {
              rightOpComponents.emplace_back(*elemOp);
            } else {
              // Catch anything that we don't know
              throw std::runtime_error("Unhandled type!");
            }
          }

          auto cudmElemOp = createElementaryOperator(rightOpComponents,
                                                     parameters, modeExtents);
          elemOps.emplace_back(cudmElemOp);
          allDegrees.emplace_back(ldags[0][i].degrees());
          all_action_dual_modalities.emplace_back(
              std::vector<int>(ldags[0][i].degrees().size(), 1));
        }
        cudensitymatOperatorTerm_t D1_term = createProductOperatorTerm(
            elemOps, modeExtents, allDegrees, all_action_dual_modalities);
        lindbladTerms.emplace_back(std::make_pair(coeffs, D1_term));
      }

      std::vector<product_op<matrix_handler>> L_daggerTimesL;
      std::vector<scalar_operator> L_daggerTimesL_coeffs;
      L_daggerTimesL.reserve(batchedSize);
      L_daggerTimesL_coeffs.reserve(batchedSize);
      for (std::size_t i = 0; i < batchedSize; ++i) {
        // -0.5 * L_dagger * L
        auto ldag = computeDagger(r_ops[i]);
        auto l_op = l_ops[i];
        L_daggerTimesL.emplace_back(-0.5 * ldag * l_op);
        L_daggerTimesL_coeffs.push_back(-0.5 * l_ops[i].get_coefficient() *
                                        ldag.get_coefficient());
      }

      {
        std::vector<cudensitymatElementaryOperator_t> elemOps;
        std::vector<std::vector<std::size_t>> allDegrees;
        std::vector<std::vector<int>> all_action_dual_modalities_left;
        std::vector<std::vector<int>> all_action_dual_modalities_right;

        const auto numOps = L_daggerTimesL[0].num_ops();
        for (std::size_t i = 0; i < numOps; ++i) {
          std::vector<cudaq::matrix_handler> components;
          for (const auto &prodOp : L_daggerTimesL) {
            const auto &component = prodOp[i];
            if (const auto *elemOp =
                    dynamic_cast<const cudaq::matrix_handler *>(&component)) {
              components.emplace_back(*elemOp);
            } else {
              // Catch anything that we don't know
              throw std::runtime_error("Unhandled type!");
            }
          }

          auto cudmElemOp =
              createElementaryOperator(components, parameters, modeExtents);
          elemOps.emplace_back(cudmElemOp);
          allDegrees.emplace_back(L_daggerTimesL[0][i].degrees());
          all_action_dual_modalities_left.emplace_back(
              std::vector<int>(L_daggerTimesL[0][i].degrees().size(), 0));
          all_action_dual_modalities_right.emplace_back(
              std::vector<int>(L_daggerTimesL[0][i].degrees().size(), 1));
        }

        {
          // For left side, we need to reverse the order
          std::vector<cudensitymatElementaryOperator_t> d2Ops(elemOps);
          std::reverse(d2Ops.begin(), d2Ops.end());
          std::vector<std::vector<std::size_t>> d2Degrees(allDegrees);
          std::reverse(d2Degrees.begin(), d2Degrees.end());
          cudensitymatOperatorTerm_t D2_term = createProductOperatorTerm(
              d2Ops, modeExtents, d2Degrees, all_action_dual_modalities_left);
          lindbladTerms.emplace_back(
              std::make_pair(L_daggerTimesL_coeffs, D2_term));
        }
        {
          cudensitymatOperatorTerm_t D3_term =
              createProductOperatorTerm(elemOps, modeExtents, allDegrees,
                                        all_action_dual_modalities_right);
          lindbladTerms.emplace_back(
              std::make_pair(L_daggerTimesL_coeffs, D3_term));
        }
      }
    }
  }
  return lindbladTerms;
}

cudensitymatOperator_t
cudaq::dynamics::CuDensityMatOpConverter::constructLiouvillian(
    const std::vector<sum_op<cudaq::matrix_handler>> &hamOperators,
    const std::vector<std::vector<sum_op<cudaq::matrix_handler>>>
        &collapseOperators,
    const std::vector<int64_t> &modeExtents,
    const std::unordered_map<std::string, std::complex<double>> &parameters,
    bool isMasterEquation) {
  LOG_API_TIME();
  if (hamOperators.empty()) {
    throw std::invalid_argument(
        "Cannot construct Liouvillian operator from an empty list of "
        "Hamiltonians.");
  }
  const auto batchSize = hamOperators.size();
  const auto numberProductTerms = hamOperators[0].num_terms();
  // Check if all Hamiltonians have the same number of product terms
  for (const auto &hamiltonian : hamOperators) {
    if (hamiltonian.num_terms() != numberProductTerms) {
      throw std::invalid_argument(
          "All Hamiltonians must have the same number of product terms.");
    }
  }

  const bool noCollapseOperators =
      collapseOperators.empty() ||
      std::all_of(collapseOperators.begin(), collapseOperators.end(),
                  [](const std::vector<sum_op<cudaq::matrix_handler>> &ops) {
                    return ops.empty();
                  });

  if (!isMasterEquation && noCollapseOperators) {
    CUDAQ_INFO("Construct state vector Liouvillian");
    std::vector<sum_op<cudaq::matrix_handler>> liouvillians;
    liouvillians.reserve(batchSize);
    for (const auto &ham : hamOperators) {
      liouvillians.emplace_back(ham * std::complex<double>(0.0, -1.0));
    }
    return convertToCudensitymatOperator(parameters, liouvillians, modeExtents);
  } else {
    CUDAQ_INFO("Construct density matrix Liouvillian");
    cudensitymatOperator_t liouvillian;
    HANDLE_CUDM_ERROR(cudensitymatCreateOperator(
        m_handle, static_cast<int32_t>(modeExtents.size()), modeExtents.data(),
        &liouvillian));
    // Append an operator term to the operator (super-operator)
    // Handle the Hamiltonian
    const std::map<std::string, std::complex<double>> sortedParameters(
        parameters.begin(), parameters.end());
    auto ks = std::views::keys(sortedParameters);
    const std::vector<std::string> keys{ks.begin(), ks.end()};
    std::vector<sum_op<cudaq::matrix_handler>> leftHam;
    std::vector<sum_op<cudaq::matrix_handler>> rightHam;
    leftHam.reserve(batchSize);
    rightHam.reserve(batchSize);
    for (const auto &ham : hamOperators) {
      leftHam.emplace_back(ham * std::complex<double>(0.0, -1.0));
      rightHam.emplace_back(computeDagger(ham) *
                            std::complex<double>(0.0, 1.0));
    }
    // -i constant (left multiplication)
    appendToCudensitymatOperator(liouvillian, parameters, leftHam, modeExtents,
                                 /*duality=*/0);
    // +i constant (right multiplication, i.e., dual)
    appendToCudensitymatOperator(liouvillian, parameters, rightHam, modeExtents,
                                 /*duality=*/1);

    // Check that all collapsed operator vectors have the same size
    if (!collapseOperators.empty()) {
      const auto collapseSize = collapseOperators[0].size();
      for (const auto &collapseOperator : collapseOperators) {
        if (collapseOperator.size() != collapseSize) {
          throw std::invalid_argument(
              "All collapse operator vectors must have the same size.");
        }
      }
      // Handle collapsed operators
      for (std::size_t i = 0; i < collapseSize; ++i) {
        std::vector<sum_op<cudaq::matrix_handler>> batchedCollapseTerms;
        for (const auto &collapseOperator : collapseOperators) {
          batchedCollapseTerms.push_back(collapseOperator[i]);
        }
        for (auto &[coeffs, term] : computeLindbladTerms(
                 batchedCollapseTerms, modeExtents, parameters)) {
          assert(coeffs.size() == batchSize);
          appendBatchedTermToOperator(liouvillian, term, coeffs, keys);
        }
      }
    }

    return liouvillian;
  }
}

cudensitymatOperator_t
cudaq::dynamics::CuDensityMatOpConverter::constructLiouvillian(
    const std::vector<super_op> &superOps,
    const std::vector<int64_t> &modeExtents,
    const std::unordered_map<std::string, std::complex<double>> &parameters) {
  LOG_API_TIME();
  if (superOps.empty())
    throw std::invalid_argument(
        "Super-operator cannot be empty. At least one super-operator is "
        "required.");

  cudensitymatOperator_t liouvillian;
  HANDLE_CUDM_ERROR(cudensitymatCreateOperator(
      m_handle, static_cast<int32_t>(modeExtents.size()), modeExtents.data(),
      &liouvillian));
  // Append an operator term to the operator (super-operator)
  // Handle the Hamiltonian
  const std::map<std::string, std::complex<double>> sortedParameters(
      parameters.begin(), parameters.end());
  auto ks = std::views::keys(sortedParameters);
  const std::vector<std::string> keys{ks.begin(), ks.end()};

  const auto batchSize = superOps.size();
  const auto hasLeftMultiplication = [](const super_op::term &term) {
    return term.first.has_value();
  };
  const auto hasRightMultiplication = [](const super_op::term &term) {
    return term.second.has_value();
  };
  for (std::size_t i = 1; i < batchSize; ++i) {
    if (superOps[i].num_terms() != superOps[0].num_terms()) {
      throw std::invalid_argument(
          "All super-operators in the batch must have the same number of "
          "terms.");
    }

    for (std::size_t j = 0; j < superOps[i].num_terms(); ++j) {
      if (hasLeftMultiplication(superOps[i][j]) !=
              hasLeftMultiplication(superOps[0][j]) ||
          hasRightMultiplication(superOps[i][j]) !=
              hasRightMultiplication(superOps[0][j])) {
        throw std::invalid_argument(
            "All super-operators in the batch must have the same structure");
      }
    }
  }

  for (std::size_t termId = 0; termId < superOps[0].num_terms(); ++termId) {
    std::vector<cudaq::product_op<cudaq::matrix_handler>> leftOps;
    std::vector<cudaq::product_op<cudaq::matrix_handler>> rightOps;
    leftOps.reserve(batchSize);
    rightOps.reserve(batchSize);
    for (std::size_t i = 0; i < batchSize; ++i) {
      if (superOps[i][termId].first.has_value())
        leftOps.push_back(superOps[i][termId].first.value());
      if (superOps[i][termId].second.has_value())
        rightOps.push_back(superOps[i][termId].second.value());
    }

    const auto allSameDegrees =
        [](const std::vector<cudaq::product_op<cudaq::matrix_handler>> &ops) {
          return std::all_of(
              ops.begin(), ops.end(),
              [&](const cudaq::product_op<cudaq::matrix_handler> &op) {
                return op.degrees() == ops[0].degrees();
              });
        };

    if (!leftOps.empty()) {
      if (!rightOps.empty()) {
        if (leftOps.size() != rightOps.size()) {
          throw std::invalid_argument(
              "Left and right product terms in a super-operator must have the "
              "same number of terms.");
        }
        if (!allSameDegrees(leftOps) || !allSameDegrees(rightOps)) {
          throw std::invalid_argument(
              "All product terms in a super-operator must have the same "
              "degrees.");
        }

        std::vector<cudaq::scalar_operator> coeffs;
        coeffs.reserve(batchSize);
        // L * rho * R
        std::vector<cudensitymatElementaryOperator_t> elemOps;
        std::vector<std::vector<std::size_t>> allDegrees;
        std::vector<std::vector<int>> all_action_dual_modalities;

        const auto leftNumOps = leftOps[0].num_ops();
        for (std::size_t i = 0; i < leftNumOps; ++i) {
          std::vector<cudaq::matrix_handler> leftOpComponents;
          for (const auto &leftOp : leftOps) {
            const auto &component = leftOp[i];
            if (const auto *elemOp =
                    dynamic_cast<const cudaq::matrix_handler *>(&component)) {
              leftOpComponents.emplace_back(*elemOp);
            } else {
              // Catch anything that we don't know
              throw std::runtime_error("Unhandled type!");
            }
          }

          auto cudmElemOp = createElementaryOperator(leftOpComponents,
                                                     parameters, modeExtents);
          elemOps.emplace_back(cudmElemOp);
          allDegrees.emplace_back(leftOps[0][i].degrees());
          all_action_dual_modalities.emplace_back(
              std::vector<int>(leftOps[0][i].degrees().size(), 0));
        }

        const auto rightNumOps = rightOps[0].num_ops();
        for (std::size_t i = 0; i < rightNumOps; ++i) {
          std::vector<cudaq::matrix_handler> rightOpComponents;
          for (const auto &rightOp : rightOps) {
            const auto &component = rightOp[i];
            if (const auto *elemOp =
                    dynamic_cast<const cudaq::matrix_handler *>(&component)) {
              rightOpComponents.emplace_back(*elemOp);
            } else {
              // Catch anything that we don't know
              throw std::runtime_error("Unhandled type!");
            }
          }

          auto cudmElemOp = createElementaryOperator(rightOpComponents,
                                                     parameters, modeExtents);
          elemOps.emplace_back(cudmElemOp);
          allDegrees.emplace_back(rightOps[0][i].degrees());
          all_action_dual_modalities.emplace_back(
              std::vector<int>(rightOps[0][i].degrees().size(), 1));
        }

        for (std::size_t i = 0; i < batchSize; ++i) {
          coeffs.push_back(leftOps[i].get_coefficient() *
                           rightOps[i].get_coefficient());
        }

        cudensitymatOperatorTerm_t term = createProductOperatorTerm(
            elemOps, modeExtents, allDegrees, all_action_dual_modalities);
        appendBatchedTermToOperator(liouvillian, term, coeffs, keys);
      } else {
        std::vector<sum_op<cudaq::matrix_handler>> ops;
        ops.reserve(batchSize);
        for (const auto &leftOp : leftOps) {
          ops.emplace_back(sum_op<cudaq::matrix_handler>(leftOp));
        }
        const auto duality = 0; // No duality for left multiplication
        appendToCudensitymatOperator(liouvillian, parameters, ops, modeExtents,
                                     duality);
      }
    } else {
      if (!rightOps.empty()) {
        std::vector<sum_op<cudaq::matrix_handler>> ops;
        ops.reserve(batchSize);
        for (const auto &rightOp : rightOps) {
          ops.emplace_back(sum_op<cudaq::matrix_handler>(rightOp));
        }
        const auto duality = 1; // Duality for right multiplication
        appendToCudensitymatOperator(liouvillian, parameters, ops, modeExtents,
                                     duality);
      } else {
        throw std::runtime_error("Invalid super-operator term encountered: no "
                                 "operation action is specified.");
      }
    }
  }

  return liouvillian;
}
