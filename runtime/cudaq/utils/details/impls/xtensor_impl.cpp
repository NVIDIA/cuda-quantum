/*******************************************************************************
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/utils/details/tensor_impl.h"
#include "cudaq/utils/type_traits.h"
#include <fmt/ranges.h>
#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>

namespace cudaq {

/// @brief An implementation of tensor_impl using xtensor library
template <typename Scalar>
class xtensor : public cudaq::details::tensor_impl<Scalar> {
private:
  Scalar *m_data = nullptr;         ///< Pointer to the tensor data
  std::vector<std::size_t> m_shape; ///< Shape of the tensor
  bool ownsData = true; ///< Flag indicating if this object owns the data

  /// @brief Check if the given indices are valid for this tensor
  /// @param idxs Vector of indices to check
  /// @return true if indices are valid, false otherwise
  bool validIndices(const std::vector<std::size_t> &idxs) const {
    if (idxs.size() != m_shape.size())
      return false;
    for (std::size_t dim = 0; auto idx : idxs)
      if (idx < 0 || idx >= m_shape[dim++])
        return false;
    return true;
  }

public:
  /// @brief Constructor for xtensor
  /// @param d Pointer to the tensor data
  /// @param s Shape of the tensor
  xtensor(const Scalar *d, const std::vector<std::size_t> &s)
      : m_data(const_cast<Scalar *>(d)), m_shape(s) {}

  /// @brief Get the rank of the tensor
  /// @return The rank (number of dimensions) of the tensor
  std::size_t rank() const override { return m_shape.size(); }

  /// @brief Get the total size of the tensor
  /// @return The total number of elements in the tensor
  std::size_t size() const override {
    return std::accumulate(m_shape.begin(), m_shape.end(), 1,
                           std::multiplies<size_t>());
  }

  /// @brief Get the shape of the tensor
  /// @return A vector containing the dimensions of the tensor
  std::vector<std::size_t> shape() const override { return m_shape; }

  /// @brief Access a mutable element of the tensor
  /// @param indices The indices of the element to access
  /// @return A reference to the element at the specified indices
  /// @throws std::runtime_error if indices are invalid
  Scalar &at(const std::vector<size_t> &indices) override {
    if (!validIndices(indices))
      throw std::runtime_error("Invalid tensor indices: " +
                               fmt::format("{}", fmt::join(indices, ", ")));

    return xt::adapt(m_data, size(), xt::no_ownership(), m_shape)[indices];
  }

  /// @brief Access a const element of the tensor
  /// @param indices The indices of the element to access
  /// @return A const reference to the element at the specified indices
  /// @throws std::runtime_error if indices are invalid
  const Scalar &at(const std::vector<size_t> &indices) const override {
    if (!validIndices(indices))
      throw std::runtime_error("Invalid constant tensor indices: " +
                               fmt::format("{}", fmt::join(indices, ", ")));
    return xt::adapt(m_data, size(), xt::no_ownership(), m_shape)[indices];
  }

  /// @brief Copy data into the tensor
  /// @param d Pointer to the source data
  /// @param shape The shape of the source data
  void copy(const Scalar *d, const std::vector<std::size_t> &shape) override {
    auto size = std::accumulate(shape.begin(), shape.end(), 1,
                                std::multiplies<size_t>());
    if (m_data)
      delete m_data;

    m_data = m_data = new Scalar[size];
    std::copy(d, d + size, m_data);
    m_shape = shape;
    ownsData = true;
  }

  /// @brief Take ownership of the given data
  /// @param d Pointer to the source data
  /// @param shape The shape of the source data
  void take(const Scalar *d, const std::vector<std::size_t> &shape) override {
    m_data = const_cast<Scalar *>(d);
    m_shape = shape;
    ownsData = true;
  }

  /// @brief Borrow the given data without taking ownership
  /// @param d Pointer to the source data
  /// @param shape The shape of the source data
  void borrow(const Scalar *d, const std::vector<std::size_t> &shape) override {
    m_data = const_cast<Scalar *>(d);
    m_shape = shape;
    ownsData = false;
  }

  Scalar *data() override { return m_data; }
  const Scalar *data() const override { return m_data; }
  void dump() const override {
    std::cerr << xt::adapt(m_data, size(), xt::no_ownership(), m_shape) << '\n';
  }

  static constexpr auto ScalarAsString = cudaq::type_to_string<Scalar>();

  /// @brief Custom creator function for xtensor
  /// @param d Pointer to the tensor data
  /// @param s Shape of the tensor
  /// @return A unique pointer to the created xtensor object
  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION_WITH_NAME(
      xtensor<Scalar>, std::string("xtensor") + std::string(ScalarAsString),
      static std::unique_ptr<cudaq::details::tensor_impl<Scalar>> create(
          const Scalar *d, const std::vector<std::size_t> s) {
        return std::make_unique<xtensor<Scalar>>(d, s);
      })

  /// @brief Destructor for xtensor
  ~xtensor() {
    if (ownsData)
      delete m_data;
  }
};

/// @brief Register the xtensor types

template <>
const bool xtensor<std::complex<double>>::registered_ =
    xtensor<std::complex<double>>::register_type();
template <>
const bool xtensor<std::complex<float>>::registered_ =
    xtensor<std::complex<float>>::register_type();
template <>
const bool xtensor<int>::registered_ = xtensor<int>::register_type();
template <>
const bool xtensor<uint8_t>::registered_ = xtensor<uint8_t>::register_type();
template <>
const bool xtensor<double>::registered_ = xtensor<double>::register_type();
template <>
const bool xtensor<float>::registered_ = xtensor<float>::register_type();
template <>
const bool xtensor<std::size_t>::registered_ =
    xtensor<std::size_t>::register_type();

} // namespace cudaq