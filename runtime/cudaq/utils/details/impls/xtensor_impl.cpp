/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
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

using xtensor_shape_type = std::vector<std::size_t>;

/// @brief An implementation of tensor_impl using xtensor library
template <typename Scalar>
class xtensor : public cudaq::details::tensor_impl<Scalar> {
private:
  std::unique_ptr<Scalar[]> owned_data = {}; ///< Pointer to owned tensor data
  Scalar *borrow_data = nullptr; ///< Pointer to borrowed tensor data
  xtensor_shape_type m_shape;    ///< Shape of the tensor

  /// Enum indicating style of ownership of the raw data
  enum OwnershipMode { Invalid, OwnedByCopy, OwnedByTake, Borrowed };
  OwnershipMode ownership = Invalid;

  /// @brief Check if the given indices are valid for this tensor
  /// @param idxs Vector of indices to check
  /// @return true if indices are valid, false otherwise
  bool validIndices(const xtensor_shape_type &idxs) const {
    if (idxs.size() != m_shape.size())
      return false;
    for (std::size_t dim = 0; auto idx : idxs)
      if (idx < 0 || idx >= m_shape[dim++])
        return false;
    return true;
  }

  Scalar *m_data() const {
    switch (ownership) {
    case Invalid:
      return nullptr;
    case OwnedByCopy:
    case OwnedByTake:
      return owned_data.get();
    case Borrowed:
      return borrow_data;
    }
    return nullptr;
  }

  // Clear the data.
  void deallocate() {
    switch (ownership) {
    case Invalid:
      return;
    case OwnedByCopy:
      // This is the one case that the tensor object has done allocation.
      owned_data.release();
      break;
    case OwnedByTake:
      break;
    case Borrowed:
      borrow_data = nullptr;
      break;
    }
    ownership = Invalid;
  }

  static std::size_t compute_shape_size(const xtensor_shape_type &shape) {
    if (shape.empty())
      return 0;
    return std::accumulate(shape.begin(), shape.end(), 1,
                           std::multiplies<std::size_t>());
  }

public:
  xtensor() = delete;
  xtensor(const Scalar *d, const xtensor_shape_type &shape) { copy(d, shape); }
  xtensor(const xtensor &from) {
    m_shape = from.m_shape;
    ownership = from.ownership;
    switch (ownership) {
    case OwnedByCopy: {
      auto size = compute_shape_size(m_shape);
      owned_data = std::make_unique<Scalar[]>(size);
      auto *d = from.owned_data.get();
      std::copy(d, d + size, owned_data.get());
    } break;
    case OwnedByTake:
      owned_data.swap(from.owned_data);
      break;
    case Borrowed:
      borrow_data = from.borrow_data;
      break;
    }
  }
  xtensor &operator=(const xtensor &from) {
    m_shape = from.m_shape;
    ownership = from.ownership;
    switch (ownership) {
    case OwnedByCopy: {
      auto size = compute_shape_size(m_shape);
      owned_data = std::make_unique<Scalar[]>(size);
      auto *d = from.owned_data.get();
      std::copy(d, d + size, owned_data.get());
    } break;
    case OwnedByTake:
      owned_data.swap(from.owned_data);
      break;
    case Borrowed:
      borrow_data = from.borrow_data;
      break;
    }
    return *this;
  }

  /// @brief Destructor for xtensor
  ~xtensor() { ownership = Invalid; }

  /// @brief Get the rank of the tensor
  /// @return The rank (number of dimensions) of the tensor
  std::size_t rank() const override { return m_shape.size(); }

  /// @brief Get the total size of the tensor
  /// @return The total number of elements in the tensor
  std::size_t size() const override { return compute_shape_size(m_shape); }

  /// @brief Get the shape of the tensor
  /// @return A vector containing the dimensions of the tensor
  xtensor_shape_type shape() const override { return m_shape; }

  /// @brief Access a mutable element of the tensor
  /// @param indices The indices of the element to access
  /// @return A reference to the element at the specified indices
  /// @throws std::runtime_error if indices are invalid
  Scalar &at(const xtensor_shape_type &indices) override {
    if (!validIndices(indices))
      throw std::runtime_error("Invalid tensor indices: " +
                               fmt::format("{}", fmt::join(indices, ", ")));

    return xt::adapt(m_data(), size(), xt::no_ownership(), m_shape)[indices];
  }

  /// @brief Access a const element of the tensor
  /// @param indices The indices of the element to access
  /// @return A const reference to the element at the specified indices
  /// @throws std::runtime_error if indices are invalid
  const Scalar &at(const xtensor_shape_type &indices) const override {
    if (!validIndices(indices))
      throw std::runtime_error("Invalid constant tensor indices: " +
                               fmt::format("{}", fmt::join(indices, ", ")));
    return xt::adapt(m_data(), size(), xt::no_ownership(), m_shape)[indices];
  }

  /// @brief Copy data into the tensor
  /// @param d Pointer to the source data
  /// @param shape The shape of the source data
  void copy(const Scalar *d, const xtensor_shape_type &shape) override {
    deallocate();
    ownership = OwnedByCopy;
    auto size = compute_shape_size(shape);
    auto newData = std::make_unique<Scalar[]>(size);
    owned_data.swap(newData);
    std::copy(d, d + size, owned_data.get());
    m_shape = shape;
  }

  /// @brief Take ownership of the given data
  /// @param d Pointer to the source data
  /// @param shape The shape of the source data
  void take(std::unique_ptr<Scalar[]> &d,
            const xtensor_shape_type &shape) override {
    deallocate();
    ownership = OwnedByTake;
    owned_data.swap(d);
    m_shape = shape;
  }
  void take(std::unique_ptr<Scalar[]> &&d,
            const xtensor_shape_type &shape) override {
    deallocate();
    ownership = OwnedByTake;
    owned_data.swap(d);
    m_shape = shape;
  }

  /// @brief Borrow the given data without taking ownership
  /// @param d Pointer to the source data
  /// @param shape The shape of the source data
  void borrow(const Scalar *d, const xtensor_shape_type &shape) override {
    deallocate();
    ownership = Borrowed;
    borrow_data = const_cast<Scalar *>(d);
    m_shape = shape;
  }

  Scalar *data() override { return m_data(); }
  const Scalar *data() const override { return m_data(); }
  void dump() const override {
    std::cerr << xt::adapt(m_data(), size(), xt::no_ownership(), m_shape)
              << '\n';
  }

  // Double dispatch to make sure both arguments are this derived class.
  details::tensor_impl<Scalar>
  dd_multiply(const details::tensor_impl<Scalar> &left) const override {
    return left.multiply(this);
  }
  details::tensor_impl<Scalar>
  dd_add(const details::tensor_impl<Scalar> &left) const override {
    return left.add(this);
  }

  details::tensor_impl<Scalar>
  multiply(const details::tensor_impl<Scalar> *r) const override {
    auto *right = static_cast<const xtensor<Scalar> *>(r);
    auto *left = this;

    // TODO: call some library here
  }

  details::tensor_impl<Scalar>
  add(const details::tensor_impl<Scalar> *r) const override {
    auto *right = static_cast<const xtensor<Scalar> *>(r);
    auto *left = this;

    // TODO: call some library here
  }

  static constexpr auto ScalarAsString = cudaq::type_to_string<Scalar>();

  /// @brief Custom creator function for xtensor
  /// @param d Pointer to the tensor data
  /// @param s Shape of the tensor
  /// @return A unique pointer to the created xtensor object
  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION_WITH_NAME(
      xtensor<Scalar>, std::string("xtensor") + std::string(ScalarAsString),
      static std::unique_ptr<cudaq::details::tensor_impl<Scalar>> create(
          const Scalar *d, const xtensor_shape_type s) {
        return std::make_unique<xtensor<Scalar>>(d, s);
      })
};

// Register all the xtensor types.

template <>
const bool xtensor<std::complex<double>>::registered_ =
    xtensor<std::complex<double>>::register_type();
template <>
const bool xtensor<std::complex<float>>::registered_ =
    xtensor<std::complex<float>>::register_type();
template <>
const bool xtensor<double>::registered_ = xtensor<double>::register_type();
template <>
const bool xtensor<float>::registered_ = xtensor<float>::register_type();
template <>
const bool xtensor<std::int64_t>::registered_ =
    xtensor<std::int64_t>::register_type();
template <>
const bool xtensor<std::int32_t>::registered_ =
    xtensor<std::int32_t>::register_type();
template <>
const bool xtensor<std::int16_t>::registered_ =
    xtensor<std::int16_t>::register_type();
template <>
const bool xtensor<std::uint8_t>::registered_ =
    xtensor<std::uint8_t>::register_type();

} // namespace cudaq
