/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "details/tensor_impl.h"
#include "extension_point.h"
#include "type_traits.h"

namespace cudaq {

/// @brief A tensor class implementing the PIMPL idiom. The flattened data is
/// stored where the strides grow from right to left (similar to a
/// multi-dimensional C array).
template <typename Scalar = std::complex<double>>
class tensor {
private:
  std::shared_ptr<details::tensor_impl<Scalar>> pimpl;

public:
  /// @brief Type alias for the scalar type used in the tensor
  using scalar_type = typename details::tensor_impl<Scalar>::scalar_type;
  static constexpr auto ScalarAsString = type_to_string<Scalar>();

  /// @brief Construct an empty tensor
  tensor()
      : pimpl(std::shared_ptr<details::tensor_impl<Scalar>>(
            details::tensor_impl<Scalar>::get(
                std::string("xtensor") + std::string(ScalarAsString), {})
                .release())) {}

  /// @brief Construct a tensor with the given shape
  /// @param shape The shape of the tensor
  tensor(const std::vector<std::size_t> &shape)
      : pimpl(std::shared_ptr<details::tensor_impl<Scalar>>(
            details::tensor_impl<Scalar>::get(
                std::string("xtensor") + std::string(ScalarAsString), shape)
                .release())) {}

  /// @brief Construct a tensor with the given data and shape
  /// @param data Pointer to the tensor data
  /// @param shape The shape of the tensor
  tensor(const scalar_type *data, const std::vector<std::size_t> &shape)
      : pimpl(std::shared_ptr<details::tensor_impl<Scalar>>(
            details::tensor_impl<Scalar>::get(std::string("xtensor") +
                                                  std::string(ScalarAsString),
                                              data, shape)
                .release())) {}

  /// @brief Get the rank of the tensor
  /// @return The rank of the tensor
  std::size_t rank() const { return pimpl->rank(); }

  /// @brief Get the total size of the tensor
  /// @return The total number of elements in the tensor
  std::size_t size() const { return pimpl->size(); }

  /// @brief Get the shape of the tensor
  /// @return A vector containing the dimensions of the tensor
  std::vector<std::size_t> shape() const { return pimpl->shape(); }

  /// @brief Access a mutable element of the tensor
  /// @param indices The indices of the element to access
  /// @return A reference to the element at the specified indices
  scalar_type &at(const std::vector<size_t> &indices) {
    if (indices.size() != rank())
      throw std::runtime_error("Invalid indices provided to tensor::at(), size "
                               "must be equal to rank.");
    return pimpl->at(indices);
  }

  /// @brief Access a constant element of the tensor
  /// @param indices The indices of the element to access
  /// @return A constant reference to the element at the specified indices
  const scalar_type &at(const std::vector<size_t> &indices) const {
    return pimpl->at(indices);
  }

  /// @brief Copy data into the tensor
  /// @param data Pointer to the source data
  /// @param shape The shape of the source data
  void copy(const scalar_type *data,
            const std::vector<std::size_t> shape = {}) {
    if (pimpl->shape().empty() && shape.empty())
      throw std::runtime_error(
          "This tensor does not have a shape yet, must provide one to copy()");

    pimpl->copy(data, pimpl->shape().empty() ? shape : pimpl->shape());
  }

  /// @brief Take ownership of the given data
  /// @param data Pointer to the source data
  /// @param shape The shape of the source data
  void take(std::unique_ptr<scalar_type[]> &data,
            const std::vector<std::size_t> shape = {}) {
    if (pimpl->shape().empty() && shape.empty())
      throw std::runtime_error(
          "This tensor does not have a shape yet, must provide one to take()");

    pimpl->take(data, pimpl->shape().empty() ? shape : pimpl->shape());
  }

  /// @brief Borrow the given data without taking ownership
  /// @param data Pointer to the source data
  /// @param shape The shape of the source data
  void borrow(const scalar_type *data,
              const std::vector<std::size_t> shape = {}) {
    if (pimpl->shape().empty() && shape.empty())
      throw std::runtime_error("This tensor does not have a shape yet, must "
                               "provide one to borrow()");

    pimpl->borrow(data, pimpl->shape().empty() ? shape : pimpl->shape());
  }

  /// @brief Get a pointer to the raw data of the tensor.
  ///
  /// This method provides direct access to the underlying data storage of the
  /// tensor. It returns a pointer to the first element of the data array.
  ///
  /// @return `scalar_type*` A pointer to the mutable data of the tensor.
  ///
  /// @note Care should be taken when directly manipulating the raw data to
  /// avoid invalidating the internal state of a tensor or violating its
  /// `invariants`.
  scalar_type *data() { return pimpl->data(); }

  /// @brief Get a `const` pointer to the raw data of the tensor.
  ///
  /// This method provides read-only access to the underlying data storage of
  /// the tensor. It returns a constant pointer to the first element of the data
  /// array.
  ///
  /// @return `const scalar_type *` A constant pointer to the immutable data of
  /// the tensor.
  ///
  /// @note This constant version ensures that the data of a tensor cannot be
  /// modified through the returned pointer, preserving `const` correctness.
  const scalar_type *data() const { return pimpl->data(); }

  void dump() const { pimpl->dump(); }
};

} // namespace cudaq
