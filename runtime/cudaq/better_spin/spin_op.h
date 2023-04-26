#pragma once

#include <complex>
#include <unordered_map>
#include <vector>

namespace cudaq::v2 {

// Define friend functions for operations between spin_op and scalars.
#define CUDAQ_SPIN_SCALAR_OPERATIONS(op, U)                                    \
  friend spin_op operator op(const spin_op &lhs, const U &rhs) noexcept {      \
    spin_op nrv(lhs);                                                          \
    nrv op## = rhs;                                                            \
    return nrv;                                                                \
  }                                                                            \
  friend spin_op operator op(const spin_op &lhs, U &&rhs) noexcept {           \
    spin_op nrv(lhs);                                                          \
    nrv op## = std::move(rhs);                                                 \
    return nrv;                                                                \
  }                                                                            \
  friend spin_op &&operator op(spin_op &&lhs, const U &rhs) noexcept {         \
    lhs op## = rhs;                                                            \
    return std::move(lhs);                                                     \
  }                                                                            \
  friend spin_op &&operator op(spin_op &&lhs, U &&rhs) noexcept {              \
    lhs op## = std::move(rhs);                                                 \
    return std::move(lhs);                                                     \
  }                                                                            \
  friend spin_op operator op(const U &lhs, const spin_op &rhs) noexcept {      \
    spin_op nrv(rhs);                                                          \
    nrv op## = lhs;                                                            \
    return nrv;                                                                \
  }                                                                            \
  friend spin_op &&operator op(const U &lhs, spin_op &&rhs) noexcept {         \
    rhs op## = lhs;                                                            \
    return std::move(rhs);                                                     \
  }                                                                            \
  friend spin_op operator op(U &&lhs, const spin_op &rhs) noexcept {           \
    spin_op nrv(rhs);                                                          \
    nrv op## = std::move(lhs);                                                 \
    return nrv;                                                                \
  }                                                                            \
  friend spin_op &&operator op(U &&lhs, spin_op &&rhs) noexcept {              \
    rhs op## = std::move(lhs);                                                 \
    return std::move(rhs);                                                     \
  }

// Define friend functions for operations between two spin_ops
#define CUDAQ_SPIN_OPERATORS(op)                                               \
  friend spin_op operator op(const spin_op &lhs,                               \
                             const spin_op &rhs) noexcept {                    \
    spin_op nrv(lhs);                                                          \
    nrv op## = rhs;                                                            \
    return nrv;                                                                \
  }                                                                            \
                                                                               \
  friend spin_op &&operator op(const spin_op &lhs, spin_op &&rhs) noexcept {   \
    rhs op## = lhs;                                                            \
    return std::move(rhs);                                                     \
  }                                                                            \
                                                                               \
  friend spin_op &&operator op(spin_op &&lhs, const spin_op &rhs) noexcept {   \
    lhs op## = rhs;                                                            \
    return std::move(lhs);                                                     \
  }                                                                            \
                                                                               \
  friend spin_op &&operator op(spin_op &&lhs, spin_op &&rhs) noexcept {        \
    lhs op## = std::move(rhs);                                                 \
    return std::move(lhs);                                                     \
  }

class spin_op {
protected:
  using pauli_term = std::vector<bool>;

  std::unordered_map<pauli_term, std::complex<double>> terms;

public:
  
  spin
/// @brief Set the provided spin_op equal to this one and return *this.
  spin_op &operator=(const spin_op &);

  /// @brief Add the given spin_op to this one and return *this
  spin_op &operator+=(const spin_op &v) noexcept;

  /// @brief Subtract the given spin_op from this one and return *this
  spin_op &operator-=(const spin_op &v) noexcept;

  /// @brief Multiply the given spin_op with this one and return *this
  spin_op &operator*=(const spin_op &v) noexcept;

  /// @brief Return true if this spin_op is equal to the given one. Equality
  /// here does not consider the coefficients.
  bool operator==(const spin_op &v) const noexcept;

  /// @brief Multiply this spin_op by the given double.
  spin_op &operator*=(const double v) noexcept;

  /// @brief Multiply this spin_op by the given complex value
  spin_op &operator*=(const std::complex<double> v) noexcept;

  CUDAQ_SPIN_SCALAR_OPERATIONS(*, double)
  CUDAQ_SPIN_SCALAR_OPERATIONS(*, std::complex<double>)
  CUDAQ_SPIN_OPERATORS(+)
  CUDAQ_SPIN_OPERATORS(*)

  // Define the subtraction operators
  friend spin_op operator-(const spin_op &lhs, const spin_op &rhs) noexcept {
    spin_op nrv(lhs);
    nrv -= rhs;
    return nrv;
  }

  friend spin_op operator-(const spin_op &lhs, spin_op &&rhs) noexcept {
    spin_op nrv(lhs);
    nrv -= std::move(rhs);
    return nrv;
  }

  friend spin_op &&operator-(spin_op &&lhs, const spin_op &rhs) noexcept {
    lhs -= rhs;
    return std::move(lhs);
  }

  friend spin_op &&operator-(spin_op &&lhs, spin_op &&rhs) noexcept {
    lhs -= std::move(rhs);
    return std::move(lhs);
  }
};
} // namespace cudaq::v2