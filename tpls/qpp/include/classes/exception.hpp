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
 * \file classes/exception.hpp
 * \brief Exceptions
 */

#ifndef CLASSES_EXCEPTION_HPP_
#define CLASSES_EXCEPTION_HPP_

namespace qpp {
/**
 * \namespace qpp::exception
 * \brief Quantum++ exception hierarchy namespace
 */
namespace exception {
/**
 * \class qpp::exception::Exception
 * \brief Base class for generating Quantum++ custom exceptions
 *
 * Derive from this class if more exceptions are needed, making sure to override
 * qpp::exception::Exception::description() in the derived class and to inherit
 * the constructor qpp::exception::Exception::Exception(). Preferably keep your
 * newly defined exception classes in the namespace qpp::exception.
 *
 * Example:
 * \code
 * namespace qpp
 * {
 * namespace exception
 * {
 *     class ZeroSize : public Exception
 *     {
 *     public:
 *         std::string description() const override
 *         {
 *             return "Object has zero size";
 *         }
 *
 *         // inherit the base class' qpp::exception::Exception constructor
 *         using Exception::Exception;
 *     };
 * } // namespace exception
 * } // namespace qpp
 * \endcode
 */
class Exception : public std::exception {
  protected:
    std::string where_;
    mutable std::string msg_;
    std::string context_;

  public:
    /**
     * \brief Constructs an exception
     *
     * \param where Text representing where the exception occurred
     * \param context Optional context-dependent message
     */
    explicit Exception(std::string where, std::string context = {})
        : where_{std::move(where)}, msg_{}, context_{std::move(context)} {}

    /**
     * \brief Overrides std::exception::what()
     *
     * \return Exception description
     */
    const char* what() const noexcept override {
        msg_.clear();
        msg_ += where_;
        msg_ += ": ";
        msg_ += description();
        msg_ += '!';

        if (!context_.empty()) {
            msg_ += " [" + context_ + ']';
        }

        return msg_.c_str();
    }

    /**
     * \brief Exception description
     *
     * \return Exception description
     */
    virtual std::string description() const = 0;
}; /* class Exception */

inline std::string Exception::description() const {
    return "qpp::exception::Exception";
}

/**
 * \class qpp::exception::Unknown
 * \brief Unknown exception
 *
 * Thrown when no other exception is suitable (not recommended, it is better to
 * define another suitable exception type)
 */
class Unknown : public Exception {
  public:
    std::string description() const override { return "UNKNOWN EXCEPTION"; }

    using Exception::Exception;
};

/**
 * \class qpp::exception::ZeroSize
 * \brief Object has zero size exception
 *
 * Zero sized object, e.g., empty Eigen::Matrix or std::vector<> with no
 * elements
 */
class ZeroSize : public Exception {
  public:
    std::string description() const override { return "Object has zero size"; }

    using Exception::Exception;
};

/**
 * \class qpp::exception::MatrixNotSquare
 * \brief Matrix is not square exception
 *
 * Eigen::Matrix is not a square matrix
 */
class MatrixNotSquare : public Exception {
  public:
    std::string description() const override { return "Matrix is not square"; }

    using Exception::Exception;
};

/**
 * \class qpp::exception::MatrixNotCvector
 * \brief Matrix is not a column vector exception
 *
 * Eigen::Matrix is not a column vector
 */
class MatrixNotCvector : public Exception {
  public:
    std::string description() const override {
        return "Matrix is not a column vector";
    }

    using Exception::Exception;
};

/**
 * \class qpp::exception::MatrixNotRvector
 * \brief Matrix is not a row vector exception
 *
 * Eigen::Matrix is not a row vector
 */
class MatrixNotRvector : public Exception {
  public:
    std::string description() const override {
        return "Matrix is not a row vector";
    }

    using Exception::Exception;
};

/**
 * \class qpp::exception::MatrixNotVector
 * \brief Matrix is not a vector exception
 *
 * Eigen::Matrix is not a row or column vector
 */
class MatrixNotVector : public Exception {
  public:
    std::string description() const override {
        return "Matrix is not a vector";
    }

    using Exception::Exception;
};

/**
 * \class qpp::exception::MatrixNotSquareNorCvector
 * \brief Matrix is not square nor column vector exception
 *
 * Eigen::Matrix is not a square matrix nor a column vector
 */
class MatrixNotSquareNorCvector : public Exception {
  public:
    std::string description() const override {
        return "Matrix is not square nor column vector";
    }

    using Exception::Exception;
};

/**
 * \class qpp::exception::MatrixNotSquareNorRvector
 * \brief Matrix is not square nor row vector exception
 *
 * Eigen::Matrix is not a square matrix nor a row vector
 */
class MatrixNotSquareNorRvector : public Exception {
  public:
    std::string description() const override {
        return "Matrix is not square nor row vector";
    }

    using Exception::Exception;
};

/**
 * \class qpp::exception::MatrixNotSquareNorVector
 * \brief Matrix is not square nor vector exception
 *
 * Eigen::Matrix is not a square matrix nor a row/column vector
 */
class MatrixNotSquareNorVector : public Exception {
  public:
    std::string description() const override {
        return "Matrix is not square nor vector";
    }

    using Exception::Exception;
};

/**
 * \class qpp::exception::MatrixMismatchSubsys
 * \brief Matrix mismatch subsystems exception
 *
 * Matrix size mismatch subsystem sizes (e.g., in qpp::apply())
 */
class MatrixMismatchSubsys : public Exception {
  public:
    std::string description() const override {
        return "Matrix mismatch subsystems";
    }

    using Exception::Exception;
};

/**
 * \class qpp::exception::DimsInvalid
 * \brief Invalid dimension(s) exception
 *
 * std::vector<idx> of dimensions has zero size or contains zeros, or any other
 * related exception where some dimension(s) is(are) invalid
 */
class DimsInvalid : public Exception {
  public:
    std::string description() const override { return "Invalid dimension(s)"; }

    using Exception::Exception;
};

/**
 * \class qpp::exception::DimsNotEqual
 * \brief Dimensions not equal exception
 *
 * Local/global dimensions are not equal
 */
class DimsNotEqual : public Exception {
  public:
    std::string description() const override { return "Dimensions not equal"; }

    using Exception::Exception;
};

/**
 * \class qpp::exception::DimsMismatchMatrix
 * \brief Dimension(s) mismatch matrix size exception
 *
 * Product of the elements of std::vector<idx> of dimensions is not equal to
 * the number of rows of the Eigen::Matrix (assumed to be a square matrix)
 */
class DimsMismatchMatrix : public Exception {
  public:
    std::string description() const override {
        return "Dimension(s) mismatch matrix size";
    }

    using Exception::Exception;
};

/**
 * \class qpp::exception::DimsMismatchCvector
 * \brief Dimension(s) mismatch column vector size exception
 *
 * Product of the elements of std::vector<idx> of dimensions is not equal to
 * the number of elements of the Eigen::Matrix (assumed to be a column vector)
 */
class DimsMismatchCvector : public Exception {
  public:
    std::string description() const override {
        return "Dimension(s) mismatch column vector size";
    }

    using Exception::Exception;
};

/**
 * \class qpp::exception::DimsMismatchRvector
 * \brief Dimension(s) mismatch row vector size exception
 *
 * Product of the elements of std::vector<idx> of dimensions is not equal to
 * the number of elements of the Eigen::Matrix (assumed to be a row vector)
 */
class DimsMismatchRvector : public Exception {
  public:
    std::string description() const override {
        return "Dimension(s) mismatch row vector size";
    }

    using Exception::Exception;
};

/**
 * \class qpp::exception::DimsMismatchVector
 * \brief Dimension(s) mismatch vector size exception
 *
 * Product of the elements of std::vector<idx> of dimensions is not equal to
 * the number of elements of the Eigen::Matrix (assumed to be a row/column
 * vector)
 */
class DimsMismatchVector : public Exception {
  public:
    std::string description() const override {
        return "Dimension(s) mismatch vector size";
    }

    using Exception::Exception;
};

/**
 * \class qpp::exception::SubsysMismatchDims
 * \brief Subsystems mismatch dimensions exception
 *
 * std::vector<idx> of subsystem labels has duplicates, or has entries that are
 * larger than the size of the std::vector<idx> of dimensions
 */
class SubsysMismatchDims : public Exception {
  public:
    std::string description() const override {
        return "Subsystems mismatch dimensions";
    }

    using Exception::Exception;
};

/**
 * \class qpp::exception::PermInvalid
 * \brief Invalid permutation exception
 *
 * std::vector<idx> does note represent a valid permutation
 */
class PermInvalid : public Exception {
  public:
    std::string description() const override { return "Invalid permutation"; }

    using Exception::Exception;
};

/**
 * \class qpp::exception::PermMismatchDims
 * \brief Permutation mismatch dimensions exception
 *
 * Size of the std::vector<idx> representing the permutation is different from
 * the size of the std::vector<idx> of dimensions
 */
class PermMismatchDims : public Exception {
  public:
    std::string description() const override {
        return "Permutation mismatch dimensions";
    }

    using Exception::Exception;
};

/**
 * \class qpp::exception::NotQubitMatrix
 * \brief Matrix is not 2 x 2 exception
 *
 * Eigen::Matrix is not 2 x 2
 */
class NotQubitMatrix : public Exception {
  public:
    std::string description() const override { return "Matrix is not 2 x 2"; }

    using Exception::Exception;
};

/**
 * \class qpp::exception::NotQubitCvector
 * \brief Column vector is not 2 x 1 exception
 *
 * Eigen::Matrix is not 2 x 1
 */
class NotQubitCvector : public Exception {
  public:
    std::string description() const override {
        return "Column vector is not 2 x 1";
    }

    using Exception::Exception;
};

/**
 * \class qpp::exception::NotQubitRvector
 * \brief Row vector is not 1 x 2 exception
 *
 * Eigen::Matrix is not 1 x 2
 */
class NotQubitRvector : public Exception {
  public:
    std::string description() const override {
        return "Row vector is not 1 x 2";
    }

    using Exception::Exception;
};

/**
 * \class qpp::exception::NotQubitVector
 * \brief Vector is not 2 x 1 nor 1 x 2 exception
 *
 * Eigen::Matrix is not 2 x 1 nor 1 x 2
 */
class NotQubitVector : public Exception {
  public:
    std::string description() const override {
        return "Vector is not 2 x 1 nor 1 x 2";
    }

    using Exception::Exception;
};

/**
 * \class qpp::exception::NotQubitSubsys
 * \brief Subsystems are not qubits exception
 *
 * Subsystems are not 2-dimensional (qubits)
 */
class NotQubitSubsys : public Exception {
  public:
    std::string description() const override {
        return "Subsystems are not qubits";
    }

    using Exception::Exception;
};

/**
 * \class qpp::exception::NotBipartite
 * \brief Not bi-partite exception
 *
 * std::vector<idx> of dimensions has size different from 2
 */
class NotBipartite : public Exception {
  public:
    std::string description() const override { return "Not bi-partite"; }

    using Exception::Exception;
};

/**
 * \class qpp::exception::NoCodeword
 * \brief Codeword does not exist exception
 *
 * Codeword does not exist, thrown when calling qpp::Codes::codeword() with an
 * invalid index
 */
class NoCodeword : public Exception {
  public:
    std::string description() const override {
        return "Codeword does not exist";
    }

    using Exception::Exception;
};

/**
 * \class qpp::exception::OutOfRange
 * \brief Argument out of range exception
 *
 * Argument out of range
 */
class OutOfRange : public Exception {
  public:
    std::string description() const override { return "Argument out of range"; }

    using Exception::Exception;
};

/**
 * \class qpp::exception::TypeMismatch
 * \brief Type mismatch exception
 *
 * Scalar types do not match
 */
class TypeMismatch : public Exception {
  public:
    std::string description() const override { return "Type mismatch"; }

    using Exception::Exception;
};

/**
 * \class qpp::exception::SizeMismatch
 * \brief Size mismatch exception
 *
 * Sizes do not match
 */
class SizeMismatch : public Exception {
  public:
    std::string description() const override { return "Size mismatch"; }

    using Exception::Exception;
};

/**
 * \class qpp::exception::UndefinedType
 * \brief Not defined for this type exception
 *
 * Templated specialization is not defined for this type
 */
class UndefinedType : public Exception {
  public:
    std::string description() const override {
        return "Not defined for this type";
    }

    using Exception::Exception;
};

/**
 * \class qpp::exception::QuditAlreadyMeasured
 * \brief Qudit was already measured exception
 *
 * The qudit was already measured (destructively) and cannot be measured again
 */
class QuditAlreadyMeasured : public Exception {
  public:
    std::string description() const override {
        return "Qudit was already measured (destructively)";
    }

    using Exception::Exception;
};

/**
 * \class qpp::exception::Duplicates
 * \brief System (e.g., std::vector<>) contains duplicates exception
 *
 * System (vector/matrix etc.) contains duplicate elements
 *
 */
class Duplicates : public Exception {
  public:
    std::string description() const override {
        return "System (e.g., std::vector<>) contains duplicates";
    }

    using Exception::Exception;
};

/**
 * \class qpp::exception::CustomException
 * \brief Custom exception
 *
 * Custom exception, the user must provide a custom message
 */
class CustomException : public Exception {
  public:
    std::string description() const override { return "Custom exception"; }

    using Exception::Exception;
};

/**
 * \class qpp::exception::NotImplemented
 * \brief Code not yet implemented
 */
class NotImplemented : public Exception {
  public:
    std::string description() const override { return "Not yet implemented"; }

    using Exception::Exception;
};

/**
 * \class qpp::exception::InvalidIterator
 * \brief Invalid iterator
 */
class InvalidIterator : public Exception {
  public:
    std::string description() const override { return "Invalid iterator"; }

    using Exception::Exception;
};

} /* namespace exception */
} /* namespace qpp */

#endif /* CLASSES_EXCEPTION_HPP_ */
