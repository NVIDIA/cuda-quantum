/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <gtest/gtest.h>
#include "cudaq/cudm_op_conversion.h"
#include "cudaq/operators.h"
#include "test_mocks.h"
#include <stdexcept>
#include <complex>
#include <memory>

using namespace cudaq;

class CuDmOpConversion : public ::testing::Test {
protected:
    cudensitymatHandle_t handle;
    std::map<int, int> dimensions;
    std::shared_ptr<cudaq::Schedule> schedule;
    std::unique_ptr<cudm_op_conversion> converter;
    std::vector<int64_t> space_mode_extents;

    void SetUp() override {
        handle = mock_handle();
        dimensions = {{0, 2}, {1, 2}};
        for (const auto &dim : dimensions) {
            space_mode_extents.push_back(dim.second);
        }
        schedule = std::shared_ptr<Schedule>();
        converter = std::make_unique<cudm_op_conversion>(handle, dimensions, schedule);
    }
};

TEST_F(CuDmOpConversion, ConstructorValid) {
    EXPECT_NO_THROW(cudm_op_conversion converter(handle, dimensions, schedule));
}

TEST_F(CuDmOpConversion, ConstructorEmptyDimensions) {
    std::map<int, int> empty_dimensions;
    EXPECT_THROW(cudm_op_conversion converter(handle, empty_dimensions, schedule), std::invalid_argument);
}

TEST_F(CuDmOpConversion, ConstructorInvalidHandle) {
    cudensitymatHandle_t invalid_handle = nullptr;
    EXPECT_THROW(cudm_op_conversion converter(invalid_handle, dimensions, schedule), std::runtime_error);
}

TEST_F(CuDmOpConversion, EvaluateScalarConstant) {
    scalar_operator scalar_op(2.5);
    auto result = converter->evaluate(scalar_op);

    ASSERT_TRUE(std::holds_alternative<std::complex<double>>(result));
    EXPECT_EQ(std::get<std::complex<double>>(result), std::complex<double>(2.5, 0.0));
}

TEST_F(CuDmOpConversion, EvaluateScalarCallback) {
    scalar_operator scalar_op([](std::map<std::string, std::complex<double>>) {
        return std::complex<double>(1.0, -1.0);
    });
    auto result = converter->evaluate(scalar_op);

    ASSERT_TRUE(std::holds_alternative<cudensitymatWrappedScalarCallback_t>(result));
}

// TEST_F(CuDmOpConversion, EvaluateMatrixOperator) {
//     matrix_operator mat_op("H", {0});
//     auto result = converter->evaluate(mat_op);

//     ASSERT_TRUE(std::holds_alternative<cudensitymatOperatorTerm_t>(result));
// }

TEST_F(CuDmOpConversion, EvaluateProductOperator) {
    auto op0 = cudaq::matrix_operator::annihilate(0);
    auto op1 = cudaq::matrix_operator::create(0);
    product_operator<matrix_operator> product_op = op0 * op1;
    EXPECT_THROW(converter->evaluate(product_op), std::runtime_error);
}

TEST_F(CuDmOpConversion, AddOperators) {
    scalar_operator scalar_op1(2.0);
    scalar_operator scalar_op2(3.0);

    auto result = converter->add(converter->evaluate(scalar_op1), converter->evaluate(scalar_op2));

    ASSERT_TRUE(std::holds_alternative<std::complex<double>>(result));
    EXPECT_EQ(std::get<std::complex<double>>(result), std::complex<double>(5.0, 0.0));
}

TEST_F(CuDmOpConversion, AddComplexScalars) {
    std::complex<double> scalar_1(2.0, 1.0);
    std::complex<double> scalar_2(3.0, -1.0);

    auto result = converter->add(scalar_1, scalar_2);

    ASSERT_TRUE(std::holds_alternative<std::complex<double>>(result));
    EXPECT_EQ(std::get<std::complex<double>>(result), std::complex<double>(5.0, 0.0));
}

// TEST_F(CuDmOpConversion, AddScalarAndOperator) {
//     scalar_operator scalar_op(1.0);
//     matrix_operator mat_op("X", {0});

//     auto scalar_result = converter->evaluate(scalar_op);
//     auto op_result = converter->evaluate(mat_op);

//     auto final_result = converter->add(scalar_result, op_result);

//     ASSERT_TRUE(std::holds_alternative<cudensitymatOperatorTerm_t>(final_result));
// }

TEST_F(CuDmOpConversion, TensorProductOfScalars) {
    auto result = converter->tensor(2.0, 3.0);
    EXPECT_TRUE(std::holds_alternative<double>(result));
    EXPECT_EQ(std::get<double>(result), 6.0);
}

// TEST_F(CuDmOpConversion, TensorProductScalarAndOperator) {
//     cudensitymatOperatorTerm_t op_term;
//     HANDLE_CUDM_ERROR(cudensitymatCreateOperatorTerm(handle, dimensions.size(), space_mode_extents.data(), &op_term));

//     auto result = converter->tensor(2.0, op_term);
//     EXPECT_TRUE(std::holds_alternative<cudensitymatOperatorTerm_t>(result));

//     HANDLE_CUDM_ERROR(cudensitymatDestroyOperatorTerm(op_term));
// }
