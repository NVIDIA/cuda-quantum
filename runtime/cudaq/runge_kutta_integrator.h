/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "base_integrator.h"
#include "runge_kutta_time_stepper.h"
#include <memory>

namespace cudaq {
template <typename TState>
class RungeKuttaIntegrator : public BaseIntegrator<TState> {
public:
    using DerivativeFunction = std::function<TState(const TState &, double)>;

    explicit RungeKuttaIntegrator(DerivativeFunction f) : stepper(std::make_shared<RungeKuttaTimeStepper<TState>>(f)) {}

    // Initializes the integrator
    void post_init() override {
        if (!this->stepper) {
            throw std::runtime_error("Time stepper is not set");
        }
    }

    // Advances the system's state from current time to `t`
    void integrate(double target_t) override {
        if (!this->schedule || !this->hamiltonian) {
            throw std::runtime_error("System is not properly set!");
        }

        while (this->t < target_t) {
            stepper->compute(this->state, this->t);
            // Time step size
            this->t += 0.01;
        }
    }

private:
    std::shared_ptr<RungeKuttaTimeStepper<TState>> stepper;
};
}