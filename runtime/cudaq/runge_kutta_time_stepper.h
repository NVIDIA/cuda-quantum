/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "base_time_stepper.h"
#include <functional>

namespace cudaq {
template <typename TState>
class RungeKuttaTimeStepper : public BaseTimeStepper<TState> {
public:
    using DerivativeFunction = std::function<TState(const TState &, double)>;

    RungeKuttaTimeStepper(DerivativeFunction f) : derivativeFunc(f) {}

    void compute(TState &state, double t, double dt = 0.01) override {
        // 4th order Runge-Kutta method
        TState k1 = derivativeFunc(state, t);
        TState k2 = derivativeFunc(state + (dt / 2.0) * k1, t + dt / 2.0);
        TState k3 = derivativeFunc(state + (dt / 2.0) * k2, t + dt / 2.0);
        TState k4 = derivativeFunc(state + dt * k3, t + dt);

        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4);
    }

private:
    DerivativeFunction derivativeFunc;
};
}