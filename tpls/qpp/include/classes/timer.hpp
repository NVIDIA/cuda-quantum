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
 * \file classes/timer.hpp
 * \brief Timing
 */

#ifndef CLASSES_TIMER_HPP_
#define CLASSES_TIMER_HPP_

namespace qpp {
/**
 * \class qpp::Timer
 * \brief Chronometer
 *
 * \tparam T Tics duration, default is std::chrono::duration<double>, i.e.
 * seconds in double precision
 * \tparam CLOCK_T Clock's type, default is std::chrono::steady_clock, not
 * affected by wall clock changes during runtime
 */
template <typename T = std::chrono::duration<double>,
          typename CLOCK_T = std::chrono::steady_clock>
class Timer : public IDisplay {
  protected:
    typename CLOCK_T::time_point start_, end_;

  public:
    /**
     * \brief Constructs an instance with the current time as the starting point
     */
    Timer() noexcept : start_{CLOCK_T::now()}, end_{start_} {}

    /**
     * \brief Default virtual destructor
     */
    ~Timer() override = default;

    /**
     * \brief Resets the chronometer
     *
     * Resets the starting/ending point to the current time
     *
     * \return Reference to the current instance
     */
    Timer& tic() noexcept {
        start_ = end_ = CLOCK_T::now();

        return *this;
    }

    /**
     * \brief Stops the chronometer
     *
     * Set the current time as the ending point
     *
     * \return Reference to the current instance
     */
    Timer& toc() noexcept {
        end_ = CLOCK_T::now();

        return *this;
    }

    /**
     * \brief Time passed in the duration specified by T
     *
     * \return Number of tics (specified by T) that passed between the
     * instantiation/reset and invocation of qpp::Timer::toc()
     */
    double tics() const noexcept {
        return static_cast<double>(
            std::chrono::duration_cast<T>(end_ - start_).count());
    }

    /**
     * \brief Duration specified by U
     *
     * \tparam U Duration, default is T, which defaults to
     * std::chrono::duration<double>, i.e., seconds in double precision
     *
     * \return Duration that passed between the instantiation/reset and
     * invocation of qpp::Timer::toc()
     */
    template <typename U = T>
    U get_duration() const noexcept {
        return std::chrono::duration_cast<U>(end_ - start_);
    }

  private:
    /**
     * \brief qpp::IDisplay::display() override
     *
     * Writes to the output stream the number of tics (specified by T) that
     * passed between the instantiation/reset and invocation of
     * qpp::Timer::toc()
     *
     * \param os Output stream passed by reference
     * \return Reference to the output stream
     */
    std::ostream& display(std::ostream& os) const override {
        return os << tics();
    }
}; /* class Timer */

} /* namespace qpp */

#endif /* CLASSES_TIMER_HPP_ */
