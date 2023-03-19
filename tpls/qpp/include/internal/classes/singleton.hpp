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
 * \file internal/classes/singleton.hpp
 * \brief Singleton pattern via CRTP
 */

#ifndef INTERNAL_CLASSES_SINGLETON_HPP_
#define INTERNAL_CLASSES_SINGLETON_HPP_

namespace qpp::internal // internal class, do not modify
{
/**
 * \class qpp::internal::Singleton
 * \brief Singleton policy class, used internally to implement the singleton
 * pattern via CRTP (Curiously Recurring Template Pattern), thread safe in
 * C++11 and later.
 *
 * To implement a singleton, derive your class from qpp::internal::Singleton,
 * make qpp::internal::Singleton a friend of your class, then declare the
 * constructor and destructor of your class as private. To get an instance, use
 * the static member function qpp::internal::Singleton::get_instance(), which
 * returns a thread_local reference (if the compiler supports thread_local), or
 * otherwise a non thread_local reference to your newly created singleton. For a
 * more fine-grained control, use
 * qpp::internal::Singleton::get_thread_local_instance() or
 * qpp::internal::Singleton::get_no_thread_local_instance().
 *
 * Example:
 * \code
 * class MySingleton: public qpp::internal::Singleton<MySingleton>
 * {
 * 	   friend class qpp::internal::Singleton<MySingleton>;
 * public:
 *     // Declare all public members here
 * private:
 *     MySingleton()
 *     {
 *         // Implement the constructor here
 *     }
 *     ~MySingleton()
 *     {
 *         // Implement the destructor here
 *     }
 * };
 *
 * MySingleton& mySingleton = MySingleton::get_instance(); // Get an instance
 * thread_local MySingleton& tls = MySingleton::get_thread_local_instance();
 * // Get a thread_local instance
 *
 * \endcode
 *
 * \see Code of qpp::Codes, qpp::Gates, qpp::Init, qpp::RandomDevices,
 * qpp::States or qpp.h for real world examples of usage.
 */
template <typename T>
class Singleton {
  protected:
    Singleton() noexcept = default;

    virtual ~Singleton() = default; // to silence base class Singleton<T> has a
                                    // non-virtual destructor [-Weffc++]

  public:
    Singleton(const Singleton&) = delete;
    Singleton& operator=(const Singleton&) = delete;
    static T& get_no_thread_local_instance() noexcept(
        std::is_nothrow_constructible<T>::value) {
        // Guaranteed to be destroyed.
        // Instantiated on first use.
        // Thread safe in C++11.
        static T instance{};

        return instance;
    }

#ifndef NO_THREAD_LOCAL_
    static T& get_thread_local_instance() noexcept(
        std::is_nothrow_constructible<T>::value) {
        // Guaranteed to be destroyed.
        // Instantiated on first use.
        // Thread safe in C++11.
        thread_local static T instance{};

        return instance;
    }
#endif // NO_THREAD_LOCAL_

    static T& get_instance() noexcept(std::is_nothrow_constructible<T>::value) {
#ifdef NO_THREAD_LOCAL_
        return get_no_thread_local_instance();
#else
        return get_thread_local_instance();
#endif // NO_THREAD_LOCAL_
    }
}; /* class Singleton */

} /* namespace qpp::internal */

#endif /* INTERNAL_CLASSES_SINGLETON_HPP_ */
