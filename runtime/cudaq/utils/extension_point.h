/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <functional>
#include <memory>
#include <unordered_map>

namespace cudaq {

/// @brief A template class for implementing an extension point mechanism.
///
/// This class provides a framework for registering and retrieving plugin-like
/// extensions. It allows dynamic creation of objects based on registered types.
///
/// @tparam T The base type of the extensions.
/// @tparam CtorArgs Variadic template parameters for constructor arguments.
///
/// How to use the extension_point class
///
/// The extension_point class provides a mechanism for creating extensible
/// frameworks with plugin-like functionality. Here's how to use it:
///
/// 1. Define your extension point:
///    Create a new class that inherits from cudaq::extension_point<YourClass>.
///    This class should declare pure virtual methods that extensions will
///    implement.
///
/// @code
/// class MyExtensionPoint : public cudaq::extension_point<MyExtensionPoint> {
/// public:
///   virtual std::string parrotBack(const std::string &msg) const = 0;
/// };
/// @endcode
///
/// 2. Implement concrete extensions:
///    Create classes that inherit from your extension point and implement its
///    methods. Use the CUDAQ_EXTENSION_CREATOR_FUNCTION macro to define a
///    creator function.
///
/// @code
/// class RepeatBackOne : public MyExtensionPoint {
/// public:
///   std::string parrotBack(const std::string &msg) const override {
///     return msg + " from RepeatBackOne.";
///   }
///
///   CUDAQ_EXTENSION_CREATOR_FUNCTION(MyExtensionPoint, RepeatBackOne)
/// };
/// @endcode
///
/// 3. Register your extensions:
///    Use the CUDAQ_REGISTER_TYPE macro to register each extension.
///
/// @code
/// CUDAQ_REGISTER_TYPE(RepeatBackOne)
/// @endcode
///
/// 4. Use your extensions:
///    You can now create instances of your extensions, check registrations, and
///    more.
///
/// @code
/// auto extension = MyExtensionPoint::get("RepeatBackOne");
/// std::cout << extension->parrotBack("Hello") << std::endl;
///
/// auto registeredTypes = MyExtensionPoint::get_registered();
/// bool isRegistered = MyExtensionPoint::is_registered("RepeatBackOne");
/// @endcode
///
/// This approach allows for a flexible, extensible design where new
/// functionality can be added without modifying existing code.
template <typename T, typename... CtorArgs>
class extension_point {

  /// Type alias for the creator function.
  using CreatorFunction = std::function<std::unique_ptr<T>(CtorArgs...)>;

protected:
  /// @brief Get the registry of creator functions.
  /// @return A reference to the static registry map.
  static std::unordered_map<std::string, CreatorFunction> &get_registry() {
    static std::unordered_map<std::string, CreatorFunction> registry;
    return registry;
  }

public:
  /// @brief Create an instance of a registered extension.
  /// @param name The identifier of the registered extension.
  /// @param args Constructor arguments for the extension.
  /// @return A unique pointer to the created instance.
  /// @throws std::runtime_error if the extension is not found.
  static std::unique_ptr<T> get(const std::string &name, CtorArgs... args) {
    auto &registry = get_registry();
    auto iter = registry.find(name);
    if (iter == registry.end())
      throw std::runtime_error(
          std::string("Cannot find extension with name = ") + name);

    return iter->second(std::forward<CtorArgs>(args)...);
  }

  /// @brief Get a list of all registered extension names.
  /// @return A vector of registered extension names.
  static std::vector<std::string> get_registered() {
    std::vector<std::string> names;
    auto &registry = get_registry();
    for (auto &[k, v] : registry)
      names.push_back(k);
    return names;
  }

  /// @brief Check if an extension is registered.
  /// @param name The identifier of the extension to check.
  /// @return True if the extension is registered, false otherwise.
  static bool is_registered(const std::string &name) {
    auto &registry = get_registry();
    return registry.find(name) != registry.end();
  }
};

/// @brief Macro for defining a creator function for an extension.
/// @param BASE The base class of the extension.
/// @param TYPE The derived class implementing the extension.
#define CUDAQ_EXTENSION_CREATOR_FUNCTION(BASE, TYPE)                           \
  static inline bool register_type() {                                         \
    auto &registry = get_registry();                                           \
    registry[TYPE::class_identifier] = TYPE::create;                           \
    return true;                                                               \
  }                                                                            \
  static const bool registered_;                                               \
  static inline const std::string class_identifier = #TYPE;                    \
  static std::unique_ptr<BASE> create() { return std::make_unique<TYPE>(); }

/// @brief Macro for defining a custom creator function for an extension.
/// @param TYPE The class implementing the extension.
/// @param ... Custom implementation of the create function.
#define CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION(TYPE, ...)                     \
  static inline bool register_type() {                                         \
    auto &registry = get_registry();                                           \
    registry[TYPE::class_identifier] = TYPE::create;                           \
    return true;                                                               \
  }                                                                            \
  static const bool registered_;                                               \
  static inline const std::string class_identifier = #TYPE;                    \
  __VA_ARGS__

#define CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION_WITH_NAME(TYPE, NAME, ...)     \
  static inline bool register_type() {                                         \
    auto &registry = TYPE::get_registry();                                     \
    registry.insert({NAME, TYPE::create});                                     \
    return true;                                                               \
  }                                                                            \
  static const bool registered_;                                               \
  static inline const std::string class_identifier = #TYPE;                    \
  __VA_ARGS__

/// @brief Macro for registering an extension type.
/// @param TYPE The class to be registered as an extension.
#define CUDAQ_REGISTER_TYPE(TYPE)                                              \
  const bool TYPE::registered_ = TYPE::register_type();

} // namespace cudaq