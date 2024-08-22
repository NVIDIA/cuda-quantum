#include <iostream>
#include <type_traits>
#include <vector>

#include "../include/base64.hpp"

int runtests() {
  const std::vector<int> lengths{5, 10, 100, 1024, 10000, 1000003};
  int outcome = EXIT_SUCCESS;

  std::cout << "char is "
            << (std::is_signed<char>::value ? "signed" : "unsigned")
            << std::endl;

  std::cout << "endianness is "
#if defined(__LITTLE_ENDIAN__)
            << "little endian"
#else
            << "big endian"
#endif
            << std::endl;

  for (auto& length : lengths) {
    std::string original;
    for (int i = 0; i < length; i++) {
      original += static_cast<char>(std::rand());
    }

    auto encoded = base64::to_base64(original);
    auto s = base64::from_base64(encoded);

    if (s == original) {
      std::cout << "Test passed with length " << length << std::endl;
    } else {
      std::cout << "Test FAILED with length " << length << std::endl;
      outcome = EXIT_FAILURE;
    }
  }
  return outcome;
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) {
  try {
    return runtests();
  } catch (const std::exception& e) {
    // standard exceptions
    std::cout << "Caught std::exception in main: " << e.what() << std::endl;
    return EXIT_FAILURE;
  } catch (...) {
    // everything else
    std::cout << "Caught unknown exception in main" << std::endl;
    return EXIT_FAILURE;
  }
}
