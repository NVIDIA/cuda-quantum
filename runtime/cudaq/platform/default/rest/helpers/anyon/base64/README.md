# base64
A simple approach to convert strings from and to base64.
Header only c++ library (single header).

## Usage

```cpp
#include <iostream>

#include "base64.hpp"

int main() {
  auto encoded_str =  base64::to_base64("Hello, World!");
  std::cout << encoded_str << std::endl; // SGVsbG8sIFdvcmxkIQ==
  auto decoded_str = base64::from_base64("SGVsbG8sIFdvcmxkIQ==");
  std::cout << decoded_str << std::endl; // Hello, World!
}
```

## Notes
This library relies on C++17 but will exploit some C++20 features if available (e.g. `bit_cast`).

There are many implementations available and it may be worth looking at those. A benchmark of various c/c++ base64 implementations can be found at https://github.com/gaspardpetit/base64/

This implementation here adopts the approach of Nick Galbreath's `modp_b64` library also used by chromium (e.g.  https://github.com/chromium/chromium/tree/main/third_party/modp_b64 ) but offers it as a c++ single header file. This choice was based on the good computational performance of the underpinning algorithm. We also decided to avoid relying on a c++ `union` to perform type punning as this, while working in practice, is strictly speaking undefined behaviour in c++: https://en.wikipedia.org/wiki/Type_punning#Use_of_union

Faster c/c++ implementations exist althrough these likely exploit simd / openmp or similar acceleration techniques:
- https://github.com/aklomp/base64
- https://github.com/lemire/fastbase64 (From a [blog post](https://lemire.me/blog/2018/01/17/ridiculously-fast-base64-encoding-and-decoding/) by the authors: "My understanding is that our good results have been integrated in [Klomp’s base64 library](https://github.com/aklomp/base64).")
- Other implementations related to the one by lemire: https://github.com/WojciechMula/base64-avx512 and https://github.com/WojciechMula/base64simd
- https://github.com/powturbo/Turbo-Base64 (Note that this is licensed under GPL 3.0)

Many other C++ centric appraches exists although they seem to focus on readibility or genericity at the cost of performance, e.g.:
-  https://github.com/matheusgomes28/base64pp (C++20 library from which we borrowed the unit test code)
-  https://github.com/ReneNyffenegger/cpp-base64 (Implementation that works with older C++ versions)
-  https://github.com/azawadzki/base-n (more generic baseN such as N=16 and N=32)
