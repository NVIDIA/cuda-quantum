Namespace and Standard
**********************

**[1]** The CUDA-Q language is a library-based extension for C++. As such, all
pertinent data structures and library functions exist within the CUDA-Q
namespace. This namespace is denoted :code:`::cudaq`.

**[2]** The :code:`cudaq::contrib` namespace contains features contributed by
the larger community. While perfectly functional, these features may not have
been fully optimized for all possible use cases. In addition, they might have
breaking changes or can be removed in future releases.

**[3]** CUDA-Q is a C++ library-based language extension adherent to the `C++20 <https://en.cppreference.com/w/cpp/20>`_
language specification. Any implementations that require backwards compatibility to previous 
standards must retain the semantics of the user-facing API (e.g. `SFINAE <https://en.cppreference.com/w/cpp/language/sfinae>`_ 
instead of `concepts and constraints <https://en.cppreference.com/w/cpp/language/constraints>`_).