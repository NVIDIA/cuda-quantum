Namespace and Standard
**********************

**[1]** The CUDA-Q language is a library-based extension for C++. As such, all
pertinent data structures and library functions exist within the CUDA-Q
namespace. This namespace is denoted :code:`::cudaq`. 

**[2]** CUDA-Q is a C++ library-based language extension adherent to the `C++20 <https://en.cppreference.com/w/cpp/20>`_
language specification. Any implementations that require backwards compatibility to previous 
standards must retain the semantics of the user-facing API (e.g. `SFINAE <https://en.cppreference.com/w/cpp/language/sfinae>`_ 
instead of `concepts and constraints <https://en.cppreference.com/w/cpp/language/constraints>`_).