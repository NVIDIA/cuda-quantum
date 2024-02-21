include(FetchContent)
FetchContent_Declare(
    asio
    GIT_REPOSITORY    https://github.com/chriskohlhoff/asio.git
    GIT_TAG           asio-1-28-2
)
FetchContent_GetProperties(asio)
if(NOT asio_POPULATED)
  FetchContent_Populate(asio)
endif()

# Crow suffers from this issue when using libc++:
# https://github.com/CrowCpp/Crow/issues/576
# A workaround is applied depending on operating system rather then standard library specific.
# Instead of OS specific fix, detect in code whether _LIBCPP_VERSION is defined and apply fix.
FetchContent_Declare(
    crow_cpp
    GIT_REPOSITORY    https://github.com/CrowCpp/Crow.git
    GIT_TAG           4f3f5deaaa01825c63c83431bfa96ccec195f741
)
FetchContent_GetProperties(crow_cpp)
if(NOT crow_cpp_POPULATED)
  FetchContent_Populate(crow_cpp)
endif()

add_library(rest_server_impl OBJECT  RestServer.cpp)
target_include_directories(rest_server_impl PRIVATE ../ ${crow_cpp_SOURCE_DIR}/include ${asio_SOURCE_DIR}/asio/include)
