# Need to compile this separately with `-fno-rtti`
add_library(jit_util OBJECT JIT.cpp)
target_link_libraries(jit_util
    PRIVATE
        MLIRExecutionEngine
        cudaq-common  
        spdlog::spdlog 
        fmt::fmt-header-only
)
target_compile_options(jit_util PRIVATE -fno-rtti)