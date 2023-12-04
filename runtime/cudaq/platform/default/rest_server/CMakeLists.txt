set(LIBRARY_NAME cudaq-platform-remote-sim)
add_library(${LIBRARY_NAME} SHARED RemoteRESTQPU.cpp ../../common/QuantumExecutionQueue.cpp)
target_include_directories(${LIBRARY_NAME} 
    PUBLIC 
       $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>
       $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
       $<INSTALL_INTERFACE:include>
    PRIVATE . ../../)

target_link_libraries(${LIBRARY_NAME}
  PUBLIC 
    cudaq-em-default 
    cudaq-spin 
    cudaq-common 
  PRIVATE 
    cudaq
    cudaq-mlir-runtime 
    MLIRTranslateLib
    pthread
    spdlog::spdlog 
    fmt::fmt-header-only 
)

install(TARGETS ${LIBRARY_NAME} DESTINATION lib)
install(TARGETS ${LIBRARY_NAME} EXPORT cudaq-platform-remote-sim-targets DESTINATION lib)

add_target_config(remote-sim)

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

FetchContent_Declare(
    crow_cpp
    GIT_REPOSITORY    https://github.com/CrowCpp/Crow.git
    GIT_TAG           4f3f5deaaa01825c63c83431bfa96ccec195f741
)
FetchContent_GetProperties(crow_cpp)
if(NOT crow_cpp_POPULATED)
  FetchContent_Populate(crow_cpp)
endif()

set(LLVM_LINK_COMPONENTS Support ${LLVM_TARGETS_TO_BUILD})
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-type-limits")

add_executable(cudaq_rest_server RestServerMain.cpp)
target_include_directories(cudaq_rest_server PRIVATE ${crow_cpp_SOURCE_DIR}/include ${asio_SOURCE_DIR}/asio/include)
target_link_libraries(cudaq_rest_server
  PRIVATE
  ${dialect_libs}
  ${translation_libs}
  ${conversion_libs}
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRTranslateLib
  MLIRSupport
  MLIROptLib
  MLIRExecutionEngine
  MLIRTransforms
  MLIRTargetLLVMIRExport
  MLIRLLVMCommonConversion
  MLIRLLVMToLLVMIRTranslation
  CCDialect
  OptCodeGen
  OptTransforms
  QuakeDialect
)
target_link_libraries(cudaq_rest_server
  PRIVATE 
    cudaq-mlir-runtime 
    cudaq-common 
    spdlog::spdlog 
    fmt::fmt-header-only) 

mlir_check_all_link_libraries(cudaq_rest_server)

install(TARGETS cudaq_rest_server DESTINATION bin)