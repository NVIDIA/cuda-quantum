set(LIBRARY_NAME cudaq-platform-remote-sim)
add_library(${LIBRARY_NAME} SHARED RemoteRESTQPU.cpp RemoteRESTExecutionManager.cpp ../../common/QuantumExecutionQueue.cpp)
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
    nvqir
    MLIRTranslateLib
    pthread
    spdlog::spdlog 
    fmt::fmt-header-only 
)

install(TARGETS ${LIBRARY_NAME} DESTINATION lib)
install(TARGETS ${LIBRARY_NAME} EXPORT cudaq-platform-remote-sim-targets DESTINATION lib)
add_custom_target(alias_remote_platform_lib ALL COMMAND ${CMAKE_COMMAND} -E create_symlink 
  ${CMAKE_INSTALL_PREFIX}/lib/$<TARGET_FILE_NAME:${LIBRARY_NAME}> 
  ${CMAKE_INSTALL_PREFIX}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}cudaq-em-remote-sim${CMAKE_SHARED_LIBRARY_SUFFIX}
)
add_target_config(remote-sim)

if (NOT CUDAQ_DISABLE_TOOLS)
  set(LLVM_LINK_COMPONENTS Support ${LLVM_TARGETS_TO_BUILD})
  get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
  get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
  get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-type-limits")

  add_subdirectory(server_impl)
  add_subdirectory(llvm_jit)
  
  add_executable(cudaq_rest_server RestServerMain.cpp)
  target_link_libraries(cudaq_rest_server
    PRIVATE
      rest_server_impl
      jit_util
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
      cudaq-common  
      cudaq-spin  
      cudaq
      cudaq-mlir-runtime 
      nvqir
      cudaq-em-default 
      cudaq-platform-default
      spdlog::spdlog 
      fmt::fmt-header-only) 
  target_link_options(cudaq_rest_server PRIVATE -Wl,--no-as-needed)
  mlir_check_all_link_libraries(cudaq_rest_server)

  install(TARGETS cudaq_rest_server DESTINATION bin)
endif()