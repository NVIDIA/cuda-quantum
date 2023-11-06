find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_library(CUTENSORNET_LIB
    NAMES   libcutensornet.so.2  libcutensornet.so
    HINTS   
        ${CUTENSORNET_ROOT}/lib64
        ${CUTENSORNET_ROOT}/lib
        ${CUTENSORNET_ROOT}/lib64/${CUDA_VERSION_MAJOR}
        ${CUTENSORNET_ROOT}/lib/${CUDA_VERSION_MAJOR}
)

find_file(CUTENSORNET_INC
    NAMES   cutensornet.h
    HINTS   
        ${CUTENSORNET_ROOT}/include      
        /usr/include    
        ENV CPATH
)

if(NOT CUTENSORNET_LIB OR NOT CUTENSORNET_INC)
  message(FATAL_ERROR "\nUnable to find cutensornet installation. Please ensure it is correctly installed and available on path.")
endif()
message(STATUS "CUTENSORNET_LIB: ${CUTENSORNET_LIB}")
message(STATUS "CUTENSORNET_INC: ${CUTENSORNET_INC}")
get_filename_component(CUTENSORNET_INCLUDE_DIR ${CUTENSORNET_INC} DIRECTORY)

# Determine cutensornet version
file(READ "${CUTENSORNET_INC}" cutensornet_header)
string(REGEX MATCH "CUTENSORNET_MAJOR ([0-9]*)" _ ${cutensornet_header})
set(CUTENSORNET_MAJOR ${CMAKE_MATCH_1})

string(REGEX MATCH "CUTENSORNET_MINOR ([0-9]*)" _ ${cutensornet_header})
set(CUTENSORNET_MINOR ${CMAKE_MATCH_1})

string(REGEX MATCH "CUTENSORNET_PATCH ([0-9]*)" _ ${cutensornet_header})
set(CUTENSORNET_PATCH ${CMAKE_MATCH_1})

message(STATUS "Found cutensornet version: ${CUTENSORNET_MAJOR}.${CUTENSORNET_MINOR}.${CUTENSORNET_PATCH}")
# We need cutensornet v2.3+
if (${CUTENSORNET_MAJOR} GREATER_EQUAL 2 AND ${CUTENSORNET_MINOR} GREATER_EQUAL 3)
  # Create nvqir-tensornet: nvq++ -qpu tensornet code.cpp
  set (BASE_TENSOR_BACKEND_SRS  
        simulator_cutensornet.cpp 
        tensornet_spin_op.cpp
        tensornet_state.cpp)
  get_filename_component(CUTENSORNET_LIB_DIR ${CUTENSORNET_LIB} DIRECTORY)
  set(CMAKE_INSTALL_RPATH "$ORIGIN:$ORIGIN/lib:$ORIGIN/lib/plugins:$ORIGIN/../lib:$ORIGIN/../lib/plugins:${CUTENSORNET_LIB_DIR}")
  # Helper macro to add cutensornet-based backends
  macro (nvqir_create_cutn_plugin LIBRARY_NAME)
    add_library(nvqir-${LIBRARY_NAME} SHARED ${ARGN})
    target_include_directories(nvqir-${LIBRARY_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/runtime/common ${CMAKE_SOURCE_DIR}/runtime/nvqir ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${CUTENSORNET_INCLUDE_DIR})
    target_link_libraries(nvqir-${LIBRARY_NAME} PRIVATE fmt::fmt-header-only cudaq cudaq-common ${CUTENSORNET_LIB} CUDA::cudart)
    install(TARGETS nvqir-${LIBRARY_NAME} DESTINATION lib)
    file (WRITE ${CMAKE_BINARY_DIR}/targets/${LIBRARY_NAME}.config "NVQIR_SIMULATION_BACKEND=${LIBRARY_NAME}\n")
    install(FILES ${CMAKE_BINARY_DIR}/targets/${LIBRARY_NAME}.config DESTINATION targets)
  endmacro()

  nvqir_create_cutn_plugin(tensornet ${BASE_TENSOR_BACKEND_SRS} simulator_tensornet_register.cpp )
  # Locate CUTENSORNET_COMM_LIB
  if(EXISTS "$ENV{CUTENSORNET_COMM_LIB}")
    set(CUTENSORNET_COMM_LIB "$ENV{CUTENSORNET_COMM_LIB}")
  else()
    find_library(CUTENSORNET_COMM_LIB
      NAMES libcutensornet_distributed_interface_mpi.so
      HINTS   
          ${CUTENSORNET_ROOT}/lib64
          ${CUTENSORNET_ROOT}/lib
          ENV LD_LIBRARY_PATH
    )
  endif()
  if(CUTENSORNET_COMM_LIB)
    message(STATUS "CUTENSORNET_COMM_LIB: ${CUTENSORNET_COMM_LIB}")
    target_compile_definitions(nvqir-tensornet PRIVATE -DCUTENSORNET_COMM_LIB="${CUTENSORNET_COMM_LIB}")
  endif()
  nvqir_create_cutn_plugin(tensornet-mps ${BASE_TENSOR_BACKEND_SRS} simulator_mps_register.cpp)

  # Check if the cutensornet plugin lib for expectation value is provided
  if(NOT CUDAQ_CUTENSORNET_PLUGIN_LIB AND EXISTS "$ENV{CUDAQ_CUTENSORNET_PLUGIN_LIB}")
    set(CUDAQ_CUTENSORNET_PLUGIN_LIB "$ENV{CUDAQ_CUTENSORNET_PLUGIN_LIB}")
  endif()
  if (EXISTS "${CUDAQ_CUTENSORNET_PLUGIN_LIB}")
    message(STATUS "CUDA Quantum tensornet backends: link additional external plugin ${CUDAQ_CUTENSORNET_PLUGIN_LIB}")
    target_link_libraries(nvqir-tensornet PRIVATE -Wl,--whole-archive ${CUDAQ_CUTENSORNET_PLUGIN_LIB} -Wl,--no-whole-archive)
    target_link_libraries(nvqir-tensornet-mps PRIVATE -Wl,--whole-archive ${CUDAQ_CUTENSORNET_PLUGIN_LIB} -Wl,--no-whole-archive)
  endif()
else()
  message(WARNING "Skipped tensornet backend due to incompatible cutensornet version. Please install cutensornet v2.3.0+.")
endif()
