#==============================================================================
# NLOPT CMake file
#
# NLopt is a free/open-source library for nonlinear optimization, providing
# a common interface for a number of different free optimization routines
# available online as well as original implementations of various other
# algorithms
# WEBSITE: http://ab-initio.mit.edu/wiki/index.php/NLopt
# AUTHOR: Steven G. Johnson
#
# This CMakeLists.txt file was created to compile NLOPT with the CMAKE utility.
# Benoit Scherrer, 2010 CRL, Harvard Medical School
# Copyright (c) 2008-2009 Children's Hospital Boston
#==============================================================================

#==============================================================================
# version
set (NLOPT_MAJOR_VERSION "2")
set (NLOPT_MINOR_VERSION "7")
set (NLOPT_BUGFIX_VERSION "1")
set (NLOPT_VERSION_STRING ${NLOPT_MAJOR_VERSION}.${NLOPT_MINOR_VERSION}.${NLOPT_BUGFIX_VERSION})
set(SO_MAJOR 0)
set(SO_MINOR 11)
set(SO_PATCH 1)
set (NLOPT_CXX ON)

include (CheckIncludeFiles)
include (CheckFunctionExists)
include (CheckTypeSize)
include (CheckCCompilerFlag)
include (CheckCXXSymbolExists)
include (CheckCXXCompilerFlag)
include (CheckLibraryExists)

check_include_file (getopt.h HAVE_GETOPT_H)
check_include_file (unistd.h HAVE_UNISTD_H)
check_include_file (stdint.h HAVE_STDINT_H)
check_include_file (time.h HAVE_TIME_H)
check_include_file (sys/time.h HAVE_SYS_TIME_H)
if (HAVE_TIME_H AND HAVE_SYS_TIME_H)
  set (TIME_WITH_SYS_TIME TRUE)
endif ()
check_function_exists (getpid HAVE_GETPID)
check_function_exists (syscall HAVE_GETTID_SYSCALL)
check_function_exists (isinf HAVE_ISINF)
check_function_exists (isnan HAVE_ISNAN)
check_function_exists (gettimeofday HAVE_GETTIMEOFDAY)
check_function_exists (qsort_r HAVE_QSORT_R)
check_function_exists (time HAVE_TIME)
check_function_exists (copysign HAVE_COPYSIGN)
check_function_exists (getopt HAVE_GETOPT)
check_type_size ("uint32_t" SIZEOF_UINT32_T)
set (HAVE_UINT32_T ${SIZEOF_UINT32_T})
check_type_size ("unsigned int" SIZEOF_UNSIGNED_INT)
check_type_size ("unsigned long" SIZEOF_UNSIGNED_LONG)

configure_file (${CMAKE_CURRENT_SOURCE_DIR}/nlopt_config.h.in ${CMAKE_CURRENT_BINARY_DIR}/nlopt_config.h IMMEDIATE)

set (NLOPT_SOURCES
  src/algs/mma/mma.c src/algs/mma/mma.h src/algs/mma/ccsa_quadratic.c
  src/algs/cobyla/cobyla.c src/algs/cobyla/cobyla.h
  src/algs/neldermead/nldrmd.c src/algs/neldermead/neldermead.h src/algs/neldermead/sbplx.c
  src/api/general.c src/api/options.c src/api/optimize.c src/api/deprecated.c src/api/nlopt-internal.h src/api/nlopt.h 
  ${CMAKE_CURRENT_SOURCE_DIR}/nlopt.hpp
  src/util/mt19937ar.c src/util/sobolseq.c src/util/soboldata.h src/util/timer.c src/util/stop.c src/util/nlopt-util.h src/util/redblack.c src/util/redblack.h src/util/qsort_r.c src/util/rescale.c
)

set (CMAKE_C_FLAGS "-fPIC ${CMAKE_C_FLAGS}")
set (CMAKE_CXX_FLAGS "-fPIC ${CMAKE_CXX_FLAGS} -Wno-suggest-override -Wno-unused-function ")

set (nlopt_lib nlopt)
add_library (${nlopt_lib} STATIC ${NLOPT_SOURCES})
target_link_libraries (${nlopt_lib} m)

target_include_directories (${nlopt_lib} PRIVATE
  ${CMAKE_CURRENT_BINARY_DIR}
  src/algs/stogo
  src/util
  src/algs/direct
  src/algs/cdirect
  src/algs/mma
  src/algs/cobyla
  src/algs/neldermead
  src/algs/bobyqa
  src/api)
