# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

configure_file(IQMStartServerAndTest.sh.in IQMStartServerAndTest.sh @ONLY)
get_property(backend_unittest_libs DIRECTORY PROPERTY BACKEND_UNITTEST_LIBS)

## General IQM tests and IQM token tests
add_backend_unittest_executable(test_iqm
  SOURCES IQMTester.cpp
  BACKEND iqm
  BACKEND_CONFIG "iqm emulate=false url=http://localhost:62443"
  LINK_LIBS ${backend_unittest_libs} gmock_main
)
add_test(NAME iqm-basic-tests COMMAND bash IQMStartServerAndTest.sh test_iqm WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_property(TEST iqm-basic-tests PROPERTY SKIP_RETURN_CODE 77)

## IQM dynamic architecture file save/load
add_backend_unittest_executable(test_iqm_save_load
  SOURCES IQMDqaSaveTester.cpp
  BACKEND iqm
  BACKEND_CONFIG "iqm emulate=false url=http://localhost:62443"
  LINK_LIBS ${backend_unittest_libs} gmock_main
)
configure_file(TestDqaSaveLoad.sh.in TestDqaSaveLoad.sh
FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
@ONLY)
add_test(NAME iqm-dqa-save-load-tests COMMAND bash IQMStartServerAndTest.sh TestDqaSaveLoad.sh WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_property(TEST iqm-dqa-save-load-tests PROPERTY SKIP_RETURN_CODE 77)

## Mapping file in configuration
set(dqa_filename "dqa_mock_qpu.txt")
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/${dqa_filename} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
add_backend_unittest_executable(test_iqm_dqa
  SOURCES IQMDqaTester.cpp
  BACKEND iqm
  BACKEND_CONFIG "iqm emulate=false url=http://localhost:62443 mapping_file=${dqa_filename}"
  LINK_LIBS ${backend_unittest_libs} gmock_main
)

add_test(NAME iqm-dqa-tests COMMAND bash IQMStartServerAndTest.sh test_iqm_dqa WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_property(TEST iqm-dqa-tests PROPERTY SKIP_RETURN_CODE 77)


## Tests of split-topology handling
configure_file(dqa_crystal-20_split10-10.txt.in
               dqa_crystal-20_split10-10.txt
               NO_SOURCE_PERMISSIONS COPYONLY)
configure_file(dqa_crystal-20_split9-10.txt.in
               dqa_crystal-20_split9-10.txt
               NO_SOURCE_PERMISSIONS COPYONLY)
configure_file(dqa_crystal-20_split8-12.txt.in
               dqa_crystal-20_split8-12.txt
               NO_SOURCE_PERMISSIONS COPYONLY)
configure_file(dqa_crystal-20_split-3-8-2-2-5.txt.in
               dqa_crystal-20_split-3-8-2-2-5.txt
               NO_SOURCE_PERMISSIONS COPYONLY)
configure_file(dqa_crystal-20_snake17.txt.in
               dqa_crystal-20_snake17.txt
               NO_SOURCE_PERMISSIONS COPYONLY)
configure_file(dqa_crystal-20_snake18.txt.in
               dqa_crystal-20_snake18.txt
               NO_SOURCE_PERMISSIONS COPYONLY)
configure_file(dqa_crystal-20_snake20.txt.in
               dqa_crystal-20_snake20.txt
               NO_SOURCE_PERMISSIONS COPYONLY)
configure_file(dqa_crystal-54_split31-23.txt.in
               dqa_crystal-54_split31-23.txt
               NO_SOURCE_PERMISSIONS COPYONLY)
configure_file(dqa_crystal-54_split23-22.txt.in
               dqa_crystal-54_split23-22.txt
               NO_SOURCE_PERMISSIONS COPYONLY)
configure_file(dqa_crystal-54_split22-22.txt.in
               dqa_crystal-54_split22-22.txt
               NO_SOURCE_PERMISSIONS COPYONLY)
configure_file(dqa_crystal-54_halo32.txt.in
               dqa_crystal-54_halo32.txt
               NO_SOURCE_PERMISSIONS COPYONLY)
configure_file(dqa_crystal-54_halo33.txt.in
               dqa_crystal-54_halo33.txt
               NO_SOURCE_PERMISSIONS COPYONLY)
configure_file(dqa_crystal-54_hsnake50.txt.in
               dqa_crystal-54_hsnake50.txt
               NO_SOURCE_PERMISSIONS COPYONLY)
configure_file(dqa_crystal-54_vsnake49.txt.in
               dqa_crystal-54_vsnake49.txt
               NO_SOURCE_PERMISSIONS COPYONLY)

add_backend_unittest_executable(test_iqm_split
  SOURCES IQMTestSplitTopology.cpp
  BACKEND iqm
  BACKEND_CONFIG "iqm emulate=false url=http://localhost:62443"
  LINK_LIBS ${backend_unittest_libs} gmock_main
)
target_link_libraries(test_iqm_split PRIVATE cpr::cpr)

add_test(NAME iqm-split-tests COMMAND bash IQMStartServerAndTest.sh test_iqm_split WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_property(TEST iqm-split-tests PROPERTY ENVIRONMENT_MODIFICATION "IQM_SAVE_QPU_QA=set:dqa_received.txt")
set_property(TEST iqm-split-tests PROPERTY SKIP_RETURN_CODE 77)


# Tests are sharing the same port
set_tests_properties(
  iqm-basic-tests
  iqm-dqa-save-load-tests
  iqm-dqa-tests
  iqm-split-tests
  PROPERTIES RESOURCE_LOCK "iqm_port")
