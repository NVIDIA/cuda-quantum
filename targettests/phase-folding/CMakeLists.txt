# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

execute_process(COMMAND ${Python_EXECUTABLE} random_gen.py branch-in-loop.template --seed=1 --block-length=30-45 --rz-weight=0.5 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} OUTPUT_FILE branch-in-loop1.cpp)
execute_process(COMMAND ${Python_EXECUTABLE} random_gen.py branch-in-loop.template --seed=2 --block-length=30-45 --rz-weight=0.5 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} OUTPUT_FILE branch-in-loop2.cpp)
execute_process(COMMAND ${Python_EXECUTABLE} random_gen.py branch-in-loop.template --seed=3 --block-length=30-45 --rz-weight=0.5 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} OUTPUT_FILE branch-in-loop3.cpp)
execute_process(COMMAND ${Python_EXECUTABLE} random_gen.py branch-in-loop.template --seed=4 --block-length=30-45 --rz-weight=0.5 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} OUTPUT_FILE branch-in-loop4.cpp)
execute_process(COMMAND ${Python_EXECUTABLE} random_gen.py branch-in-loop.template --seed=5 --block-length=30-45 --rz-weight=0.5 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} OUTPUT_FILE branch-in-loop5.cpp)
execute_process(COMMAND ${Python_EXECUTABLE} random_gen.py branch-in-loop.template --seed=6 --block-length=30-45 --rz-weight=0.5 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} OUTPUT_FILE branch-in-loop6.cpp)
execute_process(COMMAND ${Python_EXECUTABLE} random_gen.py branch-in-loop.template --seed=7 --block-length=30-45 --rz-weight=0.5 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} OUTPUT_FILE branch-in-loop7.cpp)
execute_process(COMMAND ${Python_EXECUTABLE} random_gen.py branch-in-loop.template --seed=8 --block-length=30-45 --rz-weight=0.5 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} OUTPUT_FILE branch-in-loop8.cpp)
execute_process(COMMAND ${Python_EXECUTABLE} random_gen.py branch-in-loop.template --seed=9 --block-length=30-45 --rz-weight=0.5 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} OUTPUT_FILE branch-in-loop9.cpp)
execute_process(COMMAND ${Python_EXECUTABLE} random_gen.py branch-in-loop.template --seed=10 --block-length=30-45 --rz-weight=0.5 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} OUTPUT_FILE branch-in-loop10.cpp)

execute_process(COMMAND ${Python_EXECUTABLE} random_gen.py loop-with-break.template --seed=1 --block-length=20-30 --rz-weight=0.5 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} OUTPUT_FILE loop-with-break1.cpp)
execute_process(COMMAND ${Python_EXECUTABLE} random_gen.py loop-with-break.template --seed=2 --block-length=20-30 --rz-weight=0.5 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} OUTPUT_FILE loop-with-break2.cpp)
execute_process(COMMAND ${Python_EXECUTABLE} random_gen.py loop-with-break.template --seed=3 --block-length=20-30 --rz-weight=0.5 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} OUTPUT_FILE loop-with-break3.cpp)
execute_process(COMMAND ${Python_EXECUTABLE} random_gen.py loop-with-break.template --seed=4 --block-length=20-30 --rz-weight=0.5 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} OUTPUT_FILE loop-with-break4.cpp)
execute_process(COMMAND ${Python_EXECUTABLE} random_gen.py loop-with-break.template --seed=5 --block-length=20-30 --rz-weight=0.5 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} OUTPUT_FILE loop-with-break5.cpp)
execute_process(COMMAND ${Python_EXECUTABLE} random_gen.py loop-with-break.template --seed=6 --block-length=20-30 --rz-weight=0.5 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} OUTPUT_FILE loop-with-break6.cpp)
execute_process(COMMAND ${Python_EXECUTABLE} random_gen.py loop-with-break.template --seed=7 --block-length=20-30 --rz-weight=0.5 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} OUTPUT_FILE loop-with-break7.cpp)
execute_process(COMMAND ${Python_EXECUTABLE} random_gen.py loop-with-break.template --seed=8 --block-length=20-30 --rz-weight=0.5 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} OUTPUT_FILE loop-with-break8.cpp)
execute_process(COMMAND ${Python_EXECUTABLE} random_gen.py loop-with-break.template --seed=9 --block-length=20-30 --rz-weight=0.5 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} OUTPUT_FILE loop-with-break9.cpp)
execute_process(COMMAND ${Python_EXECUTABLE} random_gen.py loop-with-break.template --seed=10 --block-length=20-30 --rz-weight=0.5 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} OUTPUT_FILE loop-with-break10.cpp)

execute_process(COMMAND ${Python_EXECUTABLE} random_gen.py straight-line.template --seed=27 --block-length=100 --rz-weight=0.5 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} OUTPUT_FILE straight-line.cpp)
execute_process(COMMAND ${Python_EXECUTABLE} random_gen.py subkernel.template --seed=1 --block-length=20-30 --rz-weight=0.5 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} OUTPUT_FILE subkernel.cpp)
