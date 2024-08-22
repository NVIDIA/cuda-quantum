#!/usr/bin/env sh
clang-format -style=Google -i include/*.hpp
clang-format -style=Google -i test/*.cpp
