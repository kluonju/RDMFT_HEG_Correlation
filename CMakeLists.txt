cmake_minimum_required(VERSION 3.16)

# Some Linux setups have clang as /usr/bin/c++ but only ship libstdc++ via g++.
# Prefer g++ when available so the build works out of the box.
if(NOT DEFINED CMAKE_CXX_COMPILER)
    find_program(GXX_BIN g++)
    if(GXX_BIN)
        set(CMAKE_CXX_COMPILER ${GXX_BIN} CACHE FILEPATH "" FORCE)
    endif()
endif()

project(rdmft_heg LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

add_compile_options(-Wall -Wextra -Wpedantic -O3)

include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(rdmft_heg src/main.cpp)

enable_testing()
add_executable(test_hf_exchange tests/test_hf_exchange.cpp)
add_test(NAME hf_exchange COMMAND test_hf_exchange)
