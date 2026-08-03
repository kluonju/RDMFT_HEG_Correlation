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

add_executable(rdmft_heg
    src/main.cpp
    src/MomentumDistributionGZ.cpp
    src/NNFunctional.cpp
)

enable_testing()
add_executable(test_hf_exchange tests/test_hf_exchange.cpp)
add_test(NAME hf_exchange COMMAND test_hf_exchange)

add_executable(test_gz_momentum
    tests/test_gz_momentum.cpp
    src/MomentumDistributionGZ.cpp
)
add_test(NAME gz_momentum COMMAND test_gz_momentum)

add_executable(test_nn_functional
    tests/test_nn_functional.cpp
    src/NNFunctional.cpp
)
add_test(NAME nn_functional COMMAND test_nn_functional)
set_tests_properties(nn_functional PROPERTIES
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)

add_executable(test_solver_robustness tests/test_solver_robustness.cpp)
add_test(NAME solver_robustness COMMAND test_solver_robustness)
set_tests_properties(solver_robustness PROPERTIES
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)

add_executable(test_boundaries tests/test_boundaries.cpp)
add_test(NAME boundaries COMMAND test_boundaries)

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    add_test(
        NAME cli_validation
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tests/test_cli.py
                $<TARGET_FILE:rdmft_heg>
    )
endif()

# Helper binary used by scripts/plot_gz.py to render the
# Gori-Giorgi/Ziesche n(k, rs) parametrization (paper Fig. 6).
add_executable(dump_gz_grid
    tools/dump_gz_grid.cpp
    src/MomentumDistributionGZ.cpp
)
