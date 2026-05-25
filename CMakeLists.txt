cmake_minimum_required(VERSION 3.16)

project(KalmanFilters
    VERSION     1.0.0
    DESCRIPTION "KF / EKF / UKF from scratch — lane marker C0 estimation"
    LANGUAGES   C CXX
)

# -----------------------------------------------------------------------
# Global compiler settings
# -----------------------------------------------------------------------
set(CMAKE_C_STANDARD          99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS        OFF)

add_compile_options(-Wall -Wextra -Wpedantic)

# Default build type
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
                 "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

message(STATUS "Build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "C compiler : ${CMAKE_C_COMPILER}")

# -----------------------------------------------------------------------
# Output directories
# -----------------------------------------------------------------------
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# -----------------------------------------------------------------------
# Find Python 3 (needed for dataset generation)
# -----------------------------------------------------------------------
find_package(Python3 REQUIRED COMPONENTS Interpreter)
message(STATUS "Python3    : ${Python3_EXECUTABLE}")

# -----------------------------------------------------------------------
# Dataset CSV
#
# Generated once into <build>/data/lane_data.csv and shared by all
# example targets.  Re-run with:
#
#   cmake --build build --target generate_data
#
# -----------------------------------------------------------------------
set(DATA_DIR        "${CMAKE_BINARY_DIR}/data")
set(DATASET_CSV     "${DATA_DIR}/lane_data.csv")
set(GENERATOR_SCRIPT "${CMAKE_SOURCE_DIR}/python/data/generate_lane_data.py")

add_custom_command(
    OUTPUT  "${DATASET_CSV}"
    COMMAND "${Python3_EXECUTABLE}"
            "${GENERATOR_SCRIPT}"
            "${DATASET_CSV}"
    DEPENDS "${GENERATOR_SCRIPT}"
    COMMENT "Generating lane dataset → ${DATASET_CSV}"
    VERBATIM
)

# Convenience named target so you can do: cmake --build build --target generate_data
add_custom_target(generate_data DEPENDS "${DATASET_CSV}")

# -----------------------------------------------------------------------
# CTest
# -----------------------------------------------------------------------
enable_testing()

# -----------------------------------------------------------------------
# C sub-project
# -----------------------------------------------------------------------
add_subdirectory(C_code)

# -----------------------------------------------------------------------
# C++ / Eigen sub-project
# -----------------------------------------------------------------------
add_subdirectory(C++)
