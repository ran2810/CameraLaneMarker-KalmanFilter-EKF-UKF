# cpp/CMakeLists.txt
# -----------------------------------------------------------------------
# Builds the C++ / Eigen Kalman filter library and examples.
#
# All filters are header-only (KalmanFilter.hpp, EKFFilter.hpp,
# UKFFilter.hpp) assembled through FilterBase.hpp (CRTP).
# This CMakeLists therefore only needs to:
#   1. Create an INTERFACE library that carries the include path + Eigen
#   2. Build the test executable
#   3. Build the compare_filters example
#
# Targets
# -------
#   kalman_cpp        INTERFACE library (include path + Eigen3 + C++17)
#   test_filters_cpp  unit tests  (registered with CTest)
#   compare_filters_cpp  three-filter comparison example
#
# Run targets
# -----------
#   run_compare_cpp   generate_data → compare_filters_cpp
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# 1. Eigen3
# -----------------------------------------------------------------------
# Add Eigen manually
set(EIGEN_ROOT "C:/tools/eigen-3.4.1")

#find_package(Eigen3 3.3 REQUIRED NO_MODULE)
#message(STATUS "Eigen3     : ${EIGEN3_VERSION_STRING} (${EIGEN3_INCLUDE_DIR})")

# -----------------------------------------------------------------------
# 2. Interface library — header-only, carries compile requirements
# -----------------------------------------------------------------------
add_library(kalman_cpp INTERFACE)

target_include_directories(kalman_cpp
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_include_directories(kalman_cpp INTERFACE
    ${EIGEN_ROOT})

#target_link_libraries(kalman_cpp INTERFACE Eigen3::Eigen)

target_compile_features(kalman_cpp INTERFACE cxx_std_17)_cpp

# Eigen performs a lot of template meta-programming; -O2 is strongly
# recommended even for Debug builds to keep compile times sane.
target_compile_options(kalman_cpp INTERFACE
    $<$<CONFIG:Debug>:-O1>
)

# Disable Eigen's own asserts in Release (speeds up inner loops)
target_compile_definitions(kalman_cpp INTERFACE
    $<$<CONFIG:Release>:EIGEN_NO_DEBUG>
)

# -----------------------------------------------------------------------
# 3. Unit tests
# -----------------------------------------------------------------------
add_executable(test_filters_cpp tests/test_filters.cpp)
target_link_libraries(test_filters_cpp PRIVATE kalman_cpp)

add_test(NAME test_filters_cpp COMMAND test_filters_cpp)
set_tests_properties(test_filters_cpp PROPERTIES
    FAIL_REGULAR_EXPRESSION "FAIL[^E]|FAILED"
    PASS_REGULAR_EXPRESSION "ALL.*PASSED"
)

# -----------------------------------------------------------------------
# 4. Compare-filters example
# -----------------------------------------------------------------------
add_executable(compare_filters_cpp examples/compare_filters.cpp)
target_link_libraries(compare_filters_cpp PRIVATE kalman_cpp)

# -----------------------------------------------------------------------
# 5. Run target (depends on dataset from root CMakeLists)
# -----------------------------------------------------------------------
set(OUTPUT_CPP_CSV "${DATA_DIR}/compare_cpp_output.csv")

add_custom_target(run_compare_cpp
    COMMAND "$<TARGET_FILE:compare_filters_cpp>"
            "${DATASET_CSV}"
            "${OUTPUT_CPP_CSV}"
    DEPENDS compare_filters_cpp generate_data
    COMMENT "C++ compare_filters → ${OUTPUT_CPP_CSV}"
    VERBATIM
)
