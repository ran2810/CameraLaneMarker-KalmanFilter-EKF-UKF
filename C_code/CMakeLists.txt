# c/CMakeLists.txt
# -----------------------------------------------------------------------
# Targets
# -----------------------------------------------------------------------
#   kalman_core     static library  (matrix + kf + ekf + ukf)
#   test_matrix     unit tests — matrix.c
#   test_ekf        unit tests — ekf.c
#   test_ukf        unit tests — ukf.c
#   lane_kf         example: single KF on left/right markers
#   compare_filters example: KF / EKF / UKF side-by-side
#
# Custom targets (require Python / dataset)
#   generate_data   generate data/lane_data.csv from Python generator
#   run_lane_kf     generate_data → lane_kf
#   run_compare     generate_data → compare_filters  (writes compare_output.csv)
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# 1. Static library: kalman_core
# -----------------------------------------------------------------------
add_library(kalman_core STATIC
    src/matrix.c
    src/kf.c
    src/ekf.c
    src/ukf.c
)

target_include_directories(kalman_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# sin / cos / sqrt live in libm on Linux/macOS
target_link_libraries(kalman_core PUBLIC m)

# -----------------------------------------------------------------------
# 2. Unit tests  (registered with CTest)
# -----------------------------------------------------------------------

# Helper macro — avoids repeating the same boilerplate three times.
macro(add_kalman_test)
    cmake_parse_arguments(_T "" "NAME" "SOURCES" ${ARGN})

    add_executable(${_T_NAME} ${_T_SOURCES})
    target_link_libraries(${_T_NAME} PRIVATE kalman_core)

    add_test(NAME ${_T_NAME} COMMAND ${_T_NAME})

    set_tests_properties(${_T_NAME} PROPERTIES
        FAIL_REGULAR_EXPRESSION "FAIL[^E]|FAILED"
        PASS_REGULAR_EXPRESSION "ALL.*PASSED"
    )
endmacro()

add_kalman_test(NAME test_matrix SOURCES tests/test_matrix.c)
add_kalman_test(NAME test_ekf    SOURCES tests/test_ekf.c)
add_kalman_test(NAME test_ukf    SOURCES tests/test_ukf.c)

# -----------------------------------------------------------------------
# 3. Example executables
# -----------------------------------------------------------------------
add_executable(lane_kf         examples/lane_estimation_kf.c)
add_executable(compare_filters examples/compare_filters.c)

target_link_libraries(lane_kf         PRIVATE kalman_core)
target_link_libraries(compare_filters PRIVATE kalman_core)

# -----------------------------------------------------------------------
# 4. Run targets  (depend on the dataset custom command in root CMakeLists)
#
#   cmake --build build --target run_lane_kf
#   cmake --build build --target run_compare
# -----------------------------------------------------------------------

# DATASET_CSV / DATA_DIR defined in parent (root CMakeLists.txt)
set(OUTPUT_CSV "${DATA_DIR}/compare_output.csv")

add_custom_target(run_lane_kf
    COMMAND "$<TARGET_FILE:lane_kf>"
            "${DATASET_CSV}"
            "${DATA_DIR}/lane_kf_output.csv"
    DEPENDS lane_kf generate_data
    COMMENT "lane_kf → ${DATA_DIR}/lane_kf_output.csv"
    VERBATIM
)

add_custom_target(run_compare
    COMMAND "$<TARGET_FILE:compare_filters>"
            "${DATASET_CSV}"
            "${OUTPUT_CSV}"
    DEPENDS compare_filters generate_data
    COMMENT "compare_filters → ${OUTPUT_CSV}"
    VERBATIM
)
