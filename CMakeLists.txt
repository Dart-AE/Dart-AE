cmake_minimum_required(VERSION 3.18)
project(DART LANGUAGES CXX CUDA VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architecture settings (default: Ampere architecture)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

# DART configuration parameters
if(NOT DEFINED NUM_TESTBENCHES)
    set(NUM_TESTBENCHES 1024)
endif()

if(NOT DEFINED GPU_THREADS)
    set(GPU_THREADS 256)
endif()

# Compilation options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall -Wextra")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2 --extended-lambda -DNUM_TESTBENCHES=${NUM_TESTBENCHES} -DGPU_THREADS=${GPU_THREADS}")

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CUDA_INCLUDE_DIRS})

# Find CUDA
find_package(CUDA REQUIRED)

message(STATUS "")
message(STATUS "════════════════════════════════════════════════")
message(STATUS "  DART Configuration")
message(STATUS "════════════════════════════════════════════════")
message(STATUS "  CUDA Version: ${CUDA_VERSION}")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Install Prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  NUM_TESTBENCHES: ${NUM_TESTBENCHES}")
message(STATUS "  GPU_THREADS: ${GPU_THREADS}")
message(STATUS "════════════════════════════════════════════════")
message(STATUS "")

# DAG analysis module (C++)
set(DAG_ANALYSIS_SOURCES
    src/dag_analyzer.cpp
    src/topological_analyzer.cpp
    src/critical_path.cpp
)

# Fingerprint matching engine module (CUDA)
set(MATCHING_ENGINE_SOURCES
    src/fingerprint.cu
    src/stimulus_merger.cu
    src/hash_table.cu
)

# Execution management module (CUDA)
set(EXECUTION_MGMT_SOURCES
    src/warp_reorganizer.cu
    src/batch_overlapper.cu
    src/state_reconstructor.cu
)

# Runtime management module (C++)
set(RUNTIME_SOURCES
    src/runtime_manager.cpp
    src/utils.cpp
)

# Combine all source files
set(DART_SOURCES
    ${DAG_ANALYSIS_SOURCES}
    ${MATCHING_ENGINE_SOURCES}
    ${EXECUTION_MGMT_SOURCES}
    ${RUNTIME_SOURCES}
)

# Create DART static library
add_library(dart_lib STATIC ${DART_SOURCES})
target_include_directories(dart_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Set CUDA compilation options
set_target_properties(dart_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Link CUDA libraries
target_link_libraries(dart_lib ${CUDA_LIBRARIES})

# Example programs
option(BUILD_EXAMPLES "Build example programs" ON)
if(BUILD_EXAMPLES)
    if(EXISTS ${CMAKE_SOURCE_DIR}/examples/simple_example.cpp)
        add_executable(simple_example examples/simple_example.cpp)
        target_include_directories(simple_example PRIVATE ${CUDA_INCLUDE_DIRS})
        target_link_libraries(simple_example dart_lib ${CUDA_LIBRARIES} cudart)
    endif()
endif()

# Test programs
option(BUILD_TESTS "Build test programs" ON)
if(BUILD_TESTS)
    if(EXISTS ${CMAKE_SOURCE_DIR}/tests/quick_convergence_test.cu)
        add_executable(quick_test tests/quick_convergence_test.cu)
        target_include_directories(quick_test PRIVATE ${CUDA_INCLUDE_DIRS})
        target_link_libraries(quick_test ${CUDA_LIBRARIES} cudart)
        set_target_properties(quick_test PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
        )
    endif()
endif()

# Benchmark programs
option(BUILD_BENCHMARKS "Build benchmark programs" ON)
if(BUILD_BENCHMARKS)
    if(EXISTS ${CMAKE_SOURCE_DIR}/benchmarks/dart_benchmark.cu)
        add_executable(dart_benchmark benchmarks/dart_benchmark.cu)
        target_include_directories(dart_benchmark PRIVATE ${CUDA_INCLUDE_DIRS})
        target_link_libraries(dart_benchmark dart_lib ${CUDA_LIBRARIES} cudart)
        set_target_properties(dart_benchmark PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
        )
    endif()
endif()

# Installation configuration
install(TARGETS dart_lib
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(DIRECTORY include/dart
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

# Print configuration information
message(STATUS "Build Configuration:")
message(STATUS "  Build Examples: ${BUILD_EXAMPLES}")
message(STATUS "  Build Tests: ${BUILD_TESTS}")
message(STATUS "  Build Benchmarks: ${BUILD_BENCHMARKS}")
message(STATUS "")
