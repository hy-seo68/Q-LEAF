cmake_minimum_required(VERSION 3.10)
project(Q-LEAF VERSION 1.0.0 LANGUAGES CXX)

# 빌드 타입 기본값 설정
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# C++ 표준 설정
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 컴파일 옵션
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -march=native")

# OpenMP 지원
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# BLAS 라이브러리 찾기 (OpenBLAS 또는 시스템 BLAS)
find_package(BLAS REQUIRED)
if(BLAS_FOUND)
    message(STATUS "BLAS found: ${BLAS_LIBRARIES}")
else()
    message(FATAL_ERROR "BLAS library not found. Please install OpenBLAS or set BLAS_LIBRARIES manually.")
endif()

# Boost 라이브러리 찾기
find_package(Boost REQUIRED)
if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    message(STATUS "Boost found: ${Boost_INCLUDE_DIRS}")
endif()

# 실행 파일 출력 디렉토리 설정
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ============================================================================
# External Library Paths (set via CMake options)
# ============================================================================
# EFANNA2E (NSG) library
if(NOT DEFINED EFANNA2E_DIR)
    set(EFANNA2E_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/nsg" CACHE PATH "Path to EFANNA2E (NSG) library")
endif()
set(EFANNA2E_INCLUDE_DIR "${EFANNA2E_DIR}/include")
set(EFANNA2E_LIB_DIR "${EFANNA2E_DIR}/build/src")

# EFANNA Graph (NN-Descent) library
if(NOT DEFINED EFANNA_GRAPH_DIR)
    set(EFANNA_GRAPH_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/efanna_graph" CACHE PATH "Path to EFANNA Graph library")
endif()
set(EFANNA_GRAPH_INCLUDE_DIR "${EFANNA_GRAPH_DIR}/include")
set(EFANNA_GRAPH_LIB_DIR "${EFANNA_GRAPH_DIR}/build/src")

# FAISS library
if(NOT DEFINED FAISS_DIR)
    set(FAISS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/faiss" CACHE PATH "Path to FAISS library")
endif()
set(FAISS_INCLUDE_DIR "${FAISS_DIR}")
set(FAISS_LIB_DIR "${FAISS_DIR}/build/faiss")

# ============================================================================
# Validate external library paths
# ============================================================================
if(NOT EXISTS "${EFANNA2E_INCLUDE_DIR}/efanna2e/index_nsg.h")
    message(WARNING "EFANNA2E headers not found at ${EFANNA2E_INCLUDE_DIR}. Please set -DEFANNA2E_DIR=<path>")
endif()

if(NOT EXISTS "${EFANNA_GRAPH_INCLUDE_DIR}/efanna2e/index_graph.h")
    message(WARNING "EFANNA Graph headers not found at ${EFANNA_GRAPH_INCLUDE_DIR}. Please set -DEFANNA_GRAPH_DIR=<path>")
endif()

if(NOT EXISTS "${FAISS_INCLUDE_DIR}/faiss/Clustering.h")
    message(WARNING "FAISS headers not found at ${FAISS_INCLUDE_DIR}. Please set -DFAISS_DIR=<path>")
endif()

# ============================================================================
# Include directories
# ============================================================================
# EFANNA Graph headers first (for LockNeighbor type priority)
include_directories(${EFANNA_GRAPH_INCLUDE_DIR})
include_directories(${EFANNA2E_INCLUDE_DIR})
include_directories(${FAISS_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# ============================================================================
# Source files
# ============================================================================
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(BENCH_DIR ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks)

# Common utility library
add_library(qleaf_utils STATIC
    ${SRC_DIR}/nsg_qleaf_utils.cpp
)
target_link_libraries(qleaf_utils
    ${FAISS_LIB_DIR}/libfaiss.a
    ${BLAS_LIBRARIES}
)

# Data loader library
add_library(qleaf_data_loader STATIC
    ${SRC_DIR}/nsg_qleaf_data_loader.cpp
)
target_link_libraries(qleaf_data_loader
    qleaf_utils
    ${EFANNA2E_LIB_DIR}/libefanna2e.a
)

# ============================================================================
# Executables
# ============================================================================

# NSG Subgraph Builder (preprocessing)
add_executable(build_nsg_subgraphs
    ${SRC_DIR}/build_nsg_subgraphs.cpp
)
target_link_libraries(build_nsg_subgraphs
    qleaf_data_loader
    ${EFANNA_GRAPH_LIB_DIR}/libefanna2e.a
    ${EFANNA2E_LIB_DIR}/libefanna2e.a
    ${BLAS_LIBRARIES}
)

# Entry Point Computation (Q-LEAF algorithm)
add_executable(compute_ep_nsg
    ${SRC_DIR}/compute_ep_nsg.cpp
)
target_link_libraries(compute_ep_nsg
    qleaf_utils
    qleaf_data_loader
    ${EFANNA2E_LIB_DIR}/libefanna2e.a
    ${FAISS_LIB_DIR}/libfaiss.a
    ${BLAS_LIBRARIES}
)

# Q-LEAF Single EP Benchmark (Case C: Q-LEAF Full)
add_executable(nsg_qleaf_single_ep
    ${SRC_DIR}/nsg_qleaf_single_ep.cpp
)
target_link_libraries(nsg_qleaf_single_ep
    qleaf_utils
    qleaf_data_loader
    ${EFANNA2E_LIB_DIR}/libefanna2e.a
    ${FAISS_LIB_DIR}/libfaiss.a
    ${BLAS_LIBRARIES}
)

# Baseline: Subgraph Only (Case B)
add_executable(nsg_baseline_subgraph_benchmark
    ${BENCH_DIR}/nsg_baseline_subgraph_benchmark.cpp
)
target_link_libraries(nsg_baseline_subgraph_benchmark
    qleaf_utils
    qleaf_data_loader
    ${EFANNA2E_LIB_DIR}/libefanna2e.a
    ${FAISS_LIB_DIR}/libfaiss.a
    ${BLAS_LIBRARIES}
)

# Baseline: Full Graph (Case A)
add_executable(nsg_baseline_full_graph
    ${BENCH_DIR}/nsg_baseline_full_graph.cpp
)
target_link_libraries(nsg_baseline_full_graph
    ${EFANNA2E_LIB_DIR}/libefanna2e.a
)

# ============================================================================
# Install rules
# ============================================================================
install(TARGETS
    build_nsg_subgraphs
    compute_ep_nsg
    nsg_qleaf_single_ep
    nsg_baseline_subgraph_benchmark
    nsg_baseline_full_graph
    RUNTIME DESTINATION bin
)

# ============================================================================
# Build information
# ============================================================================
message(STATUS "")
message(STATUS "Q-LEAF Build Configuration:")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "")
message(STATUS "External Libraries:")
message(STATUS "  EFANNA2E (NSG): ${EFANNA2E_DIR}")
message(STATUS "  EFANNA Graph:   ${EFANNA_GRAPH_DIR}")
message(STATUS "  FAISS:          ${FAISS_DIR}")
message(STATUS "  BLAS:           ${BLAS_LIBRARIES}")
message(STATUS "")
message(STATUS "Executables:")
message(STATUS "  - build_nsg_subgraphs             (Preprocessing: Build subgraphs)")
message(STATUS "  - compute_ep_nsg                  (Preprocessing: Compute entry points)")
message(STATUS "  - nsg_qleaf_single_ep             (Benchmark: Case C - Q-LEAF Full)")
message(STATUS "  - nsg_baseline_subgraph_benchmark (Benchmark: Case B - Subgraph only)")
message(STATUS "  - nsg_baseline_full_graph         (Benchmark: Case A - Full graph)")
message(STATUS "")
