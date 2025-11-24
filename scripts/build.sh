#!/bin/bash

# DART Build Script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DART_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$DART_ROOT/build"

# Parse arguments
CLEAN_BUILD=0
BUILD_TYPE="Release"
CUDA_ARCH="86"
NUM_JOBS=$(nproc)
NUM_TESTBENCHES=1024
GPU_THREADS=256
GENERATE_TRACES=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN_BUILD=1
            shift
            ;;
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --arch)
            CUDA_ARCH="$2"
            shift 2
            ;;
        --jobs|-j)
            NUM_JOBS="$2"
            shift 2
            ;;
        --testbenches|-t)
            NUM_TESTBENCHES="$2"
            GENERATE_TRACES=1
            shift 2
            ;;
        --threads)
            GPU_THREADS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --clean              Clean build from scratch"
            echo "  --debug              Debug build (default: Release)"
            echo "  --arch ARCH          CUDA architecture (default: 86)"
            echo "  --jobs N, -j N       Number of parallel jobs (default: nproc)"
            echo "  --testbenches N, -t N  Number of testbenches (default: 1024)"
            echo "  --threads N          GPU threads per block (default: 256)"
            echo ""
            echo "Examples:"
            echo "  $0 --testbenches 4096 --threads 512"
            echo "  $0 --arch 80 --testbenches 2048"
            exit 1
            ;;
    esac
done

echo "════════════════════════════════════════════════"
echo "DART Build Script"
echo "════════════════════════════════════════════════"
echo "Configuration:"
echo "  Build Type: $BUILD_TYPE"
echo "  CUDA Architecture: $CUDA_ARCH"
echo "  Parallel Jobs: $NUM_JOBS"
echo "  Clean Build: $CLEAN_BUILD"
echo "  Testbenches: $NUM_TESTBENCHES"
echo "  GPU Threads: $GPU_THREADS"
echo "════════════════════════════════════════════════"
echo ""

# Generate traces (if needed)
if [ $GENERATE_TRACES -eq 1 ]; then
    echo "Generating $NUM_TESTBENCHES testbench traces..."
    "$DART_ROOT/hw_small/verif/scripts/generate_traces.sh" $NUM_TESTBENCHES "$DART_ROOT/traces"
    echo ""
fi

# Clean old build (if needed)
if [ $CLEAN_BUILD -eq 1 ]; then
    echo "Cleaning old build..."
    rm -rf "$BUILD_DIR"
fi

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure CMake
echo "Configuring CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH \
    -DNUM_TESTBENCHES=$NUM_TESTBENCHES \
    -DGPU_THREADS=$GPU_THREADS \
    -DBUILD_EXAMPLES=ON \
    -DBUILD_TESTS=ON \
    -DBUILD_BENCHMARKS=ON

# Build
echo ""
echo "Building DART..."
make -j$NUM_JOBS

echo ""
echo "════════════════════════════════════════════════"
echo "Build Complete!"
echo "════════════════════════════════════════════════"
echo "Built executables:"
[ -f dart_benchmark ] && echo "  ✓ dart_benchmark"
[ -f quick_test ] && echo "  ✓ quick_test"
[ -f simple_example ] && echo "  ✓ simple_example"
echo ""
echo "To run a benchmark:"
echo "  cd build && ./dart_benchmark"
echo ""
echo "Or use the experiment script:"
echo "  ./scripts/run_experiment.sh [stimuli] [cycles]"
echo "════════════════════════════════════════════════"
