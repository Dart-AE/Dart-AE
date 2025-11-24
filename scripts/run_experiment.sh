#!/bin/bash

# DART Experiment Runner Script
# Usage: ./run_experiment.sh [stimuli_count] [cycles] [detection_interval] [roi_threshold]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DART_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$DART_ROOT/build"
RESULTS_DIR="$DART_ROOT/results"

# Default parameters
NUM_STIMULI=${1:-1024}
NUM_CYCLES=${2:-10000}
DETECTION_INTERVAL=${3:-100}
ROI_THRESHOLD=${4:-2}

echo "════════════════════════════════════════════════"
echo "DART Experiment Runner"
echo "════════════════════════════════════════════════"
echo "Configuration:"
echo "  Stimuli: $NUM_STIMULI"
echo "  Cycles: $NUM_CYCLES"
echo "  Detection Interval: $DETECTION_INTERVAL"
echo "  ROI Threshold: $ROI_THRESHOLD"
echo "════════════════════════════════════════════════"
echo ""

# Check if already built
if [ ! -f "$BUILD_DIR/dart_benchmark" ]; then
    echo "Error: dart_benchmark not found. Please run './scripts/build.sh' first."
    exit 1
fi

# Create results directory
mkdir -p "$RESULTS_DIR"

# Generate experiment name
EXP_NAME="${NUM_STIMULI}_stimuli_${NUM_CYCLES}_cycles"
RESULT_FILE="$RESULTS_DIR/${EXP_NAME}_$(date +%Y%m%d_%H%M%S).txt"

echo "Running experiment..."
echo "Results will be saved to: $RESULT_FILE"
echo ""

cd "$BUILD_DIR"

# Run experiment (set environment variables to configure parameters)
./dart_benchmark 2>&1 | tee "$RESULT_FILE"

echo ""
echo "════════════════════════════════════════════════"
echo "Experiment Complete!"
echo "Results saved to: $RESULT_FILE"
echo "════════════════════════════════════════════════"
