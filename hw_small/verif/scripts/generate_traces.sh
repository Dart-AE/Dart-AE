#!/usr/bin/bash

# DART Trace Generation Script
# Usage: ./generate_traces.sh <NUM_TESTBENCHES> [OUTPUT_DIR]
#
# Generates specified number of testbench traces by randomly combining
# base traces from the traces/ directory

if [ $# -lt 1 ]; then
    echo "Usage: $0 <NUM_TESTBENCHES> [OUTPUT_DIR]"
    echo "Example: $0 4096 ../../../traces"
    exit 1
fi

NUM_TESTBENCHES=$1
OUTPUT_DIR=${2:-"../../../traces"}
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "=== DART Trace Generator ==="
echo "Generating $NUM_TESTBENCHES testbench traces..."
echo "Output directory: $OUTPUT_DIR"
echo ""

# Get list of base trace files
BASE_TRACES_DIR="$OUTPUT_DIR"
if [ ! -d "$BASE_TRACES_DIR" ]; then
    mkdir -p "$BASE_TRACES_DIR"
fi

# Check if we have base traces
base_trace_count=$(ls -1 $BASE_TRACES_DIR/tb*.bin 2>/dev/null | wc -l)
if [ $base_trace_count -eq 0 ]; then
    echo "ERROR: No base trace files (tb*.bin) found in $BASE_TRACES_DIR"
    echo "Please ensure you have base trace files before generating new traces."
    exit 1
fi

echo "Found $base_trace_count base trace files"

# If requested number is less than or equal to existing, just use existing traces
if [ $NUM_TESTBENCHES -le $base_trace_count ]; then
    echo "Requested $NUM_TESTBENCHES traces, already have $base_trace_count base traces."
    echo "Using first $NUM_TESTBENCHES existing traces."
    echo "Trace generation complete!"
    exit 0
fi

echo "Generating $((NUM_TESTBENCHES - base_trace_count)) additional traces..."

# Generate additional traces by randomly duplicating/combining base traces
for ((i=$base_trace_count; i<$NUM_TESTBENCHES; i++)); do
    # Randomly select a base trace to use as template
    base_idx=$((RANDOM % base_trace_count))
    base_trace="$BASE_TRACES_DIR/tb${base_idx}.bin"
    new_trace="$BASE_TRACES_DIR/tb${i}.bin"

    # Copy base trace to create new trace
    cp "$base_trace" "$new_trace"

    # Optionally combine with another trace (30% chance)
    if [ $((RANDOM % 10)) -lt 3 ]; then
        base_idx2=$((RANDOM % base_trace_count))
        base_trace2="$BASE_TRACES_DIR/tb${base_idx2}.bin"
        # Remove EOF marker from first trace (last byte 0xFF)
        head -c -1 "$new_trace" > "$new_trace.tmp"
        # Remove EOF from second trace
        head -c -1 "$base_trace2" > "$base_trace2.tmp"
        # Combine and add new EOF
        cat "$new_trace.tmp" "$base_trace2.tmp" > "$new_trace"
        python3 "$SCRIPT_DIR/add_eof.py" "$new_trace"
        rm "$new_trace.tmp" "$base_trace2.tmp"
    fi

    if [ $((i % 100)) -eq 0 ]; then
        echo "  Generated $i / $NUM_TESTBENCHES traces..."
    fi
done

echo ""
echo "✓ Successfully generated $NUM_TESTBENCHES testbench traces!"
echo "  Base traces: $base_trace_count"
echo "  Generated: $((NUM_TESTBENCHES - base_trace_count))"
echo "  Output directory: $OUTPUT_DIR"
echo ""
