# DART: Dynamic Redundancy Tracking for GPU-Accelerated RTL Simulation

DART automatically detects and eliminates redundant execution in parallel testbenches for GPU-accelerated RTL simulation.

## Requirements

- **GPU**: NVIDIA with Compute Capability ≥ 7.5
- **CUDA**: 11.0+ (Recommended: 12.x)
- **CMake**: 3.18+
- **GCC**: 7.5+ (C++17 support)

## Quick Start

### Step 1: Build DART

```bash
cd DART-Release
./scripts/build.sh
```

Build with specific testbench count and thread count:
```bash
./scripts/build.sh --testbenches 4096 --threads 512
```

Options:
- `--testbenches N`: Number of testbenches (default: 1024)
- `--threads N`: GPU threads per block (default: 256)
- `--arch N`: CUDA architecture (default: 86)
- `--clean`: Clean rebuild

### Step 2: Generate Testbench Traces (Optional)

If you need more than 1024 traces:
```bash
./hw_small/verif/scripts/generate_traces.sh 4096 traces/
```

The build system can also auto-generate traces:
```bash
./scripts/build.sh --testbenches 4096
```

### Step 3: Run Experiments

```bash
cd build
./dart_benchmark
```

Or run custom experiment:
```bash
./scripts/run_experiment.sh 1024 10000 100 2
```

Parameters: `[stimuli] [cycles] [detection_interval] [roi_threshold]`

## Typical Workflow

```bash
# Standard 1024 testbenches
./scripts/build.sh
cd build && ./quick_test && ./dart_benchmark

# Large-scale 4096 testbenches
./scripts/build.sh --testbenches 4096 --threads 512
cd build && ./dart_benchmark
```

## Directory Structure

```
DART-Release/
├── hw_small/          # NVDLA Verilog design
│   └── vmod/          # Hardware modules
├── traces/            # Testbench trace files
├── src/dart/          # DART core library
├── include/dart/      # API headers
├── scripts/           # Build and run scripts
└── build/             # Build output (generated)
```

## Troubleshooting

**Build fails with "nvcc not found":**
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

## License

MIT License

