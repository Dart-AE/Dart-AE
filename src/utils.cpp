// src/utils.cpp
#include "utils.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <sstream>

namespace dart {
namespace utils {

// Check CUDA error and print message
void CheckCudaError(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error at " << file << ":" << line << std::endl;
        std::cerr << "  " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Get CUDA device properties
void PrintDeviceInfo() {
    int device_count;
    cudaGetDeviceCount(&device_count);

    std::cout << "Found " << device_count << " CUDA device(s)" << std::endl;

    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        std::cout << "\nDevice " << i << ": " << prop.name << std::endl;
        std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Total global memory: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB" << std::endl;
        std::cout << "  Shared memory per block: " << (prop.sharedMemPerBlock / 1024) << " KB" << std::endl;
        std::cout << "  Registers per block: " << prop.regsPerBlock << std::endl;
        std::cout << "  Warp size: " << prop.warpSize << std::endl;
        std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Max threads dim: (" << prop.maxThreadsDim[0] << ", "
                  << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")" << std::endl;
        std::cout << "  Max grid size: (" << prop.maxGridSize[0] << ", "
                  << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")" << std::endl;
        std::cout << "  Clock rate: " << (prop.clockRate / 1000) << " MHz" << std::endl;
        std::cout << "  Memory clock rate: " << (prop.memoryClockRate / 1000) << " MHz" << std::endl;
        std::cout << "  Memory bus width: " << prop.memoryBusWidth << " bits" << std::endl;
        std::cout << "  L2 cache size: " << (prop.l2CacheSize / 1024) << " KB" << std::endl;
        std::cout << "  Number of SMs: " << prop.multiProcessorCount << std::endl;
    }
}

// Format bytes to human-readable string
std::string FormatBytes(size_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit_index = 0;
    double size = static_cast<double>(bytes);

    while (size >= 1024.0 && unit_index < 4) {
        size /= 1024.0;
        unit_index++;
    }

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << size << " " << units[unit_index];
    return oss.str();
}

// Format time to human-readable string
std::string FormatTime(float milliseconds) {
    if (milliseconds < 1.0f) {
        return std::to_string(static_cast<int>(milliseconds * 1000.0f)) + " us";
    } else if (milliseconds < 1000.0f) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << milliseconds << " ms";
        return oss.str();
    } else {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << (milliseconds / 1000.0f) << " s";
        return oss.str();
    }
}

// Print execution result summary
void PrintExecutionResult(const ExecutionResult& result) {
    std::cout << "\n=== DART Execution Result ===" << std::endl;
    std::cout << "Total stimuli: " << result.num_total << std::endl;
    std::cout << "Merged stimuli: " << result.num_merged << std::endl;
    std::cout << "Active stimuli: " << result.num_active << std::endl;
    std::cout << "Merge ratio: " << (result.merge_ratio * 100.0f) << "%" << std::endl;
    std::cout << "Speedup: " << result.speedup << "x" << std::endl;
    std::cout << "\nTiming breakdown:" << std::endl;
    std::cout << "  Fingerprint matching: " << FormatTime(result.fingerprint_time_ms) << std::endl;
    std::cout << "  Warp reorganization: " << FormatTime(result.reorganization_time_ms) << std::endl;
    std::cout << "  State reconstruction: " << FormatTime(result.reconstruction_time_ms) << std::endl;
    std::cout << "  Total DART overhead: " << FormatTime(result.total_time_ms) << std::endl;
    std::cout << "============================\n" << std::endl;
}

// Timer implementation
Timer::Timer() : start_time_(std::chrono::high_resolution_clock::now()) {
}

void Timer::Reset() {
    start_time_ = std::chrono::high_resolution_clock::now();
}

float Timer::ElapsedMilliseconds() const {
    auto end_time = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float, std::milli>(end_time - start_time_).count();
}

float Timer::ElapsedSeconds() const {
    return ElapsedMilliseconds() / 1000.0f;
}

// Memory tracker implementation
MemoryTracker::MemoryTracker() : peak_usage_(0), current_usage_(0) {
}

void MemoryTracker::Allocate(size_t bytes) {
    current_usage_ += bytes;
    peak_usage_ = std::max(peak_usage_, current_usage_);
}

void MemoryTracker::Free(size_t bytes) {
    current_usage_ -= bytes;
}

size_t MemoryTracker::GetCurrentUsage() const {
    return current_usage_;
}

size_t MemoryTracker::GetPeakUsage() const {
    return peak_usage_;
}

void MemoryTracker::Reset() {
    current_usage_ = 0;
    peak_usage_ = 0;
}

void MemoryTracker::PrintSummary() const {
    std::cout << "Memory usage:" << std::endl;
    std::cout << "  Current: " << FormatBytes(current_usage_) << std::endl;
    std::cout << "  Peak: " << FormatBytes(peak_usage_) << std::endl;
}

} // namespace utils
} // namespace dart
