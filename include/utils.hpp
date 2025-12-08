// include/utils.hpp
#pragma once

#include "runtime_manager.hpp"
#include <string>
#include <chrono>
#include <cstdint>
#include <cuda_runtime.h>

namespace dart {
namespace utils {

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            dart::utils::CheckCudaError(error, __FILE__, __LINE__); \
        } \
    } while(0)

// Check CUDA error and print message
void CheckCudaError(cudaError_t error, const char* file, int line);

// Print CUDA device information
void PrintDeviceInfo();

// Format bytes to human-readable string (B, KB, MB, GB)
std::string FormatBytes(size_t bytes);

// Format time to human-readable string (us, ms, s)
std::string FormatTime(float milliseconds);

// Print execution result summary
void PrintExecutionResult(const ExecutionResult& result);

// Simple timer class
class Timer {
public:
    Timer();

    // Reset timer
    void Reset();

    // Get elapsed time in milliseconds
    float ElapsedMilliseconds() const;

    // Get elapsed time in seconds
    float ElapsedSeconds() const;

private:
    std::chrono::high_resolution_clock::time_point start_time_;
};

// Memory usage tracker
class MemoryTracker {
public:
    MemoryTracker();

    // Record memory allocation
    void Allocate(size_t bytes);

    // Record memory deallocation
    void Free(size_t bytes);

    // Get current memory usage
    size_t GetCurrentUsage() const;

    // Get peak memory usage
    size_t GetPeakUsage() const;

    // Reset tracker
    void Reset();

    // Print memory usage summary
    void PrintSummary() const;

private:
    size_t peak_usage_;
    size_t current_usage_;
};

} // namespace utils
} // namespace dart
