// include/batch_overlapper.hpp
#pragma once

#include <vector>
#include <unordered_set>
#include <cstdint>
#include <cuda_runtime.h>

namespace dart {

// Batch overlapping execution manager
class BatchOverlapper {
public:
    explicit BatchOverlapper(uint32_t max_concurrent_batches = 4);
    ~BatchOverlapper();

    // Allocate a batch ID for execution
    uint32_t AllocateBatch();

    // Launch a kernel on a specific batch
    void LaunchBatchKernel(
        uint32_t batch_id,
        void (*kernel_func)(cudaStream_t),
        bool record_event = true
    );

    // Wait for a specific batch to complete
    void WaitForBatch(uint32_t batch_id);

    // Wait for all batches to complete
    void WaitForAllBatches();

    // Check if a batch is complete
    bool IsBatchComplete(uint32_t batch_id);

    // Get CUDA stream for a batch
    cudaStream_t GetStream(uint32_t batch_id);

    // Get number of active batches
    uint32_t GetNumActiveBatches() const;

private:
    void CheckBatchCompletion();

    uint32_t max_concurrent_batches_;
    uint32_t num_active_batches_;
    std::vector<cudaStream_t> streams_;
    std::vector<cudaEvent_t> events_;
    std::unordered_set<uint32_t> active_batch_ids_;
};

// Pipeline execution manager
class PipelineManager {
public:
    explicit PipelineManager(uint32_t num_stages);
    ~PipelineManager();

    // Execute a pipeline stage
    void ExecuteStage(
        uint32_t stage_id,
        void (*stage_func)(cudaStream_t)
    );

    // Wait for entire pipeline to complete
    void WaitForPipeline();

    // Get stream for a specific stage
    cudaStream_t GetStageStream(uint32_t stage_id);

private:
    uint32_t num_stages_;
    uint32_t current_stage_;
    std::vector<cudaStream_t> stage_streams_;
    std::vector<cudaEvent_t> stage_events_;
};

} // namespace dart
