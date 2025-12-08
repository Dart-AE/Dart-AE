// src/batch_overlapper.cu
#include "batch_overlapper.hpp"
#include <cuda_runtime.h>
#include <iostream>

namespace dart {

BatchOverlapper::BatchOverlapper(uint32_t max_concurrent_batches)
    : max_concurrent_batches_(max_concurrent_batches)
    , num_active_batches_(0)
{
    // Create CUDA streams for overlapping execution
    streams_.resize(max_concurrent_batches_);
    events_.resize(max_concurrent_batches_);

    for (uint32_t i = 0; i < max_concurrent_batches_; ++i) {
        cudaStreamCreate(&streams_[i]);
        cudaEventCreate(&events_[i]);
    }
}

BatchOverlapper::~BatchOverlapper() {
    // Wait for all batches to complete
    WaitForAllBatches();

    // Destroy streams and events
    for (uint32_t i = 0; i < max_concurrent_batches_; ++i) {
        cudaStreamDestroy(streams_[i]);
        cudaEventDestroy(events_[i]);
    }
}

uint32_t BatchOverlapper::AllocateBatch() {
    // Find available stream
    while (num_active_batches_ >= max_concurrent_batches_) {
        // Wait for one batch to complete
        CheckBatchCompletion();
    }

    // Allocate next available batch ID
    uint32_t batch_id = num_active_batches_;
    num_active_batches_++;

    active_batch_ids_.insert(batch_id);

    return batch_id;
}

void BatchOverlapper::LaunchBatchKernel(
    uint32_t batch_id,
    void (*kernel_func)(cudaStream_t),
    bool record_event
) {
    if (batch_id >= max_concurrent_batches_) {
        throw std::runtime_error("Invalid batch ID");
    }

    // Launch kernel on the batch's stream
    cudaStream_t stream = streams_[batch_id];
    kernel_func(stream);

    // Record completion event if requested
    if (record_event) {
        cudaEventRecord(events_[batch_id], stream);
    }
}

void BatchOverlapper::WaitForBatch(uint32_t batch_id) {
    if (batch_id >= max_concurrent_batches_) {
        return;
    }

    // Synchronize on the batch's stream
    cudaStreamSynchronize(streams_[batch_id]);

    // Mark batch as completed
    if (active_batch_ids_.count(batch_id) > 0) {
        active_batch_ids_.erase(batch_id);
        num_active_batches_--;
    }
}

void BatchOverlapper::WaitForAllBatches() {
    for (uint32_t i = 0; i < max_concurrent_batches_; ++i) {
        cudaStreamSynchronize(streams_[i]);
    }

    active_batch_ids_.clear();
    num_active_batches_ = 0;
}

bool BatchOverlapper::IsBatchComplete(uint32_t batch_id) {
    if (batch_id >= max_concurrent_batches_) {
        return true;
    }

    cudaError_t result = cudaEventQuery(events_[batch_id]);
    return (result == cudaSuccess);
}

void BatchOverlapper::CheckBatchCompletion() {
    // Check all active batches
    std::vector<uint32_t> completed_batches;

    for (uint32_t batch_id : active_batch_ids_) {
        if (IsBatchComplete(batch_id)) {
            completed_batches.push_back(batch_id);
        }
    }

    // Remove completed batches
    for (uint32_t batch_id : completed_batches) {
        active_batch_ids_.erase(batch_id);
        num_active_batches_--;
    }

    // If no batches completed, wait for the earliest one
    if (completed_batches.empty() && !active_batch_ids_.empty()) {
        uint32_t earliest_batch = *active_batch_ids_.begin();
        WaitForBatch(earliest_batch);
    }
}

cudaStream_t BatchOverlapper::GetStream(uint32_t batch_id) {
    if (batch_id >= max_concurrent_batches_) {
        return nullptr;
    }
    return streams_[batch_id];
}

uint32_t BatchOverlapper::GetNumActiveBatches() const {
    return num_active_batches_;
}

// Pipeline manager implementation
PipelineManager::PipelineManager(uint32_t num_stages)
    : num_stages_(num_stages)
    , current_stage_(0)
{
    // Create stream for each pipeline stage
    stage_streams_.resize(num_stages_);
    stage_events_.resize(num_stages_);

    for (uint32_t i = 0; i < num_stages_; ++i) {
        cudaStreamCreate(&stage_streams_[i]);
        cudaEventCreate(&stage_events_[i]);
    }
}

PipelineManager::~PipelineManager() {
    for (uint32_t i = 0; i < num_stages_; ++i) {
        cudaStreamDestroy(stage_streams_[i]);
        cudaEventDestroy(stage_events_[i]);
    }
}

void PipelineManager::ExecuteStage(
    uint32_t stage_id,
    void (*stage_func)(cudaStream_t)
) {
    if (stage_id >= num_stages_) {
        throw std::runtime_error("Invalid stage ID");
    }

    cudaStream_t stream = stage_streams_[stage_id];

    // If not the first stage, wait for previous stage
    if (stage_id > 0) {
        cudaStreamWaitEvent(stream, stage_events_[stage_id - 1], 0);
    }

    // Execute stage
    stage_func(stream);

    // Record completion
    cudaEventRecord(stage_events_[stage_id], stream);

    current_stage_ = (stage_id + 1) % num_stages_;
}

void PipelineManager::WaitForPipeline() {
    for (uint32_t i = 0; i < num_stages_; ++i) {
        cudaStreamSynchronize(stage_streams_[i]);
    }
}

cudaStream_t PipelineManager::GetStageStream(uint32_t stage_id) {
    if (stage_id >= num_stages_) {
        return nullptr;
    }
    return stage_streams_[stage_id];
}

} // namespace dart
