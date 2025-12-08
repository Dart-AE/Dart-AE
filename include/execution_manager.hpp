#ifndef DART_EXECUTION_MANAGER_HPP
#define DART_EXECUTION_MANAGER_HPP

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <queue>
#include <cuda_runtime.h>
#include "dart/dart_config.hpp"

namespace dart {

// RuntimePathSignature is already defined in dart_config.hpp, no need to repeat

// PathSig sorter - stimulus grouping based on LSH
class PathSignatureSorter {
public:
    PathSignatureSorter() = default;
    ~PathSignatureSorter() = default;

    // GPU sorting using thrust
    void SortByPathSignature(
        std::vector<RuntimePathSignature>& signatures,
        cudaStream_t stream
    );

    // Remap warps based on sorting results
    void RemapWarps(
        const std::vector<RuntimePathSignature>& sorted_sigs,
        uint32_t warp_size,
        std::vector<uint32_t>& new_warp_assignment
    );
};

// Warp reorganizer - core execution management module
class WarpReorganizer {
private:
    PathSignatureSorter sorter_;
    PathSignatureFeatures* d_ps_dataset_;  // PS dataset on GPU
    uint32_t* d_branch_history_;                  // Branch history on GPU
    size_t max_stimuli_;
    bool initialized_;

public:
    WarpReorganizer();
    ~WarpReorganizer();

    // Initialize: transfer PS dataset to GPU
    void Initialize(
        const std::unordered_map<uint32_t, PathSignatureFeatures>& ps_dataset,
        size_t max_stimuli = 65536
    );

    // Execute reorganization
    std::vector<uint32_t> Reorganize(
        const std::vector<uint32_t>& representative_ids,
        uint32_t detection_node_id,
        cudaStream_t stream
    );

    // Get branch history buffer (for simulation kernel use)
    uint32_t* GetBranchHistoryBuffer() { return d_branch_history_; }

    // Clear branch history (at new batch start)
    void ClearBranchHistory();

    // Check if initialized
    bool IsInitialized() const { return initialized_; }
};

// Batch information structure
struct BatchInfo {
    uint32_t batch_id;
    uint32_t current_cycle;
    uint32_t num_active_stimuli;
    cudaStream_t stream;
    cudaEvent_t completion_event;
};

// Multi-batch overlapping execution manager
class BatchOverlapManager {
private:
    std::vector<BatchInfo> active_batches_;
    std::queue<uint32_t> free_sm_ids_;
    uint32_t max_concurrent_batches_;

public:
    explicit BatchOverlapManager(uint32_t max_concurrent_batches = 4);
    ~BatchOverlapManager();

    // Launch new batch
    void LaunchBatch(
        uint32_t batch_id,
        uint32_t num_stimuli,
        void (*kernel_launcher)(uint32_t, cudaStream_t)
    );

    // Wait for a batch to complete
    void WaitForBatchCompletion();

    // Wait for all batches to complete
    void WaitForAllBatches();

    // Get current active batch count
    size_t GetActiveBatchCount() const { return active_batches_.size(); }
};

// ROI calculator
class ROICalculator {
public:
    // Calculate return on investment for reorganization
    static float ComputeROI(
        const AccumulatedMergeState& merges,
        uint32_t current_active,
        uint32_t remaining_cycles,
        float reorganization_overhead_cycles
    );

    // Estimate saved computations
    static float EstimateSavedComputations(
        const AccumulatedMergeState& merges,
        uint32_t remaining_cycles
    );

    // Estimate reorganization overhead
    static float EstimateReorganizationCost(
        uint32_t current_active,
        float overhead_per_stimulus
    );
};

// GPU Kernels declarations

// Compute PathSig signatures
__global__ void ComputePathSignaturesKernel(
    const uint32_t* stimulus_ids,
    uint32_t num_stimuli,
    const PathSignatureFeatures* ps_dataset,
    const uint32_t* branch_history,
    uint32_t detection_node_id,
    RuntimePathSignature* output_sigs
);

// Record branch history
__device__ void RecordBranchHistory(
    uint32_t stim_id,
    uint32_t node_id,
    bool branch_taken,
    uint32_t* branch_history
);

// Compact representative stimuli
__global__ void CompactRepresentativesKernel(
    const uint32_t* match_table,
    const uint64_t* path_signatures,
    uint32_t num_stimuli,
    uint32_t* compacted_indices,
    uint32_t* compacted_count
);

// Apply warp assignment
__global__ void ApplyWarpAssignmentKernel(
    const uint32_t* old_indices,
    const uint32_t* new_assignment,
    uint32_t num_stimuli,
    void* state_data,
    size_t state_size,
    void* output_state_data
);

// Reconstruct follower states
__global__ void ReconstructFollowerStatesKernel(
    const uint32_t* representative_map,
    uint32_t num_stimuli,
    const void* representative_states,
    void* all_states,
    size_t state_size
);

} // namespace dart

#endif // DART_EXECUTION_MANAGER_HPP
