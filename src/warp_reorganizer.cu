#include "execution_manager.hpp"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include <cstring>
#include <queue>

namespace dart {

// ============================================================================
// GPU Kernel: Compute PathSig (Equation 4)
// ============================================================================

__global__ void ComputePathSignaturesKernel(
    const uint32_t* stimulus_ids,           // Input: representative stimulus ID list
    uint32_t num_stimuli,
    const PathSignatureFeatures* ps_dataset, // Precompiled PS dataset
    const uint32_t* branch_history,         // H(si): branch history for each stimulus
    uint32_t detection_node_id,
    RuntimePathSignature* output_sigs       // Output: computed signatures
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_stimuli) return;

    uint32_t stim_id = stimulus_ids[idx];
    const PathSignatureFeatures& ps = ps_dataset[detection_node_id];

    // Implement PathSig(v, si) = Hash(P(v), L(v), H(si))
    uint64_t hash = 0x123456789ABCDEF0ULL;

    // 1. Hash P(v): predecessor node signature set
    for (int i = 0; i < ps.num_predecessors; ++i) {
        uint64_t pred_sig = ps.predecessor_signatures[i];
        hash ^= pred_sig;
        hash *= 0xc6a4a7935bd1e995ULL;  // MurmurHash style
        hash ^= hash >> 47;
    }

    // 2. Hash L(v): topological level
    uint64_t level = ps.topological_level;
    hash ^= level;
    hash *= 0x87c37b91114253d5ULL;
    hash ^= hash >> 33;

    // 3. Hash H(si): control flow branch history of stimulus
    // Branch history encoding: records stimulus choices at critical branch points
    uint32_t branch_bits = branch_history[stim_id];
    hash ^= branch_bits;
    hash *= 0xff51afd7ed558ccdULL;
    hash ^= hash >> 33;

    // Output computed results
    output_sigs[idx].signature = hash;
    output_sigs[idx].stimulus_id = stim_id;
    output_sigs[idx].detection_node = detection_node_id;
}

// ============================================================================
// Device Function: Branch history tracking
// ============================================================================

__device__ void RecordBranchHistory(
    uint32_t stim_id,
    uint32_t node_id,
    bool branch_taken,
    uint32_t* branch_history
) {
    // Use bit vector to record branch choices
    // Each control flow node occupies 1 bit, recording true/false
    uint32_t bit_pos = node_id % 32;

    if (branch_taken) {
        atomicOr(&branch_history[stim_id], 1U << bit_pos);
    }
    // For false case, keep 0, no operation needed
}

// ============================================================================
// GPU Kernel: Warp reorganization - Compact representative stimuli
// ============================================================================

__global__ void CompactRepresentativesKernel(
    const uint32_t* match_table,
    const uint64_t* path_signatures,
    uint32_t num_stimuli,
    uint32_t* compacted_indices,     // Output: compacted index mapping
    uint32_t* compacted_count        // Output: compacted count
) {
    __shared__ uint32_t warp_offset;

    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t lane_id = threadIdx.x % 32;

    // Check if current stimulus is a representative
    bool is_representative = (tid < num_stimuli) && (match_table[tid] == tid);

    // Warp-level scan and compaction
    uint32_t warp_mask = __ballot_sync(0xFFFFFFFF, is_representative);
    uint32_t prefix = __popc(warp_mask & ((1U << lane_id) - 1));

    if (lane_id == 0) {
        warp_offset = atomicAdd(compacted_count, __popc(warp_mask));
    }
    warp_offset = __shfl_sync(0xFFFFFFFF, warp_offset, 0);

    if (is_representative) {
        uint32_t new_idx = warp_offset + prefix;
        compacted_indices[tid] = new_idx;
    }
}

// ============================================================================
// GPU Kernel: Apply warp assignment
// ============================================================================

__global__ void ApplyWarpAssignmentKernel(
    const uint32_t* old_indices,
    const uint32_t* new_assignment,
    uint32_t num_stimuli,
    void* state_data,
    size_t state_size,
    void* output_state_data
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_stimuli) return;

    uint32_t old_pos = old_indices[idx];
    uint32_t new_pos = new_assignment[idx];

    // Copy state data to new position
    char* src = ((char*)state_data) + old_pos * state_size;
    char* dst = ((char*)output_state_data) + new_pos * state_size;

    for (size_t i = 0; i < state_size; ++i) {
        dst[i] = src[i];
    }
}

// ============================================================================
// GPU Kernel: Reconstruct follower states
// ============================================================================

__global__ void ReconstructFollowerStatesKernel(
    const uint32_t* representative_map,
    uint32_t num_stimuli,
    const void* representative_states,
    void* all_states,
    size_t state_size
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_stimuli) return;

    // Find the representative for this stimulus
    uint32_t rep_id = representative_map[idx];

    // Copy state from representative
    const char* src = ((const char*)representative_states) + rep_id * state_size;
    char* dst = ((char*)all_states) + idx * state_size;

    for (size_t i = 0; i < state_size; ++i) {
        dst[i] = src[i];
    }
}

// ============================================================================
// PathSignatureSorter Implementation
// ============================================================================

void PathSignatureSorter::SortByPathSignature(
    std::vector<RuntimePathSignature>& signatures,
    cudaStream_t stream
) {
    if (signatures.empty()) return;

    // 1. Transfer data to GPU
    thrust::device_vector<RuntimePathSignature> d_sigs = signatures;

    // 2. Sort by signature field (LSH ensures similar paths have similar hash values)
    thrust::sort(
        thrust::cuda::par.on(stream),
        d_sigs.begin(), d_sigs.end(),
        [] __device__ (const RuntimePathSignature& a,
                      const RuntimePathSignature& b) {
            return a.signature < b.signature;
        }
    );

    // 3. Copy back to host
    thrust::copy(d_sigs.begin(), d_sigs.end(), signatures.begin());
}

void PathSignatureSorter::RemapWarps(
    const std::vector<RuntimePathSignature>& sorted_sigs,
    uint32_t warp_size,
    std::vector<uint32_t>& new_warp_assignment
) {
    // Similar PathSigs will be grouped together
    // Directly assign to warps in order, naturally achieving memory access locality
    for (size_t i = 0; i < sorted_sigs.size(); ++i) {
        uint32_t stim_id = sorted_sigs[i].stimulus_id;
        uint32_t warp_id = i / warp_size;
        uint32_t lane_id = i % warp_size;

        new_warp_assignment[stim_id] = warp_id * warp_size + lane_id;
    }
}

// ============================================================================
// WarpReorganizer Implementation
// ============================================================================

WarpReorganizer::WarpReorganizer()
    : d_ps_dataset_(nullptr)
    , d_branch_history_(nullptr)
    , max_stimuli_(0)
    , initialized_(false)
{
}

WarpReorganizer::~WarpReorganizer() {
    if (d_ps_dataset_) {
        cudaFree(d_ps_dataset_);
    }
    if (d_branch_history_) {
        cudaFree(d_branch_history_);
    }
}

void WarpReorganizer::Initialize(
    const std::unordered_map<uint32_t, PathSignatureFeatures>& ps_dataset,
    size_t max_stimuli
) {
    max_stimuli_ = max_stimuli;

    // 1. Allocate and transfer PS dataset to GPU
    if (!ps_dataset.empty()) {
        size_t dataset_size = ps_dataset.size() * sizeof(PathSignatureFeatures);
        cudaMalloc(&d_ps_dataset_, dataset_size);

        // Convert to array format and copy
        std::vector<PathSignatureFeatures> ps_array;
        ps_array.reserve(ps_dataset.size());

        // Sort by node_id for indexed access
        for (const auto& [node_id, features] : ps_dataset) {
            ps_array.push_back(features);
        }

        cudaMemcpy(d_ps_dataset_, ps_array.data(), dataset_size,
                   cudaMemcpyHostToDevice);
    }

    // 2. Allocate branch history buffer
    cudaMalloc(&d_branch_history_, max_stimuli_ * sizeof(uint32_t));
    cudaMemset(d_branch_history_, 0, max_stimuli_ * sizeof(uint32_t));

    initialized_ = true;
}

std::vector<uint32_t> WarpReorganizer::Reorganize(
    const std::vector<uint32_t>& representative_ids,
    uint32_t detection_node_id,
    cudaStream_t stream
) {
    if (!initialized_) {
        std::cerr << "WarpReorganizer not initialized!" << std::endl;
        return std::vector<uint32_t>();
    }

    uint32_t num_reps = representative_ids.size();
    if (num_reps == 0) {
        return std::vector<uint32_t>();
    }

    // 1. Compute PathSig for each representative stimulus
    RuntimePathSignature* d_sigs;
    cudaMalloc(&d_sigs, num_reps * sizeof(RuntimePathSignature));

    uint32_t* d_rep_ids;
    cudaMalloc(&d_rep_ids, num_reps * sizeof(uint32_t));
    cudaMemcpy(d_rep_ids, representative_ids.data(),
               num_reps * sizeof(uint32_t), cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (num_reps + block_size - 1) / block_size;

    ComputePathSignaturesKernel<<<grid_size, block_size, 0, stream>>>(
        d_rep_ids,
        num_reps,
        d_ps_dataset_,
        d_branch_history_,
        detection_node_id,
        d_sigs
    );

    cudaStreamSynchronize(stream);

    // 2. Copy PathSig to host and sort
    std::vector<RuntimePathSignature> h_sigs(num_reps);
    cudaMemcpy(h_sigs.data(), d_sigs,
               num_reps * sizeof(RuntimePathSignature),
               cudaMemcpyDeviceToHost);

    sorter_.SortByPathSignature(h_sigs, stream);

    // 3. Reassign warps based on sorted results
    std::vector<uint32_t> new_assignment(num_reps);
    sorter_.RemapWarps(h_sigs, 32, new_assignment);

    cudaFree(d_sigs);
    cudaFree(d_rep_ids);

    return new_assignment;
}

void WarpReorganizer::ClearBranchHistory() {
    if (d_branch_history_) {
        cudaMemset(d_branch_history_, 0, max_stimuli_ * sizeof(uint32_t));
    }
}

// ============================================================================
// BatchOverlapManager Implementation
// ============================================================================

BatchOverlapManager::BatchOverlapManager(uint32_t max_concurrent_batches)
    : max_concurrent_batches_(max_concurrent_batches)
{
    // Initialize free SM ID queue
    for (uint32_t i = 0; i < max_concurrent_batches_; ++i) {
        free_sm_ids_.push(i);
    }
}

BatchOverlapManager::~BatchOverlapManager() {
    WaitForAllBatches();
}

void BatchOverlapManager::LaunchBatch(
    uint32_t batch_id,
    uint32_t num_stimuli,
    void (*kernel_launcher)(uint32_t, cudaStream_t)
) {
    // Wait for free SM
    if (free_sm_ids_.empty()) {
        WaitForBatchCompletion();
    }

    BatchInfo info;
    info.batch_id = batch_id;
    info.current_cycle = 0;
    info.num_active_stimuli = num_stimuli;
    cudaStreamCreate(&info.stream);
    cudaEventCreate(&info.completion_event);

    // Launch kernel
    kernel_launcher(batch_id, info.stream);

    // Record completion event
    cudaEventRecord(info.completion_event, info.stream);

    active_batches_.push_back(info);
}

void BatchOverlapManager::WaitForBatchCompletion() {
    if (active_batches_.empty()) return;

    // Check earliest batch
    auto& batch = active_batches_.front();
    cudaEventSynchronize(batch.completion_event);

    // Release resources
    cudaStreamDestroy(batch.stream);
    cudaEventDestroy(batch.completion_event);
    free_sm_ids_.push(batch.batch_id);

    active_batches_.erase(active_batches_.begin());
}

void BatchOverlapManager::WaitForAllBatches() {
    while (!active_batches_.empty()) {
        WaitForBatchCompletion();
    }
}

// ============================================================================
// ROICalculator Implementation
// ============================================================================

float ROICalculator::ComputeROI(
    const AccumulatedMergeState& merges,
    uint32_t current_active,
    uint32_t remaining_cycles,
    float reorganization_overhead_cycles
) {
    float benefit = EstimateSavedComputations(merges, remaining_cycles);
    float cost = EstimateReorganizationCost(current_active, reorganization_overhead_cycles);

    if (cost < 1e-6f) return 0.0f;  // Avoid division by zero

    return benefit / cost;
}

float ROICalculator::EstimateSavedComputations(
    const AccumulatedMergeState& merges,
    uint32_t remaining_cycles
) {
    // Benefit: saved computation amount
    // Assume computation cost per stimulus per cycle is 1 unit
    // Savings after merging = merged count × remaining cycles
    return static_cast<float>(merges.total_merged * remaining_cycles);
}

float ROICalculator::EstimateReorganizationCost(
    uint32_t current_active,
    float overhead_per_stimulus
) {
    // Cost: reorganization overhead (in cycles)
    // Assume reorganization cost per active stimulus is fixed
    return overhead_per_stimulus * current_active;
}

} // namespace dart
