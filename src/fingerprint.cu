// src/fingerprint.cu
#include "fingerprint_matcher.hpp"
#include <cuda_runtime.h>

namespace dart {

// 128-bit fingerprint hash (computed on GPU)
__device__ uint128_t ComputeFingerprint(
    const uint32_t* state_vector,
    const uint32_t* critical_regs,
    uint32_t num_critical_regs,
    uint32_t detection_node_id
) {
    uint128_t hash;
    hash.high = 0x123456789ABCDEF0ULL;
    hash.low  = 0xFEDCBA9876543210ULL;

    // Hash only critical register values
    for (uint32_t i = 0; i < num_critical_regs; ++i) {
        uint32_t reg_idx = critical_regs[i];
        uint32_t value = state_vector[reg_idx];

        // MurmurHash3-style mixing
        hash.low ^= value;
        hash.low *= 0xc6a4a7935bd1e995ULL;
        hash.low ^= hash.low >> 47;

        hash.high ^= value;
        hash.high *= 0x87c37b91114253d5ULL;
        hash.high ^= hash.high >> 33;
    }

    // Mix in detection node ID
    hash.low ^= detection_node_id;
    hash.high ^= detection_node_id;

    return hash;
}

// GPU global hash table (using CUDA atomic operations)
__global__ void FingerprintMatchingKernel(
    const uint32_t* state_vectors,      // [num_stimuli * state_size]
    const uint32_t* critical_reg_indices, // [num_critical_regs]
    uint32_t num_critical_regs,
    uint32_t state_size,
    uint32_t num_stimuli,
    uint32_t detection_node_id,
    uint128_t* fingerprints,            // Output: fingerprint for each stimulus
    uint32_t* match_table,              // Output: match table [num_stimuli]
    uint32_t* merge_count               // Output: merge count
) {
    uint32_t stim_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (stim_id >= num_stimuli) return;

    // Compute fingerprint for current stimulus
    const uint32_t* my_state = state_vectors + stim_id * state_size;
    uint128_t my_fp = ComputeFingerprint(
        my_state, critical_reg_indices, num_critical_regs, detection_node_id
    );

    fingerprints[stim_id] = my_fp;

    // Fast comparison within warp first (using warp shuffle)
    uint32_t lane_id = threadIdx.x % 32;
    bool found_match = false;
    uint32_t representative = stim_id;

    for (uint32_t i = 0; i < 32; ++i) {
        uint64_t other_fp_low = __shfl_sync(0xFFFFFFFF, my_fp.low, i);
        uint64_t other_fp_high = __shfl_sync(0xFFFFFFFF, my_fp.high, i);
        uint32_t other_id = __shfl_sync(0xFFFFFFFF, stim_id, i);

        if (i < lane_id &&
            other_fp_low == my_fp.low &&
            other_fp_high == my_fp.high) {
            found_match = true;
            representative = other_id;
            break;
        }
    }

    if (found_match) {
        match_table[stim_id] = representative;
        atomicAdd(merge_count, 1);
    } else {
        match_table[stim_id] = stim_id;  // Self is representative
    }
}

// Exact validation (check critical register values)
__global__ void ValidateCriticalRegistersKernel(
    const uint32_t* state_vectors,
    const uint32_t* critical_reg_indices,
    uint32_t num_critical_regs,
    uint32_t state_size,
    const uint32_t* match_table,
    uint32_t num_stimuli,
    bool* validation_results
) {
    uint32_t stim_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (stim_id >= num_stimuli) return;

    uint32_t rep_id = match_table[stim_id];
    if (rep_id == stim_id) {
        validation_results[stim_id] = true;  // Representative is always valid
        return;
    }

    // Validate critical register values match exactly
    const uint32_t* my_state = state_vectors + stim_id * state_size;
    const uint32_t* rep_state = state_vectors + rep_id * state_size;

    bool valid = true;
    for (uint32_t i = 0; i < num_critical_regs; ++i) {
        uint32_t reg_idx = critical_reg_indices[i];
        if (my_state[reg_idx] != rep_state[reg_idx]) {
            valid = false;
            break;
        }
    }

    validation_results[stim_id] = valid;
}

// Immediate masking - mask out follower threads
__global__ void ImmediateMaskFollowersKernel(
    const uint32_t* match_table,
    uint32_t num_stimuli,
    bool* thread_active_mask,           // Output: thread active mask
    uint32_t* num_active_remaining      // Output: remaining active threads
) {
    uint32_t stim_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (stim_id >= num_stimuli) return;

    uint32_t rep_id = match_table[stim_id];

    if (rep_id == stim_id) {
        // I am representative, stay active
        thread_active_mask[stim_id] = true;
        atomicAdd(num_active_remaining, 1);
    } else {
        // I am follower, mask immediately
        thread_active_mask[stim_id] = false;
    }
}

// Use mask in subsequent simulation kernel
__global__ void MaskedSimulationKernel(
    const uint32_t* stimulus_ids,
    const bool* thread_active_mask,
    uint32_t num_stimuli
    /* ... other simulation parameters ... */
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_stimuli) return;

    uint32_t stim_id = stimulus_ids[idx];

    // Check if masked
    if (!thread_active_mask[stim_id]) {
        return;  // Exit immediately, no computation
    }

    // Only representative stimuli execute simulation logic
    // ... normal simulation code ...
}

// State reconstruction - restore follower states when output is needed
__global__ void ReconstructFollowerStatesKernel(
    const uint32_t* match_table,
    const uint32_t* representative_states,  // Representative states
    uint32_t* all_states,                   // All stimuli states (output)
    uint32_t state_size,
    uint32_t num_stimuli
) {
    uint32_t stim_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (stim_id >= num_stimuli) return;

    uint32_t rep_id = match_table[stim_id];

    if (rep_id != stim_id) {
        // I am follower, copy state from representative
        const uint32_t* rep_state = representative_states + rep_id * state_size;
        uint32_t* my_state = all_states + stim_id * state_size;

        for (uint32_t i = 0; i < state_size; ++i) {
            my_state[i] = rep_state[i];
        }
    }
}

// Host-side wrapper class implementation
void FingerprintMatcher::Initialize(uint32_t max_stimuli) {
    cudaMalloc(&d_fingerprints_, max_stimuli * sizeof(uint128_t));
    cudaMalloc(&d_match_table_, max_stimuli * sizeof(uint32_t));
    cudaMalloc(&d_thread_active_mask_, max_stimuli * sizeof(bool));
    cudaMalloc(&d_validation_results_, max_stimuli * sizeof(bool));
}

// Execute matching and immediate masking
FingerprintMatcher::MatchResult FingerprintMatcher::MatchAndMask(
    const uint32_t* d_state_vectors,
    uint32_t num_stimuli,
    uint32_t state_size,
    const std::vector<uint32_t>& critical_regs,
    uint32_t detection_node_id,
    bool enable_immediate_masking,
    cudaStream_t stream
) {
    // 1. Compute fingerprints and match
    uint32_t* d_critical_regs;
    cudaMalloc(&d_critical_regs, critical_regs.size() * sizeof(uint32_t));
    cudaMemcpy(d_critical_regs, critical_regs.data(),
               critical_regs.size() * sizeof(uint32_t),
               cudaMemcpyHostToDevice);

    uint32_t* d_merge_count;
    cudaMalloc(&d_merge_count, sizeof(uint32_t));
    cudaMemset(d_merge_count, 0, sizeof(uint32_t));

    int block_size = 256;
    int grid_size = (num_stimuli + block_size - 1) / block_size;

    FingerprintMatchingKernel<<<grid_size, block_size, 0, stream>>>(
        d_state_vectors,
        d_critical_regs,
        critical_regs.size(),
        state_size,
        num_stimuli,
        detection_node_id,
        d_fingerprints_,
        d_match_table_,
        d_merge_count
    );

    // 2. Exact validation
    ValidateCriticalRegistersKernel<<<grid_size, block_size, 0, stream>>>(
        d_state_vectors,
        d_critical_regs,
        critical_regs.size(),
        state_size,
        d_match_table_,
        num_stimuli,
        d_validation_results_
    );

    uint32_t num_merged;
    cudaMemcpy(&num_merged, d_merge_count, sizeof(uint32_t),
               cudaMemcpyDeviceToHost);

    // 3. Immediate masking (if enabled)
    uint32_t num_active = num_stimuli;
    if (enable_immediate_masking) {
        uint32_t* d_num_active;
        cudaMalloc(&d_num_active, sizeof(uint32_t));
        cudaMemset(d_num_active, 0, sizeof(uint32_t));

        ImmediateMaskFollowersKernel<<<grid_size, block_size, 0, stream>>>(
            d_match_table_,
            num_stimuli,
            d_thread_active_mask_,
            d_num_active
        );

        cudaMemcpy(&num_active, d_num_active, sizeof(uint32_t),
                   cudaMemcpyDeviceToHost);
        cudaFree(d_num_active);
    }

    // 4. Extract results
    MatchResult result;
    result.num_merged = num_merged;
    result.num_active_remaining = num_active;

    result.match_table.resize(num_stimuli);
    cudaMemcpy(result.match_table.data(), d_match_table_,
               num_stimuli * sizeof(uint32_t),
               cudaMemcpyDeviceToHost);

    // Extract representative list
    for (uint32_t i = 0; i < num_stimuli; ++i) {
        if (result.match_table[i] == i) {
            result.representatives.push_back(i);
        }
    }

    cudaFree(d_critical_regs);
    cudaFree(d_merge_count);

    return result;
}

// Get thread active mask (for use by subsequent kernels)
bool* FingerprintMatcher::GetThreadActiveMask() {
    return d_thread_active_mask_;
}

FingerprintMatcher::~FingerprintMatcher() {
    cudaFree(d_fingerprints_);
    cudaFree(d_match_table_);
    cudaFree(d_thread_active_mask_);
    cudaFree(d_validation_results_);
}

} // namespace dart
