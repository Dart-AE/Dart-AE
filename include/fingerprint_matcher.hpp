#ifndef DART_FINGERPRINT_MATCHER_HPP
#define DART_FINGERPRINT_MATCHER_HPP

#include <vector>
#include <cstdint>
#include <cuda_runtime.h>
#include "dart/dart_config.hpp"

namespace dart {

// uint128_t and Fingerprint are already defined in dart_config.hpp

// CUDA Kernel function declarations
__device__ uint128_t ComputeFingerprint(
    const uint32_t* state_vector,
    const uint32_t* critical_regs,
    uint32_t num_critical_regs,
    uint32_t detection_node_id
);

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
);

__global__ void ValidateCriticalRegistersKernel(
    const uint32_t* state_vectors,
    const uint32_t* critical_reg_indices,
    uint32_t num_critical_regs,
    uint32_t state_size,
    const uint32_t* match_table,
    uint32_t num_stimuli,
    bool* validation_results
);

__global__ void ImmediateMaskFollowersKernel(
    const uint32_t* match_table,
    uint32_t num_stimuli,
    bool* thread_active_mask,           // Output: thread active mask
    uint32_t* num_active_remaining      // Output: remaining active thread count
);

__global__ void MaskedSimulationKernel(
    const uint32_t* stimulus_ids,
    const bool* thread_active_mask,
    uint32_t num_stimuli
    /* ... other simulation parameters ... */
);

__global__ void ReconstructFollowerStatesKernel(
    const uint32_t* match_table,
    const uint32_t* representative_states,  // Representative states
    uint32_t* all_states,                   // All stimuli states (output)
    uint32_t state_size,
    uint32_t num_stimuli
);

// Host-side wrapper class
class FingerprintMatcher {
private:
    uint128_t* d_fingerprints_;
    uint32_t* d_match_table_;
    bool* d_thread_active_mask_;
    bool* d_validation_results_;

public:
    // Match result structure
    struct MatchResult {
        std::vector<uint32_t> representatives;
        std::vector<uint32_t> match_table;
        uint32_t num_merged;
        uint32_t num_active_remaining;
    };

    // Initialize
    void Initialize(uint32_t max_stimuli);

    // Execute matching and immediate masking
    MatchResult MatchAndMask(
        const uint32_t* d_state_vectors,
        uint32_t num_stimuli,
        uint32_t state_size,
        const std::vector<uint32_t>& critical_regs,
        uint32_t detection_node_id,
        bool enable_immediate_masking,
        cudaStream_t stream
    );

    // Get thread active mask (for use by subsequent kernels)
    bool* GetThreadActiveMask();

    // Destructor
    ~FingerprintMatcher();
};

} // namespace dart

#endif // DART_FINGERPRINT_MATCHER_HPP
