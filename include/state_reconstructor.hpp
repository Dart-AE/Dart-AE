// include/state_reconstructor.hpp
#pragma once

#include <vector>
#include <cstdint>
#include <cuda_runtime.h>

namespace dart {

// State reconstruction manager for follower stimuli
class StateReconstructor {
public:
    StateReconstructor();
    ~StateReconstructor();

    // Initialize with maximum stimuli and state size
    void Initialize(uint32_t max_stimuli, uint32_t state_size);

    // Reconstruct all follower states from representatives
    void ReconstructAll(
        const uint32_t* d_match_table,
        const uint8_t* d_representative_states,
        uint8_t* d_all_states,
        uint32_t num_stimuli,
        cudaStream_t stream = nullptr
    );

    // Selectively reconstruct specific stimuli
    void ReconstructSelective(
        const uint32_t* d_stimulus_ids,
        const uint32_t* d_match_table,
        const uint8_t* d_representative_states,
        uint8_t* d_all_states,
        uint32_t num_to_reconstruct,
        cudaStream_t stream = nullptr
    );

    // Pack representative states into compact array
    void PackRepresentatives(
        const std::vector<uint32_t>& representatives,
        const uint8_t* d_all_states,
        uint8_t* d_packed_states,
        cudaStream_t stream = nullptr
    );

    // Verify reconstruction correctness
    bool VerifyReconstruction(
        const uint32_t* d_match_table,
        const uint8_t* d_all_states,
        uint32_t num_stimuli,
        const std::vector<uint32_t>& critical_regs,
        cudaStream_t stream = nullptr
    );

    // Get temporary buffer
    uint8_t* GetTempBuffer();

    // Cleanup resources
    void Cleanup();

private:
    uint8_t* d_temp_buffer_;
    size_t temp_buffer_size_;
    uint32_t max_stimuli_;
    uint32_t state_size_;
    bool initialized_;
};

} // namespace dart
