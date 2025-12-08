// src/state_reconstructor.cu
#include "state_reconstructor.hpp"
#include <cuda_runtime.h>
#include <cstring>

namespace dart {

// GPU kernel to reconstruct follower states from representatives
__global__ void ReconstructStatesKernel(
    const uint32_t* match_table,
    const uint8_t* representative_states,
    uint8_t* all_states,
    uint32_t num_stimuli,
    uint32_t state_size
) {
    uint32_t stim_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (stim_id >= num_stimuli) return;

    uint32_t rep_id = match_table[stim_id];

    // If not a representative, copy state from representative
    if (rep_id != stim_id) {
        const uint8_t* src = representative_states + rep_id * state_size;
        uint8_t* dst = all_states + stim_id * state_size;

        for (uint32_t i = 0; i < state_size; ++i) {
            dst[i] = src[i];
        }
    }
}

// GPU kernel to selectively reconstruct only specified stimuli
__global__ void SelectiveReconstructKernel(
    const uint32_t* stimulus_ids,
    const uint32_t* match_table,
    const uint8_t* representative_states,
    uint8_t* all_states,
    uint32_t num_to_reconstruct,
    uint32_t state_size
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_to_reconstruct) return;

    uint32_t stim_id = stimulus_ids[idx];
    uint32_t rep_id = match_table[stim_id];

    const uint8_t* src = representative_states + rep_id * state_size;
    uint8_t* dst = all_states + stim_id * state_size;

    for (uint32_t i = 0; i < state_size; ++i) {
        dst[i] = src[i];
    }
}

// GPU kernel to pack representative states into compact array
__global__ void PackRepresentativeStatesKernel(
    const uint32_t* representatives,
    const uint8_t* all_states,
    uint8_t* packed_states,
    uint32_t num_representatives,
    uint32_t state_size
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_representatives) return;

    uint32_t rep_id = representatives[idx];
    const uint8_t* src = all_states + rep_id * state_size;
    uint8_t* dst = packed_states + idx * state_size;

    for (uint32_t i = 0; i < state_size; ++i) {
        dst[i] = src[i];
    }
}

// GPU kernel to verify state reconstruction correctness
__global__ void VerifyReconstructionKernel(
    const uint32_t* match_table,
    const uint8_t* all_states,
    uint32_t num_stimuli,
    uint32_t state_size,
    const uint32_t* critical_regs,
    uint32_t num_critical_regs,
    bool* verification_results
) {
    uint32_t stim_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (stim_id >= num_stimuli) return;

    uint32_t rep_id = match_table[stim_id];

    // Verify critical registers match
    const uint8_t* my_state = all_states + stim_id * state_size;
    const uint8_t* rep_state = all_states + rep_id * state_size;

    bool match = true;
    for (uint32_t i = 0; i < num_critical_regs; ++i) {
        uint32_t reg_offset = critical_regs[i];
        if (my_state[reg_offset] != rep_state[reg_offset]) {
            match = false;
            break;
        }
    }

    verification_results[stim_id] = match;
}

// StateReconstructor implementation
StateReconstructor::StateReconstructor()
    : d_temp_buffer_(nullptr)
    , temp_buffer_size_(0)
    , max_stimuli_(0)
    , state_size_(0)
    , initialized_(false)
{
}

StateReconstructor::~StateReconstructor() {
    Cleanup();
}

void StateReconstructor::Initialize(uint32_t max_stimuli, uint32_t state_size) {
    max_stimuli_ = max_stimuli;
    state_size_ = state_size;

    // Allocate temporary buffer
    temp_buffer_size_ = max_stimuli_ * state_size_;
    cudaMalloc(&d_temp_buffer_, temp_buffer_size_);

    initialized_ = true;
}

void StateReconstructor::Cleanup() {
    if (d_temp_buffer_) {
        cudaFree(d_temp_buffer_);
        d_temp_buffer_ = nullptr;
    }
    initialized_ = false;
}

void StateReconstructor::ReconstructAll(
    const uint32_t* d_match_table,
    const uint8_t* d_representative_states,
    uint8_t* d_all_states,
    uint32_t num_stimuli,
    cudaStream_t stream
) {
    if (!initialized_) {
        throw std::runtime_error("StateReconstructor not initialized");
    }

    int block_size = 256;
    int grid_size = (num_stimuli + block_size - 1) / block_size;

    ReconstructStatesKernel<<<grid_size, block_size, 0, stream>>>(
        d_match_table,
        d_representative_states,
        d_all_states,
        num_stimuli,
        state_size_
    );
}

void StateReconstructor::ReconstructSelective(
    const uint32_t* d_stimulus_ids,
    const uint32_t* d_match_table,
    const uint8_t* d_representative_states,
    uint8_t* d_all_states,
    uint32_t num_to_reconstruct,
    cudaStream_t stream
) {
    if (!initialized_) {
        throw std::runtime_error("StateReconstructor not initialized");
    }

    int block_size = 256;
    int grid_size = (num_to_reconstruct + block_size - 1) / block_size;

    SelectiveReconstructKernel<<<grid_size, block_size, 0, stream>>>(
        d_stimulus_ids,
        d_match_table,
        d_representative_states,
        d_all_states,
        num_to_reconstruct,
        state_size_
    );
}

void StateReconstructor::PackRepresentatives(
    const std::vector<uint32_t>& representatives,
    const uint8_t* d_all_states,
    uint8_t* d_packed_states,
    cudaStream_t stream
) {
    if (!initialized_) {
        throw std::runtime_error("StateReconstructor not initialized");
    }

    // Copy representatives to device
    uint32_t* d_representatives;
    size_t rep_bytes = representatives.size() * sizeof(uint32_t);
    cudaMalloc(&d_representatives, rep_bytes);
    cudaMemcpyAsync(d_representatives, representatives.data(), rep_bytes,
                    cudaMemcpyHostToDevice, stream);

    int block_size = 256;
    int grid_size = (representatives.size() + block_size - 1) / block_size;

    PackRepresentativeStatesKernel<<<grid_size, block_size, 0, stream>>>(
        d_representatives,
        d_all_states,
        d_packed_states,
        representatives.size(),
        state_size_
    );

    cudaFree(d_representatives);
}

bool StateReconstructor::VerifyReconstruction(
    const uint32_t* d_match_table,
    const uint8_t* d_all_states,
    uint32_t num_stimuli,
    const std::vector<uint32_t>& critical_regs,
    cudaStream_t stream
) {
    if (!initialized_) {
        return false;
    }

    // Allocate device memory for critical registers
    uint32_t* d_critical_regs;
    cudaMalloc(&d_critical_regs, critical_regs.size() * sizeof(uint32_t));
    cudaMemcpyAsync(d_critical_regs, critical_regs.data(),
                    critical_regs.size() * sizeof(uint32_t),
                    cudaMemcpyHostToDevice, stream);

    // Allocate verification results
    bool* d_verification_results;
    cudaMalloc(&d_verification_results, num_stimuli * sizeof(bool));

    int block_size = 256;
    int grid_size = (num_stimuli + block_size - 1) / block_size;

    VerifyReconstructionKernel<<<grid_size, block_size, 0, stream>>>(
        d_match_table,
        d_all_states,
        num_stimuli,
        state_size_,
        d_critical_regs,
        critical_regs.size(),
        d_verification_results
    );

    // Check results
    std::vector<bool> h_verification_results(num_stimuli);
    cudaMemcpyAsync(h_verification_results.data(), d_verification_results,
                    num_stimuli * sizeof(bool),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // All stimuli should pass verification
    bool all_passed = true;
    for (bool result : h_verification_results) {
        if (!result) {
            all_passed = false;
            break;
        }
    }

    cudaFree(d_critical_regs);
    cudaFree(d_verification_results);

    return all_passed;
}

uint8_t* StateReconstructor::GetTempBuffer() {
    return d_temp_buffer_;
}

} // namespace dart
