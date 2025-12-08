// src/stimulus_merger.cu
#include "stimulus_merger.hpp"
#include <cuda_runtime.h>
#include <unordered_map>

namespace dart {

// GPU kernel to compute merge groups based on fingerprints
__global__ void ComputeMergeGroupsKernel(
    const uint128_t* fingerprints,
    uint32_t num_stimuli,
    uint32_t* merge_groups,      // Output: group ID for each stimulus
    uint32_t* group_sizes,       // Output: size of each group
    uint32_t* num_groups         // Output: total number of groups
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_stimuli) return;

    // Simple approach: use fingerprint hash as group ID
    uint64_t group_id = fingerprints[tid].low % num_stimuli;
    merge_groups[tid] = group_id;

    // Atomically increment group size
    atomicAdd(&group_sizes[group_id], 1);
}

// GPU kernel to compact representatives
__global__ void CompactRepresentativesKernel(
    const uint32_t* merge_groups,
    uint32_t num_stimuli,
    uint32_t* representatives,    // Output: list of representative IDs
    uint32_t* rep_count          // Output: number of representatives
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_stimuli) return;

    // Check if this is the first occurrence of the group
    bool is_representative = true;
    for (uint32_t i = 0; i < tid; ++i) {
        if (merge_groups[i] == merge_groups[tid]) {
            is_representative = false;
            break;
        }
    }

    if (is_representative) {
        uint32_t idx = atomicAdd(rep_count, 1);
        representatives[idx] = tid;
    }
}

// GPU kernel to merge stimulus data
__global__ void MergeStimulusDataKernel(
    const uint32_t* merge_table,
    const uint32_t* stimulus_data,
    uint32_t num_stimuli,
    uint32_t data_size,
    uint32_t* merged_data         // Output: merged stimulus data
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_stimuli) return;

    uint32_t rep_id = merge_table[tid];

    // Copy data from representative
    for (uint32_t i = 0; i < data_size; ++i) {
        merged_data[tid * data_size + i] = stimulus_data[rep_id * data_size + i];
    }
}

// StimulusMerger implementation
StimulusMerger::StimulusMerger()
    : d_merge_groups_(nullptr)
    , d_group_sizes_(nullptr)
    , d_representatives_(nullptr)
    , max_stimuli_(0)
    , initialized_(false)
{
}

StimulusMerger::~StimulusMerger() {
    Cleanup();
}

void StimulusMerger::Initialize(uint32_t max_stimuli) {
    max_stimuli_ = max_stimuli;

    cudaMalloc(&d_merge_groups_, max_stimuli_ * sizeof(uint32_t));
    cudaMalloc(&d_group_sizes_, max_stimuli_ * sizeof(uint32_t));
    cudaMalloc(&d_representatives_, max_stimuli_ * sizeof(uint32_t));

    initialized_ = true;
}

void StimulusMerger::Cleanup() {
    if (d_merge_groups_) {
        cudaFree(d_merge_groups_);
        d_merge_groups_ = nullptr;
    }
    if (d_group_sizes_) {
        cudaFree(d_group_sizes_);
        d_group_sizes_ = nullptr;
    }
    if (d_representatives_) {
        cudaFree(d_representatives_);
        d_representatives_ = nullptr;
    }
    initialized_ = false;
}

MergeResult StimulusMerger::MergeStimuli(
    const uint128_t* d_fingerprints,
    uint32_t num_stimuli,
    cudaStream_t stream
) {
    if (!initialized_) {
        throw std::runtime_error("StimulusMerger not initialized");
    }

    MergeResult result;

    // Reset group sizes
    cudaMemsetAsync(d_group_sizes_, 0, max_stimuli_ * sizeof(uint32_t), stream);

    // Compute merge groups
    uint32_t* d_num_groups;
    cudaMalloc(&d_num_groups, sizeof(uint32_t));
    cudaMemsetAsync(d_num_groups, 0, sizeof(uint32_t), stream);

    int block_size = 256;
    int grid_size = (num_stimuli + block_size - 1) / block_size;

    ComputeMergeGroupsKernel<<<grid_size, block_size, 0, stream>>>(
        d_fingerprints,
        num_stimuli,
        d_merge_groups_,
        d_group_sizes_,
        d_num_groups
    );

    // Compact representatives
    uint32_t* d_rep_count;
    cudaMalloc(&d_rep_count, sizeof(uint32_t));
    cudaMemsetAsync(d_rep_count, 0, sizeof(uint32_t), stream);

    CompactRepresentativesKernel<<<grid_size, block_size, 0, stream>>>(
        d_merge_groups_,
        num_stimuli,
        d_representatives_,
        d_rep_count
    );

    // Copy results back to host
    uint32_t num_groups, num_reps;
    cudaMemcpyAsync(&num_groups, d_num_groups, sizeof(uint32_t),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&num_reps, d_rep_count, sizeof(uint32_t),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    result.num_groups = num_groups;
    result.num_representatives = num_reps;

    // Copy merge groups and representatives
    result.merge_groups.resize(num_stimuli);
    result.representatives.resize(num_reps);

    cudaMemcpyAsync(result.merge_groups.data(), d_merge_groups_,
                    num_stimuli * sizeof(uint32_t),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(result.representatives.data(), d_representatives_,
                    num_reps * sizeof(uint32_t),
                    cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    // Cleanup temporary buffers
    cudaFree(d_num_groups);
    cudaFree(d_rep_count);

    return result;
}

std::vector<MergeRecord> StimulusMerger::BuildMergeRecords(
    const MergeResult& merge_result,
    uint32_t detection_node
) {
    std::unordered_map<uint32_t, std::vector<uint32_t>> group_map;

    // Group stimuli by their merge group
    for (size_t i = 0; i < merge_result.merge_groups.size(); ++i) {
        uint32_t group_id = merge_result.merge_groups[i];
        group_map[group_id].push_back(i);
    }

    // Build merge records
    std::vector<MergeRecord> records;
    for (const auto& [group_id, stimuli] : group_map) {
        if (stimuli.size() > 1) {
            MergeRecord record;
            record.representative_id = stimuli[0];
            record.follower_ids.assign(stimuli.begin() + 1, stimuli.end());
            record.detection_node = detection_node;
            records.push_back(record);
        }
    }

    return records;
}

// Get merge statistics
MergeStatistics StimulusMerger::ComputeStatistics(const MergeResult& merge_result) {
    MergeStatistics stats;

    stats.total_stimuli = merge_result.merge_groups.size();
    stats.num_representatives = merge_result.num_representatives;
    stats.num_merged = stats.total_stimuli - stats.num_representatives;

    if (stats.total_stimuli > 0) {
        stats.merge_ratio = static_cast<float>(stats.num_merged) /
                           static_cast<float>(stats.total_stimuli);
    } else {
        stats.merge_ratio = 0.0f;
    }

    return stats;
}

} // namespace dart
