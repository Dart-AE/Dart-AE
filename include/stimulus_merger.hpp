// include/stimulus_merger.hpp
#pragma once

#include "dart_config.hpp"
#include <vector>
#include <cstdint>
#include <cuda_runtime.h>

namespace dart {

// Result of stimulus merging operation
struct MergeResult {
    uint32_t num_groups;
    uint32_t num_representatives;
    std::vector<uint32_t> merge_groups;      // Group ID for each stimulus
    std::vector<uint32_t> representatives;    // List of representative IDs
};

// Merge statistics
struct MergeStatistics {
    uint32_t total_stimuli;
    uint32_t num_representatives;
    uint32_t num_merged;
    float merge_ratio;
};

// Stimulus merger engine
class StimulusMerger {
public:
    StimulusMerger();
    ~StimulusMerger();

    // Initialize with maximum number of stimuli
    void Initialize(uint32_t max_stimuli);

    // Merge stimuli based on fingerprints
    MergeResult MergeStimuli(
        const uint128_t* d_fingerprints,
        uint32_t num_stimuli,
        cudaStream_t stream = nullptr
    );

    // Build merge records from merge result
    std::vector<MergeRecord> BuildMergeRecords(
        const MergeResult& merge_result,
        uint32_t detection_node
    );

    // Compute merge statistics
    MergeStatistics ComputeStatistics(const MergeResult& merge_result);

    // Cleanup resources
    void Cleanup();

private:
    uint32_t* d_merge_groups_;
    uint32_t* d_group_sizes_;
    uint32_t* d_representatives_;
    uint32_t max_stimuli_;
    bool initialized_;
};

} // namespace dart
