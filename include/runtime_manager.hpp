// include/runtime_manager.hpp
#pragma once

#include "dart_config.hpp"
#include "fingerprint_matcher.hpp"
#include "execution_manager.hpp"
#include "state_reconstructor.hpp"
#include "stimulus_merger.hpp"
#include "batch_overlapper.hpp"
#include <memory>
#include <unordered_map>
#include <cuda_runtime.h>

namespace dart {

// DART configuration
struct DARTConfig {
    uint32_t max_stimuli = 65536;
    uint32_t state_size = 4096;
    uint32_t max_concurrent_batches = 4;
    bool enable_warp_reorg = true;
    bool enable_batch_overlap = true;
    float roi_threshold = 2.0f;
};

// Execution result
struct ExecutionResult {
    uint32_t num_total;
    uint32_t num_merged;
    uint32_t num_active;
    float merge_ratio;
    float speedup;

    float fingerprint_time_ms;
    float reorganization_time_ms;
    float reconstruction_time_ms;
    float total_time_ms;
};

// Main DART runtime manager
class DARTRuntimeManager {
public:
    DARTRuntimeManager();
    ~DARTRuntimeManager();

    // Initialize DART runtime
    void Initialize(const DARTConfig& config);

    // Set path signature dataset (from DAG analyzer)
    void SetPathSignatureDataset(
        const std::unordered_map<uint32_t, PathSignatureFeatures>& ps_dataset
    );

    // Execute a batch of stimuli with DART optimizations
    ExecutionResult ExecuteBatch(
        uint8_t* d_state_vectors,
        uint32_t num_stimuli,
        uint32_t num_cycles,
        const std::vector<uint32_t>& critical_regs
    );

    // Set detection node
    void SetDetectionNode(uint32_t node_id);

    // Enable/disable immediate masking
    void EnableImmediateMasking(bool enable);

    // Enable/disable warp reorganization
    void EnableWarpReorganization(bool enable);

    // Get statistics
    DARTStats GetStatistics() const;

    // Reset statistics
    void ResetStatistics();

    // Get CUDA stream
    cudaStream_t GetStream() const;

private:
    void Cleanup();

    // Configuration
    DARTConfig config_;

    // Core components
    FingerprintMatcher fingerprint_matcher_;
    WarpReorganizer warp_reorganizer_;
    StateReconstructor state_reconstructor_;
    StimulusMerger stimulus_merger_;
    std::unique_ptr<BatchOverlapper> batch_overlapper_;

    // Path signature dataset
    std::unordered_map<uint32_t, PathSignatureFeatures> ps_dataset_;

    // Runtime state
    cudaStream_t stream_;
    uint32_t detection_node_id_;
    bool enable_immediate_masking_;
    bool enable_warp_reorganization_;
    bool initialized_;
};

} // namespace dart
