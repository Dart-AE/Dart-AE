// src/runtime_manager.cpp
#include "runtime_manager.hpp"
#include <iostream>
#include <chrono>

namespace dart {

DARTRuntimeManager::DARTRuntimeManager()
    : initialized_(false)
    , detection_node_id_(0)
    , enable_immediate_masking_(true)
    , enable_warp_reorganization_(true)
{
}

DARTRuntimeManager::~DARTRuntimeManager() {
    Cleanup();
}

void DARTRuntimeManager::Initialize(const DARTConfig& config) {
    config_ = config;

    // Initialize fingerprint matcher
    fingerprint_matcher_.Initialize(config_.max_stimuli);

    // Initialize warp reorganizer
    if (config_.enable_warp_reorg) {
        // PS dataset should be provided by DAG analyzer
        warp_reorganizer_.Initialize(ps_dataset_, config_.max_stimuli);
    }

    // Initialize state reconstructor
    state_reconstructor_.Initialize(config_.max_stimuli, config_.state_size);

    // Initialize stimulus merger
    stimulus_merger_.Initialize(config_.max_stimuli);

    // Initialize batch overlapper
    if (config_.enable_batch_overlap) {
        batch_overlapper_ = std::make_unique<BatchOverlapper>(config_.max_concurrent_batches);
    }

    // Create CUDA stream
    cudaStreamCreate(&stream_);

    initialized_ = true;

    std::cout << "DART Runtime Manager initialized successfully" << std::endl;
    std::cout << "  Max stimuli: " << config_.max_stimuli << std::endl;
    std::cout << "  State size: " << config_.state_size << std::endl;
    std::cout << "  Warp reorganization: " << (config_.enable_warp_reorg ? "enabled" : "disabled") << std::endl;
    std::cout << "  Batch overlap: " << (config_.enable_batch_overlap ? "enabled" : "disabled") << std::endl;
}

void DARTRuntimeManager::Cleanup() {
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
    initialized_ = false;
}

void DARTRuntimeManager::SetPathSignatureDataset(
    const std::unordered_map<uint32_t, PathSignatureFeatures>& ps_dataset
) {
    ps_dataset_ = ps_dataset;

    if (initialized_ && config_.enable_warp_reorg) {
        warp_reorganizer_.Initialize(ps_dataset_, config_.max_stimuli);
    }
}

ExecutionResult DARTRuntimeManager::ExecuteBatch(
    uint8_t* d_state_vectors,
    uint32_t num_stimuli,
    uint32_t num_cycles,
    const std::vector<uint32_t>& critical_regs
) {
    if (!initialized_) {
        throw std::runtime_error("DARTRuntimeManager not initialized");
    }

    ExecutionResult result;
    auto start_time = std::chrono::high_resolution_clock::now();

    // Phase 1: Fingerprint matching and detection
    auto fp_start = std::chrono::high_resolution_clock::now();

    auto match_result = fingerprint_matcher_.MatchAndMask(
        reinterpret_cast<uint32_t*>(d_state_vectors),
        num_stimuli,
        config_.state_size / sizeof(uint32_t),
        critical_regs,
        detection_node_id_,
        enable_immediate_masking_,
        stream_
    );

    auto fp_end = std::chrono::high_resolution_clock::now();
    float fp_time = std::chrono::duration<float, std::milli>(fp_end - fp_start).count();

    result.num_total = num_stimuli;
    result.num_merged = match_result.num_merged;
    result.num_active = match_result.num_active_remaining;

    // Phase 2: Warp reorganization (if enabled)
    float reorg_time = 0.0f;
    if (config_.enable_warp_reorg && !match_result.representatives.empty()) {
        auto reorg_start = std::chrono::high_resolution_clock::now();

        auto new_assignment = warp_reorganizer_.Reorganize(
            match_result.representatives,
            detection_node_id_,
            stream_
        );

        auto reorg_end = std::chrono::high_resolution_clock::now();
        reorg_time = std::chrono::duration<float, std::milli>(reorg_end - reorg_start).count();
    }

    // Phase 3: Execute simulation with masked stimuli
    // (Simulation kernel would be called here by the user)
    // We just prepare the execution environment

    // Phase 4: State reconstruction
    auto recon_start = std::chrono::high_resolution_clock::now();

    state_reconstructor_.ReconstructAll(
        match_result.match_table.data(),
        d_state_vectors,  // Representative states
        d_state_vectors,  // All states (in-place reconstruction)
        num_stimuli,
        stream_
    );

    cudaStreamSynchronize(stream_);

    auto recon_end = std::chrono::high_resolution_clock::now();
    float recon_time = std::chrono::duration<float, std::milli>(recon_end - recon_start).count();

    auto end_time = std::chrono::high_resolution_clock::now();
    float total_time = std::chrono::duration<float, std::milli>(end_time - start_time).count();

    // Fill in timing results
    result.fingerprint_time_ms = fp_time;
    result.reorganization_time_ms = reorg_time;
    result.reconstruction_time_ms = recon_time;
    result.total_time_ms = total_time;

    // Compute statistics
    result.merge_ratio = static_cast<float>(result.num_merged) /
                        static_cast<float>(result.num_total);
    result.speedup = 1.0f / (1.0f - result.merge_ratio);  // Amdahl's law approximation

    return result;
}

void DARTRuntimeManager::SetDetectionNode(uint32_t node_id) {
    detection_node_id_ = node_id;
}

void DARTRuntimeManager::EnableImmediateMasking(bool enable) {
    enable_immediate_masking_ = enable;
}

void DARTRuntimeManager::EnableWarpReorganization(bool enable) {
    enable_warp_reorganization_ = enable;
}

DARTStats DARTRuntimeManager::GetStatistics() const {
    DARTStats stats;
    // Accumulated statistics would be tracked across multiple executions
    // For now, return default values
    return stats;
}

void DARTRuntimeManager::ResetStatistics() {
    // Reset accumulated statistics
}

cudaStream_t DARTRuntimeManager::GetStream() const {
    return stream_;
}

} // namespace dart
