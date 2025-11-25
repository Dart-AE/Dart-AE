// include/dart/dart_config.hpp
#pragma once

#include <cstdint>
#include <cstdlib>
#include <string>
#include <iostream>
#include <vector>
#include <unordered_map>

namespace dart {

// 128-bit integer for fingerprints
struct uint128_t {
    uint64_t low;
    uint64_t high;
};

// DART configurable parameters
struct DARTConfig {
    // === Detection strategy parameters ===
    uint32_t detection_interval = 100;          // Detection interval (in cycles)
                                                // For 500K cycles, detect every 100 cycles = 5000 detections
                                                // Tunable range: 50-500

    uint32_t initial_detection_cycle = 50;      // Initial detection cycle (let state converge first)

    bool adaptive_detection = true;             // Adaptive detection interval
    uint32_t min_detection_interval = 50;       // Minimum detection interval
    uint32_t max_detection_interval = 500;      // Maximum detection interval

    uint32_t num_detection_nodes = 3;           // Number of detection nodes to use (1-5)

    // === Merge strategy parameters ===
    uint32_t min_merge_count = 16;              // Minimum merge count to trigger reorganization
                                                // Need to accumulate enough merges to justify reorganization

    float merge_accumulation_threshold = 0.3f;  // Accumulated merge rate threshold
                                                // Consider reorganization when 30% of active stimuli are merged

    // === Warp reorganization parameters ===
    uint32_t warp_size = 32;
    uint32_t min_warp_utilization = 16;         // Minimum warp utilization

    float roi_threshold = 2.0f;                  // ROI threshold, minimum ROI to trigger reorganization
                                                // 2.0 means benefit must be at least 2x the overhead
                                                // Tunable range: 1.5-3.0

    float reorganization_overhead_cycles = 50.0f; // Reorganization overhead (in equivalent cycles)
                                                 // Used for ROI calculation

    bool enable_path_sorting = true;            // Whether to enable PathSig sorting
    bool track_branch_history = true;           // Whether to track branch history

    // === Execution strategy parameters ===
    bool immediate_masking = true;              // Immediately mask followers (don't wait for reorganization)
    bool enable_batch_overlap = true;           // Enable batch overlapping execution

    // === Debug parameters ===
    bool verbose = false;                       // Verbose output
    bool debug_fingerprint = false;             // Debug fingerprint matching
    bool debug_reorganization = false;          // Debug reorganization process
    bool print_detection_stats = false;         // Print statistics for each detection

    // Load from command line arguments
    void LoadFromCommandLine(int argc, char** argv) {
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];

            if (arg == "--detection-interval" && i + 1 < argc) {
                detection_interval = std::atoi(argv[++i]);
            }
            else if (arg == "--roi-threshold" && i + 1 < argc) {
                roi_threshold = std::atof(argv[++i]);
            }
            else if (arg == "--num-detection-nodes" && i + 1 < argc) {
                num_detection_nodes = std::atoi(argv[++i]);
            }
            else if (arg == "--min-merge-count" && i + 1 < argc) {
                min_merge_count = std::atoi(argv[++i]);
            }
            else if (arg == "--disable-immediate-masking") {
                immediate_masking = false;
            }
            else if (arg == "--disable-adaptive-detection") {
                adaptive_detection = false;
            }
            else if (arg == "--verbose") {
                verbose = true;
            }
        }
    }

    // Print configuration
    void Print() const {
        std::cout << "\n=== DART Configuration ===\n";
        std::cout << "Detection interval: " << detection_interval << " cycles\n";
        std::cout << "ROI threshold: " << roi_threshold << "\n";
        std::cout << "Detection nodes: " << num_detection_nodes << "\n";
        std::cout << "Minimum merge count: " << min_merge_count << "\n";
        std::cout << "Immediate masking: " << (immediate_masking ? "Enabled" : "Disabled") << "\n";
        std::cout << "Adaptive detection: " << (adaptive_detection ? "Enabled" : "Disabled") << "\n";
        std::cout << "PathSig sorting: " << (enable_path_sorting ? "Enabled" : "Disabled") << "\n";
    }
};

// DAG node metadata (§4.1)
struct DAGNodeMetadata {
    uint32_t node_id;
    uint32_t topological_level;     // Topological level
    float    convergence_score;      // Convergence score Score(v)
    uint32_t fanout;
    uint32_t critical_reg_count;
    std::vector<uint32_t> critical_regs;  // R_critical
};

// Path signature feature set (PS) - precomputed at compile time (§4.1 & §4.3)
struct PathSignatureFeatures {
    uint32_t node_id;                           // Node ID

    // P(v): predecessor node signatures
    uint32_t num_predecessors;
    uint64_t predecessor_signatures[32];        // Fixed-size array for GPU access

    // L(v): topological level identifier
    uint32_t topological_level;

    // Control flow nodes (for H(si) construction)
    uint32_t num_control_flow_nodes;
    uint32_t control_flow_node_ids[16];         // Nodes affecting control flow

    // Precomputed hash components (optimization)
    uint64_t level_hash;                        // Pre-hashed level value
    uint64_t predecessors_combined_hash;        // Combined hash of all predecessors
};

// Runtime path signature (§4.3 Equation 4)
struct RuntimePathSignature {
    uint64_t signature;         // PathSig(v, si) = Hash(P(v), L(v), H(si))
    uint32_t stimulus_id;       // Stimulus ID
    uint32_t detection_node;    // Detection node ID
    uint32_t original_warp_id;  // Original warp ID (for before/after comparison)
};

// Branch history record H(si) - dynamically built at runtime
struct BranchHistory {
    uint32_t stimulus_id;
    uint32_t history_bits;      // Bit vector, each bit records a branch choice
    uint32_t num_branches;      // Number of recorded branches

    // Optional: detailed branch record (for debugging)
    struct BranchRecord {
        uint32_t node_id;
        uint32_t cycle;
        bool taken;
    };
    std::vector<BranchRecord> detailed_history;  // Only enabled in debug mode
};

// Detection node configuration
struct DetectionNode {
    uint32_t node_id;
    float    score;                  // Comprehensive score
    uint32_t level;
    std::vector<uint32_t> input_ports;
    std::vector<uint32_t> critical_inputs;  // Filtered critical inputs
};

// Fingerprint structure (§4.2)
struct Fingerprint {
    uint128_t hash;                  // 128-bit fingerprint
    uint32_t  detection_node;
    uint32_t  stimulus_id;
};

// Stimulus merge record
struct MergeRecord {
    uint32_t representative_id;      // Representative stimulus ID
    std::vector<uint32_t> follower_ids;  // Follower ID list
    uint32_t merge_cycle;            // Cycle when merge occurred
    uint32_t detection_node;
    uint64_t path_signature;         // Representative's path signature
};

// Accumulated merge state (accumulated across multiple detections)
struct AccumulatedMergeState {
    std::unordered_map<uint32_t, uint32_t> representative_map; // follower -> representative
    std::unordered_map<uint32_t, std::vector<uint32_t>> follower_groups; // representative -> followers

    uint32_t total_merged = 0;       // Accumulated merged stimuli count
    uint32_t num_representatives = 0; // Current representative count

    void Clear() {
        representative_map.clear();
        follower_groups.clear();
        total_merged = 0;
        num_representatives = 0;
    }

    float GetMergeRatio(uint32_t total_stimuli) const {
        return (float)total_merged / total_stimuli;
    }
};

// DART runtime statistics
struct DARTStats {
    uint64_t total_stimuli;
    uint64_t merged_stimuli;
    uint64_t reorganization_count;
    uint64_t detection_count;               // Total detection count
    uint64_t immediate_masked_count;        // Immediately masked follower count

    double   detection_time_ms;
    double   merging_time_ms;
    double   reorganization_time_ms;
    double   path_sig_computation_ms;       // PathSig computation time
    double   sorting_time_ms;               // Sorting time
    double   masking_time_ms;               // Masking time
    double   total_time_ms;

    // PathSig related statistics
    uint32_t num_path_sig_groups;           // LSH group count
    float    avg_group_size;                // Average group size

    // Detection interval adaptive statistics
    std::vector<uint32_t> detection_intervals_history;  // Historical detection intervals
    std::vector<float> redundancy_history;              // Historical redundancy rates

    float GetRedundancyRatio() const {
        return (float)merged_stimuli / total_stimuli;
    }

    float GetSpeedup(double baseline_time_ms) const {
        return baseline_time_ms / total_time_ms;
    }

    float GetPathSignatureOverheadRatio() const {
        return (path_sig_computation_ms + sorting_time_ms) / total_time_ms;
    }

    float GetDetectionOverheadRatio() const {
        return detection_time_ms / total_time_ms;
    }

    float GetAverageDetectionInterval() const {
        if (detection_intervals_history.empty()) return 0.0f;
        float sum = 0;
        for (auto interval : detection_intervals_history) {
            sum += interval;
        }
        return sum / detection_intervals_history.size();
    }
};

} // namespace dart
