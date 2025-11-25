/*
 * Simple DART Integration Example
 * Demonstrates how to use the DART library in your own project
 */

#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// DART header files
#include "dart/dart_config.hpp"
#include "dart/fingerprint_matcher.hpp"
#include "dart/execution_manager.hpp"

int main() {
    std::cout << "╔════════════════════════════════════════════════╗\n";
    std::cout << "║        DART Simple Integration Example        ║\n";
    std::cout << "╚════════════════════════════════════════════════╝\n\n";

    // Configuration parameters
    const size_t NUM_TESTBENCHES = 1024;
    const size_t STATE_SIZE = 128;  // State size per testbench

    std::cout << "Initializing DART components...\n";
    std::cout << "  Testbenches: " << NUM_TESTBENCHES << "\n";
    std::cout << "  State Size: " << STATE_SIZE << " uint32_t\n\n";

    // 1. Initialize DART components
    dart::FingerprintMatcher matcher;
    matcher.Initialize(NUM_TESTBENCHES);

    dart::WarpReorganizer reorganizer;
    // Note: WarpReorganizer requires PathSignature dataset initialization
    // This is for demonstration only, actual use requires providing PS dataset

    // 2. Allocate GPU memory
    uint32_t* d_states;
    size_t state_bytes = NUM_TESTBENCHES * STATE_SIZE * sizeof(uint32_t);
    cudaMalloc(&d_states, state_bytes);

    std::cout << "Allocated GPU memory: "
              << (state_bytes / (1024.0 * 1024.0)) << " MB\n\n";

    // 3. Prepare critical registers list
    std::vector<uint32_t> critical_regs;
    for (uint32_t i = 0; i < 32; i++) {
        critical_regs.push_back(i * 4);  // Select 32 critical registers
    }

    std::cout << "Selected " << critical_regs.size() << " critical registers\n\n";

    // 4. Perform fingerprint matching (example)
    std::cout << "Performing fingerprint matching...\n";

    try {
        dart::FingerprintMatcher::MatchResult result = matcher.MatchAndMask(
            d_states,
            NUM_TESTBENCHES,
            STATE_SIZE,
            critical_regs,
            1,                              // detection_node_id
            true,                           // enable_immediate_masking
            0                               // stream
        );

        std::cout << "Match Result:\n";
        std::cout << "  Representatives: " << result.representatives.size() << "\n";
        std::cout << "  Merged: " << result.num_merged << "\n";
        std::cout << "  Active Remaining: " << result.num_active_remaining << "\n";
        std::cout << "  Redundancy Rate: "
                  << (result.num_merged * 100.0 / NUM_TESTBENCHES) << "%\n\n";

    } catch (const std::exception& e) {
        std::cerr << "Error during matching: " << e.what() << "\n";
    }

    // 5. Clean up resources
    cudaFree(d_states);

    std::cout << "╔════════════════════════════════════════════════╗\n";
    std::cout << "║              Example Complete!                 ║\n";
    std::cout << "╚════════════════════════════════════════════════╝\n";

    return 0;
}
