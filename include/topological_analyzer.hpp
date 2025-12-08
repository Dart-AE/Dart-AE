// include/topological_analyzer.hpp
#pragma once

#include "circuit.hpp"
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace dart {

// Topological analyzer for DAG circuits
class TopologicalAnalyzer {
public:
    explicit TopologicalAnalyzer(const Circuit& circuit);

    // Compute topological sort using Kahn's algorithm
    std::vector<uint32_t> ComputeTopologicalOrder();

    // Compute topological levels (depth from inputs)
    std::unordered_map<uint32_t, uint32_t> ComputeTopologicalLevels();

    // Find strongly connected components using Tarjan's algorithm
    std::vector<std::vector<uint32_t>> FindStronglyConnectedComponents();

    // Compute reverse topological order
    std::vector<uint32_t> ComputeReverseTopologicalOrder();

    // Get maximum topological level
    uint32_t GetMaxLevel();

    // Get all nodes at a specific topological level
    std::vector<uint32_t> GetNodesAtLevel(uint32_t level);

private:
    const Circuit& circuit_;
};

} // namespace dart
