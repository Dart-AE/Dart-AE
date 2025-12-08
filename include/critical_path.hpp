// include/critical_path.hpp
#pragma once

#include "circuit.hpp"
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace dart {

// Critical path analyzer for timing analysis
class CriticalPathAnalyzer {
public:
    explicit CriticalPathAnalyzer(const Circuit& circuit);

    // Compute delays for all nodes
    std::unordered_map<uint32_t, float> ComputeNodeDelays();

    // Find the critical path (longest path from inputs to outputs)
    std::vector<uint32_t> FindCriticalPath();

    // Identify critical registers on critical paths
    std::vector<uint32_t> IdentifyCriticalRegisters();

    // Compute slack for each node
    std::unordered_map<uint32_t, float> ComputeSlacks();

    // Find all critical paths (zero slack paths)
    std::vector<std::vector<uint32_t>> FindAllCriticalPaths();

private:
    const Circuit& circuit_;

    // Get delay estimate for a node type
    float GetNodeDelay(const Node& node);
};

} // namespace dart
