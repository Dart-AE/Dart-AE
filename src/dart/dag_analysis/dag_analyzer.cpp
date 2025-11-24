// src/dart/dag_analysis/dag_analyzer.cpp
#include "dart/dag_analyzer.hpp"
#include "dart/dart_config.hpp"
#include <queue>
#include <cmath>
#include <algorithm>

namespace dart {

DAGAnalyzer::DAGAnalyzer(const Circuit& circuit) : circuit_(circuit) {
    // Initialize node_metadata_ vector size
    const auto& all_nodes = circuit_.GetAllNodes();
    if (!all_nodes.empty()) {
        uint32_t max_node_id = 0;
        for (const auto& node : all_nodes) {
            if (node.id > max_node_id) {
                max_node_id = node.id;
            }
        }
        node_metadata_.resize(max_node_id + 1);
    }
}

// Topological level computation (L(v) in Equation 2)
void DAGAnalyzer::ComputeTopologicalLevels() {
    std::unordered_map<uint32_t, uint32_t> levels;
    std::queue<uint32_t> queue;

    // Start from input nodes (level = 0)
    for (auto& node : circuit_.GetInputNodes()) {
        levels[node.id] = 0;
        queue.push(node.id);
    }

    // BFS to compute longest path
    while (!queue.empty()) {
        uint32_t current = queue.front();
        queue.pop();

        for (auto successor : circuit_.GetSuccessors(current)) {
            uint32_t new_level = levels[current] + 1;
            if (levels.find(successor) == levels.end() ||
                levels[successor] < new_level) {
                levels[successor] = new_level;
                queue.push(successor);
            }
        }
    }

    // Save to metadata
    for (auto& [node_id, level] : levels) {
        if (node_id < node_metadata_.size()) {
            node_metadata_[node_id].topological_level = level;
        }
    }
}

// Compute detection node score (Equation 1)
float DAGAnalyzer::ComputeScore(uint32_t node_id) {
    if (node_id >= node_metadata_.size()) {
        return 0.0f;
    }

    auto& meta = node_metadata_[node_id];

    // w_type: operation type weight
    float w_type = GetTypeWeight(circuit_.GetNodeType(node_id));

    // w_fanout: fanout weight (normalized)
    float fanout = circuit_.GetFanout(node_id);
    float max_fanout = circuit_.GetMaxFanout();
    float w_fanout = (max_fanout > 0) ? (fanout / max_fanout) : 0.0f;

    // w_level: level weight (Gaussian function, Equation 2)
    uint32_t max_level = circuit_.GetMaxLevel();
    uint32_t L_mid = max_level / 2;
    float sigma = max_level / 6.0f;  // Controls decay rate

    float level_diff = static_cast<float>(meta.topological_level) - L_mid;
    float w_level = std::exp(-(level_diff * level_diff) / (2 * sigma * sigma));

    return w_type * w_fanout * w_level;
}

// Select optimal detection nodes
std::vector<DetectionNode> DAGAnalyzer::SelectDetectionNodes(size_t count) {
    std::vector<DetectionNode> candidates;

    for (auto& node : circuit_.GetAllNodes()) {
        if (node.type == NodeType::OPERATION) {  // Only consider operation nodes
            DetectionNode dn;
            dn.node_id = node.id;
            dn.score = ComputeScore(node.id);
            dn.level = (node.id < node_metadata_.size()) ?
                       node_metadata_[node.id].topological_level : 0;
            candidates.push_back(dn);
        }
    }

    // Sort by score
    std::sort(candidates.begin(), candidates.end(),
             [](const auto& a, const auto& b) { return a.score > b.score; });

    // Return top-k
    candidates.resize(std::min(count, candidates.size()));
    return candidates;
}

// Critical register identification (backward data flow tracing)
std::vector<uint32_t> DAGAnalyzer::ExtractCriticalRegisters(uint32_t node_id) {
    std::vector<uint32_t> critical_regs;
    std::unordered_set<uint32_t> visited;
    std::queue<uint32_t> queue;

    queue.push(node_id);
    visited.insert(node_id);

    while (!queue.empty()) {
        uint32_t current = queue.front();
        queue.pop();

        auto& node = circuit_.GetNode(current);

        // If it's a register and affects control flow
        if (node.type == NodeType::REGISTER &&
            node.affects_control_flow) {
            critical_regs.push_back(current);
        }

        // Continue backward tracing
        for (auto pred : circuit_.GetPredecessors(current)) {
            if (visited.find(pred) == visited.end()) {
                visited.insert(pred);
                queue.push(pred);
            }
        }
    }

    return critical_regs;
}

// Build PS feature set for each DAG node
std::unordered_map<uint32_t, PathSignatureFeatures>
DAGAnalyzer::BuildPathSignatureDataset() {
    std::unordered_map<uint32_t, PathSignatureFeatures> ps_dataset;

    for (auto& node : circuit_.GetAllNodes()) {
        PathSignatureFeatures ps;
        ps.node_id = node.id;

        // 1. Extract predecessor node IDs
        auto predecessors = circuit_.GetPredecessors(node.id);
        ps.num_predecessors = std::min(static_cast<uint32_t>(predecessors.size()), 32u);

        // 2. Topological level
        ps.topological_level = (node.id < node_metadata_.size()) ?
                               node_metadata_[node.id].topological_level : 0;

        // 3. Extract control flow related nodes (branches, conditionals, etc.)
        auto cf_nodes = ExtractControlFlowNodes(node.id);
        ps.num_control_flow_nodes = std::min(static_cast<uint32_t>(cf_nodes.size()), 16u);

        // Copy control flow node IDs to fixed-size array
        for (uint32_t i = 0; i < ps.num_control_flow_nodes; ++i) {
            ps.control_flow_node_ids[i] = cf_nodes[i];
        }

        // 4. Precompute predecessor signatures (accelerates runtime computation)
        uint64_t combined_hash = 0;
        for (uint32_t i = 0; i < ps.num_predecessors; ++i) {
            uint32_t pred_id = predecessors[i];
            // Simple hash of predecessor node type and ID
            uint64_t pred_sig = HashNode(circuit_.GetNode(pred_id));
            ps.predecessor_signatures[i] = pred_sig;
            combined_hash ^= pred_sig;
        }

        // 5. Precompute hash components
        ps.level_hash = static_cast<uint64_t>(ps.topological_level) * 0x9e3779b97f4a7c15ULL;
        ps.predecessors_combined_hash = combined_hash;

        ps_dataset[node.id] = ps;
    }

    return ps_dataset;
}

// Extract nodes that affect control flow
std::vector<uint32_t> DAGAnalyzer::ExtractControlFlowNodes(uint32_t node_id) {
    std::vector<uint32_t> cf_nodes;
    std::unordered_set<uint32_t> visited;
    std::queue<uint32_t> queue;

    queue.push(node_id);
    visited.insert(node_id);

    while (!queue.empty()) {
        uint32_t current = queue.front();
        queue.pop();

        auto& node = circuit_.GetNode(current);

        // Check if it affects control flow
        if (node.type == NodeType::MUX ||
            node.type == NodeType::COMPARISON ||
            node.type == NodeType::BRANCH) {
            cf_nodes.push_back(current);
        }

        // Continue tracing predecessors (limit depth to avoid excessive tracing)
        if (cf_nodes.size() < 10) {  // Limit to max 10 control flow nodes
            for (auto pred : circuit_.GetPredecessors(current)) {
                if (visited.find(pred) == visited.end()) {
                    visited.insert(pred);
                    queue.push(pred);
                }
            }
        }
    }

    return cf_nodes;
}

// Node hash function
uint64_t DAGAnalyzer::HashNode(const Node& node) {
    uint64_t hash = 0x123456789ABCDEF0ULL;
    hash ^= static_cast<uint64_t>(node.id);
    hash ^= static_cast<uint64_t>(node.type) << 32;
    return hash;
}

// Get operation type weight
float DAGAnalyzer::GetTypeWeight(NodeType type) {
    switch (type) {
        case NodeType::MUX:        return 2.0f;  // High convergence
        case NodeType::COMPARISON: return 1.5f;  // Binary output
        case NodeType::LOGICAL:    return 1.2f;  // Limited output
        case NodeType::ARITHMETIC: return 1.0f;
        default:                   return 0.5f;
    }
}

} // namespace dart
