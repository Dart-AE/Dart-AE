// src/critical_path.cpp
#include "critical_path.hpp"
#include <queue>
#include <algorithm>
#include <limits>

namespace dart {

CriticalPathAnalyzer::CriticalPathAnalyzer(const Circuit& circuit)
    : circuit_(circuit) {
}

// Compute critical path delays using longest path algorithm
std::unordered_map<uint32_t, float> CriticalPathAnalyzer::ComputeNodeDelays() {
    std::unordered_map<uint32_t, float> delays;
    std::queue<uint32_t> queue;

    // Initialize input nodes with zero delay
    for (const auto& node : circuit_.GetInputNodes()) {
        delays[node.id] = 0.0f;
        queue.push(node.id);
    }

    // Propagate delays through the circuit
    while (!queue.empty()) {
        uint32_t current = queue.front();
        queue.pop();

        float current_delay = delays[current];
        float node_delay = GetNodeDelay(circuit_.GetNode(current));

        for (uint32_t succ : circuit_.GetSuccessors(current)) {
            float new_delay = current_delay + node_delay;

            if (delays.find(succ) == delays.end() || delays[succ] < new_delay) {
                delays[succ] = new_delay;
                queue.push(succ);
            }
        }
    }

    return delays;
}

// Find critical path from inputs to outputs
std::vector<uint32_t> CriticalPathAnalyzer::FindCriticalPath() {
    auto delays = ComputeNodeDelays();

    // Find output node with maximum delay
    float max_delay = 0.0f;
    uint32_t critical_output = 0;

    for (const auto& node : circuit_.GetOutputNodes()) {
        if (delays[node.id] > max_delay) {
            max_delay = delays[node.id];
            critical_output = node.id;
        }
    }

    // Backtrack to construct critical path
    std::vector<uint32_t> critical_path;
    uint32_t current = critical_output;

    while (true) {
        critical_path.push_back(current);

        // Find predecessor on critical path
        auto predecessors = circuit_.GetPredecessors(current);
        if (predecessors.empty()) break;

        uint32_t critical_pred = predecessors[0];
        float max_pred_delay = delays[predecessors[0]];

        for (uint32_t pred : predecessors) {
            if (delays[pred] > max_pred_delay) {
                max_pred_delay = delays[pred];
                critical_pred = pred;
            }
        }

        current = critical_pred;
    }

    std::reverse(critical_path.begin(), critical_path.end());
    return critical_path;
}

// Identify critical registers on critical paths
std::vector<uint32_t> CriticalPathAnalyzer::IdentifyCriticalRegisters() {
    auto critical_path = FindCriticalPath();
    std::vector<uint32_t> critical_regs;

    for (uint32_t node_id : critical_path) {
        const auto& node = circuit_.GetNode(node_id);
        if (node.type == NodeType::REGISTER) {
            critical_regs.push_back(node_id);
        }
    }

    return critical_regs;
}

// Compute slack for each node
std::unordered_map<uint32_t, float> CriticalPathAnalyzer::ComputeSlacks() {
    auto early_times = ComputeNodeDelays();
    std::unordered_map<uint32_t, float> late_times;
    std::unordered_map<uint32_t, float> slacks;

    // Find maximum delay (circuit delay)
    float circuit_delay = 0.0f;
    for (const auto& [node_id, delay] : early_times) {
        circuit_delay = std::max(circuit_delay, delay);
    }

    // Initialize late times for outputs
    for (const auto& node : circuit_.GetOutputNodes()) {
        late_times[node.id] = circuit_delay;
    }

    // Backpropagate late times
    auto reverse_order = circuit_.GetReverseTopologicalOrder();
    for (uint32_t node_id : reverse_order) {
        if (late_times.find(node_id) == late_times.end()) {
            late_times[node_id] = std::numeric_limits<float>::max();

            for (uint32_t succ : circuit_.GetSuccessors(node_id)) {
                float succ_late = late_times[succ];
                float node_delay = GetNodeDelay(circuit_.GetNode(node_id));
                late_times[node_id] = std::min(late_times[node_id],
                                                succ_late - node_delay);
            }
        }
    }

    // Compute slacks
    for (const auto& [node_id, early] : early_times) {
        slacks[node_id] = late_times[node_id] - early;
    }

    return slacks;
}

// Find all critical paths (zero slack paths)
std::vector<std::vector<uint32_t>> CriticalPathAnalyzer::FindAllCriticalPaths() {
    auto slacks = ComputeSlacks();
    std::vector<std::vector<uint32_t>> critical_paths;

    // DFS to find all zero-slack paths
    std::function<void(uint32_t, std::vector<uint32_t>&)> dfs =
        [&](uint32_t node_id, std::vector<uint32_t>& current_path) {

        current_path.push_back(node_id);

        // Check if this is an output node
        bool is_output = false;
        for (const auto& output : circuit_.GetOutputNodes()) {
            if (output.id == node_id) {
                is_output = true;
                break;
            }
        }

        if (is_output) {
            critical_paths.push_back(current_path);
        } else {
            // Continue DFS on zero-slack successors
            for (uint32_t succ : circuit_.GetSuccessors(node_id)) {
                if (std::abs(slacks[succ]) < 1e-6f) {
                    dfs(succ, current_path);
                }
            }
        }

        current_path.pop_back();
    };

    // Start DFS from zero-slack input nodes
    for (const auto& input : circuit_.GetInputNodes()) {
        if (std::abs(slacks[input.id]) < 1e-6f) {
            std::vector<uint32_t> path;
            dfs(input.id, path);
        }
    }

    return critical_paths;
}

// Get delay estimate for a node type
float CriticalPathAnalyzer::GetNodeDelay(const Node& node) {
    switch (node.type) {
        case NodeType::REGISTER:    return 0.5f;  // Clock-to-Q + Setup
        case NodeType::MUX:         return 0.3f;
        case NodeType::COMPARISON:  return 0.4f;
        case NodeType::ARITHMETIC:  return 0.6f;  // Adder/multiplier
        case NodeType::LOGICAL:     return 0.2f;  // AND/OR/XOR
        case NodeType::WIRE:        return 0.1f;
        default:                    return 0.1f;
    }
}

} // namespace dart
