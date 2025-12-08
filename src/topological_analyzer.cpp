// src/topological_analyzer.cpp
#include "topological_analyzer.hpp"
#include <queue>
#include <algorithm>
#include <unordered_set>

namespace dart {

TopologicalAnalyzer::TopologicalAnalyzer(const Circuit& circuit)
    : circuit_(circuit) {
}

// Compute topological sort using Kahn's algorithm
std::vector<uint32_t> TopologicalAnalyzer::ComputeTopologicalOrder() {
    std::vector<uint32_t> result;
    std::unordered_map<uint32_t, uint32_t> in_degree;
    std::queue<uint32_t> zero_in_degree;

    // Initialize in-degree for all nodes
    for (const auto& node : circuit_.GetAllNodes()) {
        in_degree[node.id] = circuit_.GetPredecessors(node.id).size();
        if (in_degree[node.id] == 0) {
            zero_in_degree.push(node.id);
        }
    }

    // Process nodes with zero in-degree
    while (!zero_in_degree.empty()) {
        uint32_t current = zero_in_degree.front();
        zero_in_degree.pop();
        result.push_back(current);

        // Reduce in-degree of successors
        for (uint32_t succ : circuit_.GetSuccessors(current)) {
            if (--in_degree[succ] == 0) {
                zero_in_degree.push(succ);
            }
        }
    }

    return result;
}

// Compute topological levels (depth from inputs)
std::unordered_map<uint32_t, uint32_t> TopologicalAnalyzer::ComputeTopologicalLevels() {
    std::unordered_map<uint32_t, uint32_t> levels;
    std::queue<uint32_t> queue;

    // Initialize input nodes at level 0
    for (const auto& node : circuit_.GetInputNodes()) {
        levels[node.id] = 0;
        queue.push(node.id);
    }

    // BFS to propagate levels
    while (!queue.empty()) {
        uint32_t current = queue.front();
        queue.pop();

        for (uint32_t succ : circuit_.GetSuccessors(current)) {
            uint32_t new_level = levels[current] + 1;
            if (levels.find(succ) == levels.end() || levels[succ] < new_level) {
                levels[succ] = new_level;
                queue.push(succ);
            }
        }
    }

    return levels;
}

// Find strongly connected components using Tarjan's algorithm
std::vector<std::vector<uint32_t>> TopologicalAnalyzer::FindStronglyConnectedComponents() {
    std::vector<std::vector<uint32_t>> sccs;
    std::unordered_map<uint32_t, uint32_t> index;
    std::unordered_map<uint32_t, uint32_t> lowlink;
    std::unordered_map<uint32_t, bool> on_stack;
    std::vector<uint32_t> stack;
    uint32_t current_index = 0;

    std::function<void(uint32_t)> strong_connect = [&](uint32_t v) {
        index[v] = current_index;
        lowlink[v] = current_index;
        current_index++;
        stack.push_back(v);
        on_stack[v] = true;

        for (uint32_t w : circuit_.GetSuccessors(v)) {
            if (index.find(w) == index.end()) {
                strong_connect(w);
                lowlink[v] = std::min(lowlink[v], lowlink[w]);
            } else if (on_stack[w]) {
                lowlink[v] = std::min(lowlink[v], index[w]);
            }
        }

        if (lowlink[v] == index[v]) {
            std::vector<uint32_t> scc;
            uint32_t w;
            do {
                w = stack.back();
                stack.pop_back();
                on_stack[w] = false;
                scc.push_back(w);
            } while (w != v);
            sccs.push_back(scc);
        }
    };

    for (const auto& node : circuit_.GetAllNodes()) {
        if (index.find(node.id) == index.end()) {
            strong_connect(node.id);
        }
    }

    return sccs;
}

// Compute reverse topological order
std::vector<uint32_t> TopologicalAnalyzer::ComputeReverseTopologicalOrder() {
    auto forward_order = ComputeTopologicalOrder();
    std::reverse(forward_order.begin(), forward_order.end());
    return forward_order;
}

// Get max topological level
uint32_t TopologicalAnalyzer::GetMaxLevel() {
    auto levels = ComputeTopologicalLevels();
    uint32_t max_level = 0;
    for (const auto& [node_id, level] : levels) {
        max_level = std::max(max_level, level);
    }
    return max_level;
}

// Get nodes at specific level
std::vector<uint32_t> TopologicalAnalyzer::GetNodesAtLevel(uint32_t level) {
    auto levels = ComputeTopologicalLevels();
    std::vector<uint32_t> nodes;

    for (const auto& [node_id, node_level] : levels) {
        if (node_level == level) {
            nodes.push_back(node_id);
        }
    }

    return nodes;
}

} // namespace dart
