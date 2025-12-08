// include/circuit.hpp
#pragma once

#include <vector>
#include <string>
#include <cstdint>

namespace dart {

// Node types in the circuit
enum class NodeType {
    INPUT,
    OUTPUT,
    REGISTER,
    WIRE,
    MUX,
    COMPARISON,
    ARITHMETIC,
    LOGICAL,
    BRANCH,
    OPERATION
};

// Circuit node structure
struct Node {
    uint32_t id;
    NodeType type;
    std::string name;
    bool affects_control_flow;

    Node() : id(0), type(NodeType::WIRE), affects_control_flow(false) {}
};

// Circuit representation (placeholder - should be implemented based on actual RTL parser)
class Circuit {
public:
    Circuit() = default;

    // Add a node to the circuit
    void AddNode(const Node& node) {
        nodes_.push_back(node);
    }

    // Add an edge between two nodes
    void AddEdge(uint32_t from, uint32_t to) {
        successors_[from].push_back(to);
        predecessors_[to].push_back(from);
    }

    // Get all nodes
    const std::vector<Node>& GetAllNodes() const {
        return nodes_;
    }

    // Get input nodes
    std::vector<Node> GetInputNodes() const {
        std::vector<Node> inputs;
        for (const auto& node : nodes_) {
            if (node.type == NodeType::INPUT) {
                inputs.push_back(node);
            }
        }
        return inputs;
    }

    // Get output nodes
    std::vector<Node> GetOutputNodes() const {
        std::vector<Node> outputs;
        for (const auto& node : nodes_) {
            if (node.type == NodeType::OUTPUT) {
                outputs.push_back(node);
            }
        }
        return outputs;
    }

    // Get successors of a node
    std::vector<uint32_t> GetSuccessors(uint32_t node_id) const {
        auto it = successors_.find(node_id);
        return (it != successors_.end()) ? it->second : std::vector<uint32_t>();
    }

    // Get predecessors of a node
    std::vector<uint32_t> GetPredecessors(uint32_t node_id) const {
        auto it = predecessors_.find(node_id);
        return (it != predecessors_.end()) ? it->second : std::vector<uint32_t>();
    }

    // Get a specific node
    const Node& GetNode(uint32_t node_id) const {
        for (const auto& node : nodes_) {
            if (node.id == node_id) {
                return node;
            }
        }
        static Node dummy;
        return dummy;
    }

    // Get node type
    NodeType GetNodeType(uint32_t node_id) const {
        return GetNode(node_id).type;
    }

    // Get fanout of a node
    uint32_t GetFanout(uint32_t node_id) const {
        return GetSuccessors(node_id).size();
    }

    // Get maximum fanout in the circuit
    uint32_t GetMaxFanout() const {
        uint32_t max_fanout = 0;
        for (const auto& node : nodes_) {
            max_fanout = std::max(max_fanout, GetFanout(node.id));
        }
        return max_fanout;
    }

    // Get maximum topological level
    uint32_t GetMaxLevel() const {
        // Placeholder - should be computed
        return 100;
    }

    // Get reverse topological order
    std::vector<uint32_t> GetReverseTopologicalOrder() const {
        // Placeholder - should be implemented
        std::vector<uint32_t> order;
        for (const auto& node : nodes_) {
            order.push_back(node.id);
        }
        return order;
    }

private:
    std::vector<Node> nodes_;
    std::unordered_map<uint32_t, std::vector<uint32_t>> successors_;
    std::unordered_map<uint32_t, std::vector<uint32_t>> predecessors_;
};

} // namespace dart
