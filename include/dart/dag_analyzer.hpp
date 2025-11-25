// include/dart/dag_analyzer.hpp
#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace dart {

// Forward declarations
struct DAGNodeMetadata;
struct DetectionNode;
struct PathSignatureFeatures;
struct uint128_t;

// Node type enumeration
enum class NodeType {
    INPUT,
    OUTPUT,
    REGISTER,
    OPERATION,
    MUX,
    COMPARISON,
    LOGICAL,
    ARITHMETIC,
    BRANCH,
    UNKNOWN
};

// Node structure
struct Node {
    uint32_t id;
    NodeType type;
    bool affects_control_flow;
};

// Circuit interface (simplified version, actual implementation may be elsewhere)
class Circuit {
public:
    virtual ~Circuit() = default;
    virtual const std::vector<Node>& GetInputNodes() const = 0;
    virtual const std::vector<Node>& GetAllNodes() const = 0;
    virtual std::vector<uint32_t> GetSuccessors(uint32_t node_id) const = 0;
    virtual std::vector<uint32_t> GetPredecessors(uint32_t node_id) const = 0;
    virtual const Node& GetNode(uint32_t node_id) const = 0;
    virtual NodeType GetNodeType(uint32_t node_id) const = 0;
    virtual uint32_t GetFanout(uint32_t node_id) const = 0;
    virtual uint32_t GetMaxFanout() const = 0;
    virtual uint32_t GetMaxLevel() const = 0;
};

// DAG analyzer class
class DAGAnalyzer {
private:
    const Circuit& circuit_;
    std::vector<DAGNodeMetadata> node_metadata_;

    // Helper functions
    float GetTypeWeight(NodeType type);
    uint64_t HashNode(const Node& node);

public:
    explicit DAGAnalyzer(const Circuit& circuit);

    // Topological level computation (L(v) in Equation 2)
    void ComputeTopologicalLevels();

    // Compute detection node score (Equation 1)
    float ComputeScore(uint32_t node_id);

    // Select optimal detection nodes
    std::vector<DetectionNode> SelectDetectionNodes(size_t count = 3);

    // Critical register identification (backward dataflow tracing)
    std::vector<uint32_t> ExtractCriticalRegisters(uint32_t node_id);

    // Path signature feature set (PS) precomputation - built at compile time
    std::unordered_map<uint32_t, PathSignatureFeatures> BuildPathSignatureDataset();

    // Extract nodes affecting control flow
    std::vector<uint32_t> ExtractControlFlowNodes(uint32_t node_id);
};

} // namespace dart
