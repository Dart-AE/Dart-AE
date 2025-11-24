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

// 节点类型枚举
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

// 节点结构
struct Node {
    uint32_t id;
    NodeType type;
    bool affects_control_flow;
};

// 电路接口（简化版，实际实现可能在其他地方）
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

// DAG分析器类
class DAGAnalyzer {
private:
    const Circuit& circuit_;
    std::vector<DAGNodeMetadata> node_metadata_;

    // 辅助函数
    float GetTypeWeight(NodeType type);
    uint64_t HashNode(const Node& node);

public:
    explicit DAGAnalyzer(const Circuit& circuit);

    // 拓扑层级计算（Equation 2中的L(v)）
    void ComputeTopologicalLevels();

    // 计算检测节点得分（Equation 1）
    float ComputeScore(uint32_t node_id);

    // 选择最优检测节点
    std::vector<DetectionNode> SelectDetectionNodes(size_t count = 3);

    // 关键寄存器识别（反向数据流追踪）
    std::vector<uint32_t> ExtractCriticalRegisters(uint32_t node_id);

    // 路径签名特征集(PS)预计算 - 编译时构建
    std::unordered_map<uint32_t, PathSignatureFeatures> BuildPathSignatureDataset();

    // 提取影响控制流的节点
    std::vector<uint32_t> ExtractControlFlowNodes(uint32_t node_id);
};

} // namespace dart
