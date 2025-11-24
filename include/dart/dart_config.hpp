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

// DART可调参数配置
struct DARTConfig {
    // === 检测策略参数 ===
    uint32_t detection_interval = 100;          // 检测间隔（周期数）
                                                // 对于500K周期，每100周期检测一次 = 5000次检测
                                                // 可调范围：50-500

    uint32_t initial_detection_cycle = 50;      // 首次检测周期（让状态先收敛）

    bool adaptive_detection = true;             // 自适应检测间隔
    uint32_t min_detection_interval = 50;       // 最小检测间隔
    uint32_t max_detection_interval = 500;      // 最大检测间隔

    uint32_t num_detection_nodes = 3;           // 使用的检测节点数量（1-5）

    // === 合并策略参数 ===
    uint32_t min_merge_count = 16;              // 触发重组的最小合并数
                                                // 需要累积足够的合并才值得重组

    float merge_accumulation_threshold = 0.3f;  // 累积合并率阈值
                                                // 当前活跃刺激的30%被合并时考虑重组

    // === Warp重组参数 ===
    uint32_t warp_size = 32;
    uint32_t min_warp_utilization = 16;         // 最小warp利用率

    float roi_threshold = 2.0f;                  // ROI阈值，触发重组的最小ROI
                                                // 2.0表示收益至少是开销的2倍
                                                // 可调范围：1.5-3.0

    float reorganization_overhead_cycles = 50.0f; // 重组开销（等效周期数）
                                                 // 用于ROI计算

    bool enable_path_sorting = true;            // 是否启用PathSig排序
    bool track_branch_history = true;           // 是否追踪分支历史

    // === 执行策略参数 ===
    bool immediate_masking = true;              // 立即mask followers（不等重组）
    bool enable_batch_overlap = true;           // 启用批次重叠执行

    // === 调试参数 ===
    bool verbose = false;                       // 详细输出
    bool debug_fingerprint = false;             // 调试指纹匹配
    bool debug_reorganization = false;          // 调试重组过程
    bool print_detection_stats = false;         // 打印每次检测的统计

    // 从命令行参数加载
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

    // 打印配置
    void Print() const {
        std::cout << "\n=== DART配置 ===\n";
        std::cout << "检测间隔: " << detection_interval << " 周期\n";
        std::cout << "ROI阈值: " << roi_threshold << "\n";
        std::cout << "检测节点数: " << num_detection_nodes << "\n";
        std::cout << "最小合并数: " << min_merge_count << "\n";
        std::cout << "立即masking: " << (immediate_masking ? "启用" : "禁用") << "\n";
        std::cout << "自适应检测: " << (adaptive_detection ? "启用" : "禁用") << "\n";
        std::cout << "PathSig排序: " << (enable_path_sorting ? "启用" : "禁用") << "\n";
    }
};

// DAG节点元数据（§4.1）
struct DAGNodeMetadata {
    uint32_t node_id;
    uint32_t topological_level;     // 拓扑层级
    float    convergence_score;      // 收敛得分 Score(v)
    uint32_t fanout;
    uint32_t critical_reg_count;
    std::vector<uint32_t> critical_regs;  // R_critical
};

// 路径签名特征集 (PS) - 编译时预计算（§4.1 & §4.3）
struct PathSignatureFeatures {
    uint32_t node_id;                           // 节点ID

    // P(v): 前驱节点签名
    uint32_t num_predecessors;
    uint64_t predecessor_signatures[32];        // 固定大小数组便于GPU访问

    // L(v): 拓扑层级标识符
    uint32_t topological_level;

    // 控制流节点（用于H(si)的构建）
    uint32_t num_control_flow_nodes;
    uint32_t control_flow_node_ids[16];         // 影响控制流的节点

    // 预计算的哈希组件（优化）
    uint64_t level_hash;                        // 层级的预哈希值
    uint64_t predecessors_combined_hash;        // 所有前驱的组合哈希
};

// 运行时路径签名（§4.3 Equation 4）
struct RuntimePathSignature {
    uint64_t signature;         // PathSig(v, si) = Hash(P(v), L(v), H(si))
    uint32_t stimulus_id;       // 刺激ID
    uint32_t detection_node;    // 检测节点ID
    uint32_t original_warp_id;  // 原始warp ID（用于重组前后对比）
};

// 分支历史记录 H(si) - 运行时动态构建
struct BranchHistory {
    uint32_t stimulus_id;
    uint32_t history_bits;      // 位向量，每位记录一个分支选择
    uint32_t num_branches;      // 记录的分支数量

    // 可选：详细的分支记录（调试用）
    struct BranchRecord {
        uint32_t node_id;
        uint32_t cycle;
        bool taken;
    };
    std::vector<BranchRecord> detailed_history;  // 仅在调试模式启用
};

// 检测节点配置
struct DetectionNode {
    uint32_t node_id;
    float    score;                  // 综合评分
    uint32_t level;
    std::vector<uint32_t> input_ports;
    std::vector<uint32_t> critical_inputs;  // 筛选后的关键输入
};

// 指纹结构（§4.2）
struct Fingerprint {
    uint128_t hash;                  // 128位指纹
    uint32_t  detection_node;
    uint32_t  stimulus_id;
};

// 刺激合并记录
struct MergeRecord {
    uint32_t representative_id;      // 代表刺激ID
    std::vector<uint32_t> follower_ids;  // 追随者ID列表
    uint32_t merge_cycle;            // 合并发生的周期
    uint32_t detection_node;
    uint64_t path_signature;         // 代表的路径签名
};

// 累积合并状态（在多次检测间累积）
struct AccumulatedMergeState {
    std::unordered_map<uint32_t, uint32_t> representative_map; // follower -> representative
    std::unordered_map<uint32_t, std::vector<uint32_t>> follower_groups; // representative -> followers

    uint32_t total_merged = 0;       // 累积合并的刺激数
    uint32_t num_representatives = 0; // 当前代表数量

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

// DART运行时统计
struct DARTStats {
    uint64_t total_stimuli;
    uint64_t merged_stimuli;
    uint64_t reorganization_count;
    uint64_t detection_count;               // 总检测次数
    uint64_t immediate_masked_count;        // 立即masked的follower数

    double   detection_time_ms;
    double   merging_time_ms;
    double   reorganization_time_ms;
    double   path_sig_computation_ms;       // PathSig计算时间
    double   sorting_time_ms;               // 排序时间
    double   masking_time_ms;               // Masking时间
    double   total_time_ms;

    // PathSig相关统计
    uint32_t num_path_sig_groups;           // LSH分组数量
    float    avg_group_size;                // 平均每组大小

    // 检测间隔自适应统计
    std::vector<uint32_t> detection_intervals_history;  // 历史检测间隔
    std::vector<float> redundancy_history;              // 历史冗余率

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
