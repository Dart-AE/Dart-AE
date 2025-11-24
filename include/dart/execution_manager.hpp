#ifndef DART_EXECUTION_MANAGER_HPP
#define DART_EXECUTION_MANAGER_HPP

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <queue>
#include <cuda_runtime.h>
#include "dart/dart_config.hpp"

namespace dart {

// RuntimePathSignature已在dart_config.hpp中定义，无需重复

// PathSig排序器 - 基于LSH进行刺激分组
class PathSignatureSorter {
public:
    PathSignatureSorter() = default;
    ~PathSignatureSorter() = default;

    // 使用thrust进行GPU排序
    void SortByPathSignature(
        std::vector<RuntimePathSignature>& signatures,
        cudaStream_t stream
    );

    // 基于排序结果重新映射warp
    void RemapWarps(
        const std::vector<RuntimePathSignature>& sorted_sigs,
        uint32_t warp_size,
        std::vector<uint32_t>& new_warp_assignment
    );
};

// Warp重组器 - 核心执行管理模块
class WarpReorganizer {
private:
    PathSignatureSorter sorter_;
    PathSignatureFeatures* d_ps_dataset_;  // GPU上的PS数据集
    uint32_t* d_branch_history_;                  // GPU上的分支历史
    size_t max_stimuli_;
    bool initialized_;

public:
    WarpReorganizer();
    ~WarpReorganizer();

    // 初始化：将PS数据集传输到GPU
    void Initialize(
        const std::unordered_map<uint32_t, PathSignatureFeatures>& ps_dataset,
        size_t max_stimuli = 65536
    );

    // 执行重组
    std::vector<uint32_t> Reorganize(
        const std::vector<uint32_t>& representative_ids,
        uint32_t detection_node_id,
        cudaStream_t stream
    );

    // 获取分支历史缓冲区（供仿真kernel使用）
    uint32_t* GetBranchHistoryBuffer() { return d_branch_history_; }

    // 清空分支历史（新批次开始时）
    void ClearBranchHistory();

    // 检查是否已初始化
    bool IsInitialized() const { return initialized_; }
};

// 批次信息结构
struct BatchInfo {
    uint32_t batch_id;
    uint32_t current_cycle;
    uint32_t num_active_stimuli;
    cudaStream_t stream;
    cudaEvent_t completion_event;
};

// 多批次重叠执行管理器
class BatchOverlapManager {
private:
    std::vector<BatchInfo> active_batches_;
    std::queue<uint32_t> free_sm_ids_;
    uint32_t max_concurrent_batches_;

public:
    explicit BatchOverlapManager(uint32_t max_concurrent_batches = 4);
    ~BatchOverlapManager();

    // 启动新批次
    void LaunchBatch(
        uint32_t batch_id,
        uint32_t num_stimuli,
        void (*kernel_launcher)(uint32_t, cudaStream_t)
    );

    // 等待某个批次完成
    void WaitForBatchCompletion();

    // 等待所有批次完成
    void WaitForAllBatches();

    // 获取当前活跃批次数
    size_t GetActiveBatchCount() const { return active_batches_.size(); }
};

// ROI计算器
class ROICalculator {
public:
    // 计算重组的投资回报率
    static float ComputeROI(
        const AccumulatedMergeState& merges,
        uint32_t current_active,
        uint32_t remaining_cycles,
        float reorganization_overhead_cycles
    );

    // 估算节省的计算量
    static float EstimateSavedComputations(
        const AccumulatedMergeState& merges,
        uint32_t remaining_cycles
    );

    // 估算重组开销
    static float EstimateReorganizationCost(
        uint32_t current_active,
        float overhead_per_stimulus
    );
};

// GPU Kernels 声明

// 计算PathSig签名
__global__ void ComputePathSignaturesKernel(
    const uint32_t* stimulus_ids,
    uint32_t num_stimuli,
    const PathSignatureFeatures* ps_dataset,
    const uint32_t* branch_history,
    uint32_t detection_node_id,
    RuntimePathSignature* output_sigs
);

// 记录分支历史
__device__ void RecordBranchHistory(
    uint32_t stim_id,
    uint32_t node_id,
    bool branch_taken,
    uint32_t* branch_history
);

// 压缩代表刺激
__global__ void CompactRepresentativesKernel(
    const uint32_t* match_table,
    const uint64_t* path_signatures,
    uint32_t num_stimuli,
    uint32_t* compacted_indices,
    uint32_t* compacted_count
);

// 应用warp分配
__global__ void ApplyWarpAssignmentKernel(
    const uint32_t* old_indices,
    const uint32_t* new_assignment,
    uint32_t num_stimuli,
    void* state_data,
    size_t state_size,
    void* output_state_data
);

// 重建follower状态
__global__ void ReconstructFollowerStatesKernel(
    const uint32_t* representative_map,
    uint32_t num_stimuli,
    const void* representative_states,
    void* all_states,
    size_t state_size
);

} // namespace dart

#endif // DART_EXECUTION_MANAGER_HPP
