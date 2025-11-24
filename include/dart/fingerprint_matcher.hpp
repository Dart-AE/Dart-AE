#ifndef DART_FINGERPRINT_MATCHER_HPP
#define DART_FINGERPRINT_MATCHER_HPP

#include <vector>
#include <cstdint>
#include <cuda_runtime.h>
#include "dart/dart_config.hpp"

namespace dart {

// uint128_t 和 Fingerprint 已在 dart_config.hpp 中定义

// CUDA Kernel函数声明
__device__ uint128_t ComputeFingerprint(
    const uint32_t* state_vector,
    const uint32_t* critical_regs,
    uint32_t num_critical_regs,
    uint32_t detection_node_id
);

__global__ void FingerprintMatchingKernel(
    const uint32_t* state_vectors,      // [num_stimuli * state_size]
    const uint32_t* critical_reg_indices, // [num_critical_regs]
    uint32_t num_critical_regs,
    uint32_t state_size,
    uint32_t num_stimuli,
    uint32_t detection_node_id,
    uint128_t* fingerprints,            // 输出：每个刺激的指纹
    uint32_t* match_table,              // 输出：匹配表 [num_stimuli]
    uint32_t* merge_count               // 输出：合并计数
);

__global__ void ValidateCriticalRegistersKernel(
    const uint32_t* state_vectors,
    const uint32_t* critical_reg_indices,
    uint32_t num_critical_regs,
    uint32_t state_size,
    const uint32_t* match_table,
    uint32_t num_stimuli,
    bool* validation_results
);

__global__ void ImmediateMaskFollowersKernel(
    const uint32_t* match_table,
    uint32_t num_stimuli,
    bool* thread_active_mask,           // 输出：线程激活掩码
    uint32_t* num_active_remaining      // 输出：剩余活跃线程数
);

__global__ void MaskedSimulationKernel(
    const uint32_t* stimulus_ids,
    const bool* thread_active_mask,
    uint32_t num_stimuli
    /* ... 其他仿真参数 ... */
);

__global__ void ReconstructFollowerStatesKernel(
    const uint32_t* match_table,
    const uint32_t* representative_states,  // 代表的状态
    uint32_t* all_states,                   // 所有刺激的状态（输出）
    uint32_t state_size,
    uint32_t num_stimuli
);

// 主机端包装类
class FingerprintMatcher {
private:
    uint128_t* d_fingerprints_;
    uint32_t* d_match_table_;
    bool* d_thread_active_mask_;
    bool* d_validation_results_;

public:
    // 匹配结果结构体
    struct MatchResult {
        std::vector<uint32_t> representatives;
        std::vector<uint32_t> match_table;
        uint32_t num_merged;
        uint32_t num_active_remaining;
    };

    // 初始化
    void Initialize(uint32_t max_stimuli);

    // 执行匹配并立即mask
    MatchResult MatchAndMask(
        const uint32_t* d_state_vectors,
        uint32_t num_stimuli,
        uint32_t state_size,
        const std::vector<uint32_t>& critical_regs,
        uint32_t detection_node_id,
        bool enable_immediate_masking,
        cudaStream_t stream
    );

    // 获取线程激活掩码（供后续kernel使用）
    bool* GetThreadActiveMask();

    // 析构函数
    ~FingerprintMatcher();
};

} // namespace dart

#endif // DART_FINGERPRINT_MATCHER_HPP
