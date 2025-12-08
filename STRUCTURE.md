# DART-AE 项目结构说明

## 项目概述

DART (DAG-driven Redundant Testbench Elimination) 是一个基于 GPU 的 RTL 仿真加速框架，通过 DAG 导向的冗余消除机制实现高效的批量测试向量执行。

## 目录结构

```
Dart-AE/
├── src/                          # 源代码目录（直接包含所有源文件，无二级目录）
│   ├── dag_analyzer.cpp          # DAG 分析器实现
│   ├── topological_analyzer.cpp  # 拓扑排序和层级分析
│   ├── critical_path.cpp         # 关键路径提取
│   ├── fingerprint.cu            # 指纹计算和匹配（CUDA）
│   ├── stimulus_merger.cu        # 刺激合并引擎（CUDA）
│   ├── hash_table.cu             # GPU 哈希表实现（CUDA）
│   ├── warp_reorganizer.cu       # Warp 重组管理（CUDA）
│   ├── batch_overlapper.cu       # 批次重叠执行（CUDA）
│   ├── state_reconstructor.cu    # 状态重建器（CUDA）
│   ├── runtime_manager.cpp       # DART 运行时管理器
│   └── utils.cpp                 # 工具函数和辅助类
│
├── include/                      # 头文件目录（直接包含所有头文件，无二级目录）
│   ├── circuit.hpp               # 电路表示和数据结构
│   ├── dart_config.hpp           # DART 核心配置和数据结构
│   ├── dag_analyzer.hpp          # DAG 分析器接口
│   ├── topological_analyzer.hpp  # 拓扑分析器接口
│   ├── critical_path.hpp         # 关键路径分析器接口
│   ├── fingerprint_matcher.hpp   # 指纹匹配器接口
│   ├── stimulus_merger.hpp       # 刺激合并器接口
│   ├── hash_table.hpp            # GPU 哈希表接口
│   ├── execution_manager.hpp     # 执行管理器接口
│   ├── batch_overlapper.hpp      # 批次重叠管理器接口
│   ├── state_reconstructor.hpp   # 状态重建器接口
│   ├── runtime_manager.hpp       # DART 运行时管理器接口
│   └── utils.hpp                 # 工具函数接口
│
├── benchmarks/                   # 性能测试程序
│   └── dart_benchmark.cu         # DART 基准测试
│
├── examples/                     # 示例程序
│   └── dart_example.cpp          # DART 使用示例
│
├── tests/                        # 测试程序
│
├── scripts/                      # 构建和运行脚本
│
├── traces/                       # 测试刺激和trace文件
│
├── hw_small/                     # 硬件设计文件（NVDLA等）
│
├── CMakeLists.txt               # CMake 构建配置
├── README.md                     # 项目说明
├── LICENSE                       # 许可证
└── VERSION                       # 版本信息

## 核心模块说明

### 1. DAG 分析模块 (§4.1)
- **dag_analyzer.cpp/hpp**: 主 DAG 分析器，实现拓扑层级计算、检测节点选择、关键寄存器识别
- **topological_analyzer.cpp/hpp**: 拓扑排序算法（Kahn算法、Tarjan算法）
- **critical_path.cpp/hpp**: 关键路径提取和时序分析

**功能**:
- 计算节点拓扑层级 L(v)
- 基于 Equation 1 的检测节点评分
- 提取关键寄存器集合 R_critical
- 构建路径签名特征集 (PS)

### 2. 指纹匹配引擎 (§4.2)
- **fingerprint.cu/fingerprint_matcher.hpp**: 128位指纹计算和GPU并行匹配
- **stimulus_merger.cu/stimulus_merger.hpp**: 刺激合并和代表选择
- **hash_table.cu/hash_table.hpp**: GPU加速的哈希表实现

**功能**:
- 基于关键寄存器的指纹计算
- Warp内shuffle优化的快速匹配
- 精确验证和immediate masking
- 刺激分组和合并记录

### 3. 执行管理模块 (§4.3)
- **warp_reorganizer.cu/execution_manager.hpp**: PathSig计算和Warp重组
- **batch_overlapper.cu/batch_overlapper.hpp**: 批次重叠执行管理
- **state_reconstructor.cu/state_reconstructor.hpp**: 追随者状态重建

**功能**:
- 运行时 PathSig(v, si) 计算 (Equation 4)
- 基于PathSig的Warp重组和排序
- 多批次流水线重叠执行
- 状态压缩和重建验证

### 4. 运行时管理
- **runtime_manager.cpp/hpp**: DART完整运行时系统
- **utils.cpp/hpp**: 工具函数、计时器、内存跟踪

**功能**:
- 统一的DART执行流程管理
- 性能统计和profiling
- CUDA资源管理

## 文件统计

- **源文件总数**: 11 个
  - C++ 源文件: 4 个
  - CUDA 源文件: 7 个

- **头文件总数**: 13 个

## 构建说明

### 环境要求
- CUDA 11.6+
- GCC 8+
- CMake 3.18+
- CUDA架构: sm_86 (Ampere) 或更高

### 编译步骤
```bash
cd Dart-AE
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_CUDA_ARCHITECTURES=86 \
         -DCMAKE_CXX_FLAGS="-O2" \
         -DCMAKE_CUDA_FLAGS="-O2"
make -j$(nproc)
```

### 编译选项
- `NUM_TESTBENCHES`: 最大测试向量数（默认1024）
- `GPU_THREADS`: GPU线程数（默认256）
- `BUILD_EXAMPLES`: 构建示例程序（默认ON）
- `BUILD_TESTS`: 构建测试程序（默认ON）
- `BUILD_BENCHMARKS`: 构建基准测试（默认ON）

## 关键设计决策

### 1. 目录扁平化
按照指南要求，移除了 `src/dart` 和 `include/dart` 二级目录，所有文件直接放在 `src/` 和 `include/` 根目录下，便于管理和编译。

### 2. 模块化设计
每个核心功能模块都有独立的源文件和头文件，职责明确：
- DAG分析模块：编译时静态分析
- 指纹匹配引擎：运行时动态检测
- 执行管理模块：Warp重组和批次管理
- 运行时管理：统一接口和资源管理

### 3. GPU优化策略
- 使用 CUDA streams 实现批次重叠
- Warp shuffle 优化的指纹匹配
- Thrust库实现高效排序
- 原子操作实现线程安全的哈希表

## 论文对应关系

| 论文章节 | 实现文件 | 功能描述 |
|---------|---------|---------|
| §4.1 DAG-driven Analysis | dag_analyzer.cpp, topological_analyzer.cpp, critical_path.cpp | DAG分析、检测节点选择、关键寄存器提取 |
| §4.2 Fingerprint Matching | fingerprint.cu, stimulus_merger.cu, hash_table.cu | 指纹计算、匹配和刺激合并 |
| §4.3 Warp Reorganization | warp_reorganizer.cu, batch_overlapper.cu | PathSig计算、Warp重组、批次重叠 |
| §4.4 State Reconstruction | state_reconstructor.cu | 追随者状态重建和验证 |
| Equation 1 | dag_analyzer.cpp:ComputeScore() | 检测节点评分公式 |
| Equation 2 | dag_analyzer.cpp:ComputeTopologicalLevels() | 拓扑层级权重 |
| Equation 4 | warp_reorganizer.cu:ComputePathSignaturesKernel() | PathSig计算公式 |

## 下一步工作

1. 实现具体的 RTL 解析器（circuit.hpp 中的占位符）
2. 集成 RTLflow 基线系统
3. 添加完整的测试用例
4. 性能调优和profiling
5. 编写用户文档和API文档

## 性能指标（目标）

根据论文 DART 应该实现：
- 相比 CPU 基线: 89.4x - 170.9x 加速
- 相比 RTLflow GPU: 1.76x - 3.04x 额外加速
- 冗余率: 40% - 60%
- 检测准确率: >95%

## 联系方式

项目基于 ASPLOS'24 论文 "DART: A Scalable GPU-Accelerated RTL Simulator with DAG-driven Redundant Testbench Elimination" 实现。
