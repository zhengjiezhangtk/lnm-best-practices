# LNM 最佳实践

[English](README.md) | [中文](README.zh-CN.md)

标准化的病灶网络映射（Lesion Network Mapping, LNM）Python 流水线，包含完整的零模型验证，回应近期文献中的方法学质疑。

## 核心功能

- **标准 LNM**: `LNM = sum(M @ C)`
- **症状加权 LNM (sLNM)**: `sLNM = sv' @ (M @ C)`
- **零模型验证**: 空间零模型、拓扑零模型（dcSBM）、排列检验
- **统计校正**: Bonferroni、FDR（Benjamini-Hochberg）、FWER（最大统计量法）
- **特异性检验**: 度相关性分析、随机病灶比较

## 安装

```bash
pip install numpy scipy nibabel nilearn matplotlib brainspace pytest
```

## 快速开始

```python
from lnm_best_practices.core.lnm import LNM
import numpy as np

# 创建数据
M = np.random.rand(20, 100)  # 病灶矩阵 (20 个病人, 100 个脑区)
C = np.random.rand(100, 100)  # 连接组矩阵
C = (C + C.T) / 2             # 对称化

# 计算 LNM
lnm = LNM(M, C)
result = lnm.compute()

# 计算症状加权 LNM
symptoms = np.random.randn(20)
slnm_result = lnm.compute_slnm(symptoms)
```

## 运行测试

```bash
python -m pytest lnm_best_practices/tests/ -v
```

23 个单元测试覆盖：
- 核心 LNM 计算（公式正确性、维度检查、归一化）
- 零模型（空间随机化、度保持、排列检验）
- 统计方法（t 检验、相关、GLM、多重比较校正）

## 示例脚本

| 脚本 | 说明 |
|------|------|
| `examples/basic_lnm.py` | 基础 LNM 和 sLNM 计算 |
| `examples/full_validation.py` | 完整零模型验证流程 |
| `examples/clinical_example.py` | 临床分析示例（含报告生成） |

## 算法来源

本包中的算法忠实复现了以下仓库的 MATLAB 实现：

| 模块 | 原始实现 | 来源 |
|------|---------|------|
| 核心 LNM | `lnm_compute.m` | Zalesky et al. (`lnm_nulls`) |
| 空间零模型 | `lesion_assignment.m` | Zalesky et al. (`lnm_nulls`) |
| 拓扑零模型 | `dcsbm.m` | Zalesky et al. (`lnm_nulls`) |
| 排列检验 + FWER | `lnm_compute.m` | Zalesky et al. (`lnm_nulls`) |
| 可视化工具 | `neuroimage_analysis.py` | Treeratana et al. (`symptom_lnm`) |

## 项目结构

```
lnm_best_practices/
├── core/              # LNM 计算核心
│   ├── lnm.py         # 标准 LNM、sLNM、分布式 LNM
│   └── connectome.py  # 连接组加载与预处理
├── null_models/       # 零模型验证
│   ├── spatial.py     # 病灶位置随机化（3 种方案）
│   ├── topological.py # dcSBM 拓扑零模型
│   └── permutation.py # 症状排列 + FWER 校正
├── statistics/        # 统计方法
│   ├── tests.py       # t 检验、Pearson 相关、GLM
│   ├── correction.py  # 多重比较校正（Bonferroni/FDR/FWER/TFCE）
│   └── specificity.py # 特异性检验
├── visualization/     # 脑表面可视化
├── utils/             # I/O 工具与输入验证
│   └── neuroimage_analysis.py  # 来自 symptom_lnm 仓库
├── tests/             # 23 个单元测试
└── examples/          # 示例脚本
```

## 背景

近期 Van Den Heuvel et al. (2026) 对 LNM 的方法学基础提出质疑，认为 LNM 结果可能由连接组的度分布驱动，而非真实的病灶-网络关系。本包实现了多种零模型来验证 LNM 结果的特异性：

1. **空间零模型**：随机化病灶位置，保持病灶大小不变
2. **拓扑零模型**：生成保持度分布的随机网络（dcSBM）
3. **排列检验**：置换症状标签，使用最大统计量法进行 FWER 校正
4. **特异性检验**：检验 LNM 是否与度分布相关

## 参考文献

- Zalesky et al. (2026). Null models for lesion network mapping.
- Petersen et al. (2026). Permutation-based inference for LNM.
- Van Den Heuvel et al. (2026). Methodological foundation concerns.
- Siddiqi et al. (2026). Response defending LNM specificity.
- Treeratana et al. (2026). Symptom-based LNM.
- Meng et al. (2026). Response on LNM methodology.
- Edelman et al. (2026). Methodological considerations for LNM.
- Wawrzyniak et al. (2026). Spatial bias in LNM.

## 许可证

MIT
