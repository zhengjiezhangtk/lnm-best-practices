# LNM Best Practices

Lesion Network Mapping (LNM) 最佳实践 Python 包，包含完整的零模型验证流程。

## 核心功能

- **标准 LNM**: `LNM = sum(M @ C)`
- **症状加权 LNM (sLNM)**: `sLNM = sv' @ (M @ C)`
- **零模型验证**: 空间零模型、拓扑零模型、排列检验
- **统计校正**: Bonferroni、FDR、FWER
- **特异性检验**: 度相关性、随机病灶比较

## 安装

```bash
pip install numpy scipy nibabel nilearn matplotlib brainspace pytest
```

## 快速开始

```python
from lnm_best_practices.core.lnm import LNM
import numpy as np

# 创建数据
M = np.random.rand(20, 100)  # 病灶矩阵
C = np.random.rand(100, 100)  # 连接组
C = (C + C.T) / 2

# 计算 LNM
lnm = LNM(M, C)
result = lnm.compute()
```

## 运行测试

```bash
python -m pytest lnm_best_practices/tests/ -v
```

## 示例

- `examples/basic_lnm.py` - 基础 LNM 计算
- `examples/full_validation.py` - 完整零模型验证
- `examples/clinical_example.py` - 临床分析示例

## 算法来源

| 模块 | 原始实现 |
|------|---------|
| 核心 LNM | `lnm_nulls/lnm_compute.m` (Zalesky et al.) |
| 空间零模型 | `lnm_nulls/lesion_assignment.m` |
| 拓扑零模型 | `lnm_nulls/dcsbm.m` |
| 排列检验 | `lnm_nulls/lnm_compute.m` (FWER 部分) |
| 可视化工具 | `symptom_lnm/neuroimage_analysis.py` (Treeratana et al.) |

## 项目结构

```
lnm_best_practices/
├── core/           # LNM 计算核心
│   ├── lnm.py      # LNM, sLNM, 分布式 LNM
│   └── connectome.py
├── null_models/    # 零模型
│   ├── spatial.py  # 空间随机化
│   ├── topological.py  # dcSBM
│   └── permutation.py  # 排列检验 + FWER
├── statistics/     # 统计方法
│   ├── tests.py    # t 检验、相关、GLM
│   ├── correction.py  # 多重比较校正
│   └── specificity.py # 特异性检验
├── visualization/  # 可视化
├── utils/          # 工具函数
│   └── neuroimage_analysis.py  # 来自 symptom_lnm
├── tests/          # 测试 (23 个)
└── examples/       # 示例脚本
```

## 参考文献

- Zalesky et al. (2026). Null models for lesion network mapping.
- Petersen et al. (2026). Permutation-based inference for LNM.
- Van Den Heuvel et al. (2026). Methodological foundation concerns.
- Siddiqi et al. (2026). Response defending LNM specificity.
- Treeratana et al. (2026). Symptom-based LNM.

## License

MIT
