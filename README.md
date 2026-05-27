# LNM Best Practices

[English](README.md) | [中文](README.zh-CN.md)

A validated Lesion Network Mapping (LNM) pipeline with comprehensive null model validation, addressing methodological concerns from recent literature.

## Features

- **Standard LNM**: `LNM = sum(M @ C)`
- **Symptom-based LNM (sLNM)**: `sLNM = sv' @ (M @ C)`
- **Null Model Validation**: Spatial, topological (dcSBM), and permutation tests
- **Statistical Correction**: Bonferroni, FDR (Benjamini-Hochberg), FWER (max-statistic)
- **Specificity Testing**: Degree correlation, random lesion comparison

## Installation

```bash
pip install numpy scipy nibabel nilearn matplotlib brainspace pytest
```

## Quick Start

```python
from lnm_best_practices.core.lnm import LNM
import numpy as np

# Create data
M = np.random.rand(20, 100)  # Lesion matrix
C = np.random.rand(100, 100)  # Connectome
C = (C + C.T) / 2

# Compute LNM
lnm = LNM(M, C)
result = lnm.compute()
```

## Run Tests

```bash
python -m pytest lnm_best_practices/tests/ -v
```

## Examples

- `examples/basic_lnm.py` - Basic LNM computation
- `examples/full_validation.py` - Full null model validation pipeline
- `examples/clinical_example.py` - Clinical analysis with report generation

## Algorithm Sources

| Module | Original Implementation |
|--------|------------------------|
| Core LNM | `lnm_nulls/lnm_compute.m` (Zalesky et al.) |
| Spatial Null Model | `lnm_nulls/lesion_assignment.m` |
| Topological Null Model | `lnm_nulls/dcsbm.m` |
| Permutation Test | `lnm_nulls/lnm_compute.m` (FWER section) |
| Visualization | `symptom_lnm/neuroimage_analysis.py` (Treeratana et al.) |

## Project Structure

```
lnm_best_practices/
├── core/              # LNM computation
│   ├── lnm.py         # LNM, sLNM, distributional LNM
│   └── connectome.py  # Connectome loading & preprocessing
├── null_models/       # Null model validation
│   ├── spatial.py     # Location randomization
│   ├── topological.py # dcSBM (degree-preserving)
│   └── permutation.py # Symptom permutation + FWER
├── statistics/        # Statistical methods
│   ├── tests.py       # t-tests, correlation, GLM
│   ├── correction.py  # Multiple comparison correction
│   └── specificity.py # Specificity testing
├── visualization/     # Brain surface plotting
├── utils/             # I/O and validation
├── tests/             # 23 unit tests
└── examples/          # Example scripts
```

## References

- Zalesky et al. (2026). Null models for lesion network mapping.
- Petersen et al. (2026). Permutation-based inference for LNM.
- Van Den Heuvel et al. (2026). Methodological foundation concerns.
- Siddiqi et al. (2026). Response defending LNM specificity.
- Treeratana et al. (2026). Symptom-based LNM.

## License

MIT
