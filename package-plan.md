# ITH Toolkit Python 包封装规划文档

## 1. 设计目标

将现有的 `src/` 目录下的松散模块封装为一个可安装、可分发、文档完善的 Python 包，实现以下目标：

- **可安装**：`pip install ith-toolkit` 一键安装
- **可复用**：通过 `import ith` 使用所有功能
- **CLI 完备**：提供命令行入口 `ith-score`、`ith-preprocess` 等
- **向后兼容**：保留现有 `src/` 引用方式（软链接或 shim）
- **版本管理**：遵循 SemVer 规范
- **发布就绪**：可一键发布到 PyPI

---

## 2. 包名建议

| 候选名 | 可用性 | 理由 |
|--------|--------|------|
| `ith-toolkit` | 高 | 简洁，toolkit 暗示工具集合 |
| `tumor-heterogeneity` | 中高 | 描述性强，但较长 |
| `ith-analysis` | 中 | 与项目名一致，但偏通用 |
| `ith-score` | 低 | 与已有的 `ITHscore` 包容易混淆 |

**推荐**：`ith-toolkit`，模块导入名使用 `ith`（短且语义明确）。

---

## 3. 目录结构重构

当前结构是 research-project 风格（`src/` + 顶层脚本），需要重构为 package 风格。

### 3.1 重构前后对比

```
# 重构前（当前）
ITH/
  run_all_scores.py              # 顶层入口脚本
  scripts/preprocess_all.py      # 顶层预处理脚本
  src/
    __init__.py
    core/
    models/
    pipeline/
    features/
    survival/
    evaluation/
    visualization/
    utils/
  data/
  results/

# 重构后（目标）
ITH/
  pyproject.toml                 # 包元数据 + 构建配置
  setup.cfg                      # setuptools 配置（可选）
  README.md                      # GitHub README（已生成）
  LICENSE                        # MIT License
  MANIFEST.in                    # 打包包含规则

  ith/                           # 主包目录
    __init__.py                  # 包入口：暴露核心API
    __version__.py               # 版本号统一管理
    _compat.py                   # 兼容层（保留 src 引用）

    core/                        # 核心基础设施
      __init__.py
      base.py                    # BaseHeterogeneityModel
      config.py                  # DataConfig / ModelConfig
      io.py                      # 图像I/O
      preprocess.py              # 预处理工具
      preprocess_pipeline.py     # CT/mask预处理流程
      dataset.py                 # DatasetLoader
      types.py                   # 类型定义

    models/                      # 7个评分模型
      __init__.py                # MODEL_REGISTRY + get_model()
      bih.py, dhi.py, hab.py,
      pth.py, shi.py, thi.py,
      ith_fusion.py

    pipeline/                    # 执行流水线
      __init__.py
      runner.py                  # ModelRunner

    features/                    # 特征提取
      __init__.py
      calculate_radiomics.py
      fractal_fd_lac.py

    survival/                    # 生存分析
      __init__.py
      score_construction.py
      survival_model_training.py

    evaluation/                  # 评估工具
      __init__.py
      icc_analysis.py
      inter_observer_test.py
      kaplan_meier_evaluation.py
      single_cox.py

    visualization/               # 可视化
      __init__.py
      model_visualizers.py
      summary_report.py
      survival_model_comparison.py
      utils.py

    cli/                         # 命令行接口（新增）
      __init__.py
      score.py                   # ith-score 命令
      preprocess.py              # ith-preprocess 命令
      icc.py                     # ith-icc 命令
      survival.py                # ith-survival 命令
      visualize.py               # ith-viz 命令
      main.py                    # 统一入口

  tests/                         # 测试套件（新增）
    conftest.py
    test_core/
    test_models/
    test_pipeline/
    fixtures/

  examples/                      # 示例笔记本（新增）
    01_quickstart.py
    02_batch_processing.py
    03_icc_analysis.py
    04_survival_analysis.py

  docs/                          # 文档（新增，可选 Sphinx）
    conf.py
    index.rst
    api/
    tutorials/
```

### 3.2 文件迁移映射

| 原路径 | 新路径 | 变更说明 |
|--------|--------|----------|
| `src/__init__.py` | `ith/__init__.py` | 暴露高层API |
| `src/core/*.py` | `ith/core/*.py` | 直接迁移 |
| `src/models/*.py` | `ith/models/*.py` | 直接迁移 |
| `src/pipeline/*.py` | `ith/pipeline/*.py` | 直接迁移 |
| `src/features/*.py` | `ith/features/*.py` | 直接迁移 |
| `src/survival/*.py` | `ith/survival/*.py` | 直接迁移 |
| `src/evaluation/*.py` | `ith/evaluation/*.py` | 直接迁移 |
| `src/visualization/*.py` | `ith/visualization/*.py` | 直接迁移 |
| `src/utils/*.py` | `ith/core/` 或 `ith/utils/` | 合并或保留 |
| `run_all_scores.py` | `ith/cli/score.py` | 重构为CLI子命令 |
| `scripts/preprocess_all.py` | `ith/cli/preprocess.py` | 重构为CLI子命令 |
| `项目说明.md` | `docs/` 或保留 | 转为 Sphinx 文档 |
| `README.md` | `README.md` | 已生成，保持不变 |

---

## 4. 核心 API 设计

### 4.1 顶层快捷导入

```python
# ith/__init__.py

__version__ = "1.0.0"

# 核心类
from .core import (
    BaseHeterogeneityModel,
    DataConfig,
    DatasetLoader,
    ModelConfig,
    BIHConfig, DHIConfig, HABConfig,
    PTHConfig, SHIConfig, THIConfig, ITHFusionConfig,
)

# I/O 工具
from .core import load_image, load_seg, save_nifti, match_image_mask_pairs

# 预处理
from .core import preprocess_ct, batch_preprocess

# 模型注册表
from .models import MODEL_REGISTRY, get_model

# 流水线
from .pipeline import ModelRunner

# 高层函数（新增）
from .api import score_case, score_dataset, compare_models
```

### 4.2 用户 API 示例

```python
import ith

# === 单病例评分 ===
vol, msk = ith.load_image("case.nii.gz"), ith.load_seg("mask.nii.gz")
model = ith.get_model("ith_fusion")()
result = model.compute(vol, msk)
print(result["score"])

# === 批量评分 ===
from ith import DataConfig, ModelRunner
cfg = DataConfig(dataset="NSCLC-Radiomics", data_root="./data")
runner = ModelRunner(cfg, models=["bih", "dhi", "hab"])
results = runner.run(visualize=True)

# === 数据加载 ===
from ith import DatasetLoader
loader = DatasetLoader("NSCLC-Radiomics")
print(loader.info())
clinical = loader.clinical
survival = loader.get_survival_data()

# === 预处理 ===
from ith import batch_preprocess
batch_preprocess("NSCLC-Radiomics", data_root="./data")

# === ICC 测试 ===
from ith.evaluation import run_icc_workflow
import pandas as pd
data = pd.read_csv("features_all_observers.csv")
results, summary = run_icc_workflow(data, feature_cols=["bih_fd", "dhi_CV"])
```

### 4.3 新增高层 API (`ith/api.py`)

为降低使用门槛，新增 `ith/api.py` 提供一站式函数：

```python
# ith/api.py

def score_case(image, mask, model: str = "ith_fusion", config=None):
    """对单个病例计算异质性评分."""
    ModelClass = get_model(model)
    model = ModelClass(config=config)
    return model.compute(image, mask)

def score_dataset(dataset, models=None, data_root="./data", out_root="./results",
                  visualize=False, n_workers=4):
    """对数据集批量计算所有模型评分."""
    from .pipeline import ModelRunner
    from .core import DataConfig
    cfg = DataConfig(dataset=dataset, data_root=data_root, output_root=out_root)
    runner = ModelRunner(cfg, models=models)
    return runner.run(visualize=visualize)

def compare_models(csv_paths: dict, metric="c_index"):
    """对比多个模型的性能."""
    ...
```

---

## 5. CLI 设计

使用 `click` 或 `argparse` 构建统一 CLI，入口命令为 `ith`。

```bash
# 查看帮助
ith --help

# === 评分命令 ===
ith score --dataset NSCLC-Radiomics --models bih dhi hab --visualize
ith score --all-datasets --visualize
ith score --dataset immunotherapy --icc-test

# === 预处理命令 ===
ith preprocess --datasets NSCLC-Radiomics NSCLC-Radiogenomics
ith preprocess --all

# === ICC 测试 ===
ith icc --dataset immunotherapy --observers obs1 obs2 obs3

# === 生存分析 ===
ith survival --dataset NSCLC-Radiomics --model ith_fusion

# === 可视化 ===
ith viz --csv results/NSCLC-Radiomics/bih/bih_NSCLC-Radiomics.csv --model bih
```

### CLI 实现 (`ith/cli/main.py`)

```python
import click

@click.group()
@click.version_option(version=ith.__version__)
def cli():
    """ITH Toolkit - Intratumor Heterogeneity Analysis."""
    pass

cli.add_command(score_cmd, name="score")
cli.add_command(preprocess_cmd, name="preprocess")
cli.add_command(icc_cmd, name="icc")
cli.add_command(survival_cmd, name="survival")
cli.add_command(viz_cmd, name="viz")

if __name__ == "__main__":
    cli()
```

---

## 6. 配置管理 (`pyproject.toml`)

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "ith-toolkit"
version = "1.0.0"
description = "Multi-dimensional intratumor heterogeneity (ITH) scoring system"
readme = "README.md"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "your.email@institution.edu"},
]
keywords = ["radiomics", "tumor-heterogeneity", "CT", "oncology", "survival-analysis"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Medical Science Apps.",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]
requires-python = ">=3.9"
dependencies = [
    "numpy>=1.20",
    "pandas>=1.3",
    "scipy>=1.7",
    "scikit-learn>=1.0",
    "scikit-image>=0.19",
    "SimpleITK>=2.0",
    "nibabel>=3.2",
    "matplotlib>=3.4",
    "seaborn>=0.11",
    "tqdm>=4.60",
]

[project.optional-dependencies]
survival = ["lifelines>=0.27", "scikit-survival>=0.20"]
dev = ["pytest>=7.0", "pytest-cov", "black", "ruff", "mypy", "sphinx"]
all = ["ith-toolkit[survival,dev]"]

[project.scripts]
ith = "ith.cli.main:cli"
ith-score = "ith.cli.score:main"
ith-preprocess = "ith.cli.preprocess:main"
ith-icc = "ith.cli.icc:main"

[project.urls]
Homepage = "https://github.com/yourusername/ith-toolkit"
Documentation = "https://ith-toolkit.readthedocs.io"
Repository = "https://github.com/yourusername/ith-toolkit"
Issues = "https://github.com/yourusername/ith-toolkit/issues"

[tool.setuptools.packages.find]
where = ["."]
include = ["ith*"]

[tool.black]
line-length = 100
target-version = ["py39", "py310", "py311"]

[tool.ruff]
line-length = 100
select = ["E", "F", "W", "I"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
addopts = "-v --tb=short"
```

---

## 7. 依赖管理

### 7.1 核心依赖（必需）

| 包 | 最低版本 | 用途 |
|----|----------|------|
| numpy | >=1.20 | 数组运算 |
| pandas | >=1.3 | 数据表格 |
| scipy | >=1.7 | 统计/信号处理 |
| scikit-learn | >=1.0 | 聚类/降维 |
| scikit-image | >=0.19 | 图像处理 |
| SimpleITK | >=2.0 | 医学图像I/O |
| nibabel | >=3.2 | NIfTI格式 |
| matplotlib | >=3.4 | 绘图 |
| seaborn | >=0.11 | 统计可视化 |
| tqdm | >=4.60 | 进度条 |

### 7.2 可选依赖

| 包 | 用途 | 安装方式 |
|----|------|----------|
| lifelines | Cox回归/KM曲线 | `pip install ith-toolkit[survival]` |
| scikit-survival | RSF/GBSA模型 | `pip install ith-toolkit[survival]` |
| click | CLI（开发中可内置） | 建议加入核心依赖 |
| pyradiomics | 外部影像组学特征提取 | `pip install ith-toolkit[radiomics]` |

---

## 8. 向后兼容策略

现有代码中有大量 `from src.core import ...` 和 `from src.models import ...` 的引用。重构为 `ith` 包后，需要：

### 8.1 过渡期方案（推荐3个月内）

在 `src/` 目录下创建兼容 shim：

```python
# src/__init__.py
"""兼容层：将 src.* 重定向到 ith.*"""
import warnings
import sys

try:
    import ith
    sys.modules["src"] = ith
    sys.modules["src.core"] = ith.core
    sys.modules["src.models"] = ith.models
    sys.modules["src.pipeline"] = ith.pipeline
    sys.modules["src.evaluation"] = ith.evaluation
    sys.modules["src.visualization"] = ith.visualization
    sys.modules["src.features"] = ith.features
    sys.modules["src.survival"] = ith.survival
except ImportError:
    warnings.warn(
        "'src' module is deprecated. Please use 'ith' instead. "
        "Install with: pip install ith-toolkit",
        DeprecationWarning,
        stacklevel=2,
    )
```

### 8.2 完全迁移后

删除 `src/` 目录，用户统一使用 `import ith`。

---

## 9. 测试策略

### 9.1 测试框架

使用 **pytest** + **pytest-cov**，目标覆盖率 >80%。

### 9.2 测试目录结构

```
tests/
  conftest.py                  # 共享 fixture
  fixtures/
    synthetic_ct.nii.gz        # 合成CT数据（小体积，<1MB）
    synthetic_mask.nii.gz
  test_core/
    test_io.py
    test_preprocess.py
    test_config.py
    test_dataset.py
  test_models/
    test_bih.py
    test_dhi.py
    test_hab.py
    test_pth.py
    test_shi.py
    test_thi.py
    test_ith_fusion.py
  test_pipeline/
    test_runner.py
  test_cli/
    test_score_cmd.py
    test_preprocess_cmd.py
  test_evaluation/
    test_icc.py
  test_integration/
    test_end_to_end.py         # 端到端：预处理→评分→可视化
```

### 9.3 核心测试用例示例

```python
# tests/test_models/test_bih.py
import numpy as np
import pytest
from ith.models import BIHModel

@pytest.fixture
def synthetic_data():
    """生成合成3D数据."""
    vol = np.random.randn(64, 64, 64).astype(np.float32) * 100
    msk = np.zeros((64, 64, 64), dtype=np.uint8)
    msk[20:45, 20:45, 20:45] = 1
    return vol, msk

def test_bih_basic(synthetic_data):
    vol, msk = synthetic_data
    model = BIHModel()
    result = model.compute(vol, msk)
    assert "score" in result
    assert np.isfinite(result["score"])
    assert "features" in result

def test_bih_empty_mask():
    vol = np.zeros((32, 32, 32), dtype=np.float32)
    msk = np.zeros((32, 32, 32), dtype=np.uint8)
    model = BIHModel()
    result = model.compute(vol, msk)
    assert np.isnan(result["score"]) or result["score"] == 0.0

def test_bih_batch():
    from ith.core import DataConfig
    cfg = DataConfig(dataset="test-dataset", data_root="tests/fixtures")
    model = BIHModel()
    df = model.batch_compute(cfg)
    assert isinstance(df, pd.DataFrame)
    assert "case_id" in df.columns
```

---

## 10. 文档策略

### 10.1 文档结构

```
docs/
  index.md                     # 首页
  quickstart.md                # 快速入门
  installation.md              # 安装指南
  api-reference/               # API参考（自动生成）
    core.md
    models.md
    pipeline.md
    evaluation.md
    visualization.md
  tutorials/                   # 教程
    01-single-case.md
    02-batch-processing.md
    03-icc-analysis.md
    04-survival-analysis.md
  cli-reference.md             # CLI命令参考
  changelog.md                 # 变更日志
```

### 10.2 文档工具

- **mkdocs** + **mkdocstrings**（推荐）：配置简单，支持自动生成API文档
- Sphinx（备选）：生态更成熟，但配置复杂

### 10.3 文档站点托管

- ReadTheDocs（免费，自动构建）
- GitHub Pages（需手动配置 CI）

---

## 11. CI/CD 与发布流程

### 11.1 GitHub Actions 工作流

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.9", "3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: ${{ matrix.python-version }}}
      - run: pip install -e ".[dev,all]"
      - run: pytest --cov=ith --cov-report=xml
      - run: ruff check ith/
      - run: mypy ith/
```

```yaml
# .github/workflows/release.yml
name: Release
on:
  release:
    types: [created]
jobs:
  pypi:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - run: pip install build twine
      - run: python -m build
      - run: twine upload dist/*
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
```

### 11.2 发布检查清单

- [ ] 更新 `ith/__version__.py`
- [ ] 更新 `CHANGELOG.md`
- [ ] 运行完整测试套件通过
- [ ] 检查 `ruff` / `mypy` 无错误
- [ ] 构建 wheel + sdist：`python -m build`
- [ ] 测试安装：`pip install dist/ith_toolkit-*.whl`
- [ ] 创建 GitHub Release
- [ ] CI 自动发布到 PyPI

---

## 12. 迁移实施计划

### Phase 1：目录重构（1天）

1. 创建 `ith/` 目录
2. 将 `src/` 下所有模块迁移到 `ith/`
3. 添加 `ith/__version__.py`、`ith/__init__.py`
4. 添加 `pyproject.toml`
5. 验证 `pip install -e .` 成功
6. 验证 `import ith` 正常

### Phase 2：CLI 开发（1天）

1. 创建 `ith/cli/` 目录
2. 将 `run_all_scores.py` 重构为 `ith/cli/score.py`
3. 将 `scripts/preprocess_all.py` 重构为 `ith/cli/preprocess.py`
4. 创建 `ith/cli/main.py` 统一入口
5. 注册 console_scripts entry points
6. 测试 `ith --help`

### Phase 3：测试建设（2天）

1. 创建 `tests/` 目录结构
2. 编写核心模块的单元测试
3. 编写模型 compute() 的基准测试
4. 编写端到端集成测试
5. 生成合成fixture数据（小体积NIfTI）

### Phase 4：文档建设（1-2天）

1. 配置 mkdocs
2. 编写快速入门和教程
3. 配置 API 文档自动生成
4. 部署到 ReadTheDocs

### Phase 5：向后兼容（0.5天）

1. 在 `src/` 创建 shim 兼容层
2. 添加 DeprecationWarning
3. 更新 README 说明迁移方式

### Phase 6：发布（0.5天）

1. 配置 GitHub Actions
2. 注册 PyPI 账号并配置 API Token
3. 打 tag v1.0.0
4. 发布到 PyPI

---

## 13. 关键决策点

| 决策 | 当前选择 | 备选方案 | 理由 |
|------|----------|----------|------|
| 构建工具 | `setuptools` + `pyproject.toml` | poetry, hatch | setuptools 最广泛支持，pyproject.toml 是现代标准 |
| CLI 框架 | `click` | argparse, typer | click 功能丰富，社区成熟；typer 更现代但依赖较多 |
| 文档工具 | `mkdocs` | Sphinx | mkdocs 配置简单，适合快速上线 |
| 包名 | `ith-toolkit` | `ith-analysis`, `tumor-heterogeneity` | 简洁、可用性高、与现有ITHscore区分 |
| 向后兼容 | 3个月 shim 过渡期 | 直接删除 src/ | 给用户迁移时间，降低升级成本 |
| 测试数据 | 合成小体积数据 | 真实匿名数据 | 避免上传真实医疗数据，合规 |

---

## 14. 风险评估

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| `src/` 引用大量硬编码在项目代码中 | 中 | shim 兼容层 + 分阶段迁移 |
| 依赖版本冲突（SimpleITK/scikit-image） | 低-中 | 精确版本约束 + CI 多版本测试 |
| 大肿瘤数据导致内存不足 | 中 | 文档中标注内存需求，优化稀疏矩阵 |
| PyPI 包名被占用 | 低 | 提前查询可用性，准备备选名 |
| 用户不熟悉 click CLI | 低 | 保留 Python API 作为主要使用方式 |

---

## 15. 附录：参考资源

- [Python Packaging User Guide](https://packaging.python.org/)
- [PyPA setuptools 文档](https://setuptools.pypa.io/)
- [click 文档](https://click.palletsprojects.com/)
- [mkdocs 文档](https://www.mkdocs.org/)
- [ITHscore 原始包](https://github.com/Jackie-LJQ/ITHscore)
