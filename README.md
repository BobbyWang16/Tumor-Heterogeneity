# Tumor Heterogeneity

**简体中文** · [English](README.en.md)

**基于 CT 与肿瘤分割的多维异质性分析工具集。**

使用统一 Python 接口计算肿瘤密度、形状、纹理、生境与瘤周异质性，导出病例级特征表，并进行队列诊断与生存探索。

[Getting Started](docs/getting-started.md) · [安装](docs/installation.md) · [用户指南](docs/index.md) · [API Reference](docs/api.md) · [包发布设计](docs/package-design.md)

## 我们想解决什么问题？

两个肿瘤可以具有相同平均灰度，却有完全不同的内部结构。一个可能比较均匀，另一个可能由交错的小区域组成。这个工具集用多个视角描述这些差异：**灰度分布、空间排列、几何外形和瘤周环境**。

每个模型提供主分数及解释它的特征，部分模型还提供空间图。先用 DHI 理解“灰度有多分散”，再按研究问题选择其他模型。阅读 [基本概念](docs/concepts.md) 和 [设计原理](docs/design-principles.md)，无需先掌握全部算法。

## 安装

当前提供源码安装；主包要求 Python ≥ 3.9，推荐 Python 3.11。QuanTAV 扩展要求 Python ≥ 3.10。

```bash
git clone https://github.com/BobbyWang16/Tumor-Heterogeneity.git
cd Tumor-Heterogeneity
python -m pip install -e .
```

发行配置名为 `tumor-heterogeneity`，当前导入名仍为 `src`。正式发布前拟迁移为 `tumor_heterogeneity`，详见包设计；这里不假设包已发布至 PyPI。

## 第一次计算

下面的例子无需下载数据：

```python
import numpy as np
from src.models import DHIModel

rng = np.random.default_rng(42)
image = rng.normal(60, 15, (32, 32, 32)).astype(np.float32)
z, y, x = np.indices(image.shape)
mask = ((z - 16)**2 + (y - 16)**2 + (x - 16)**2 <= 9**2)

result = DHIModel().compute(image, mask)
print(result["score"])
print(result["features"])
```

也可以运行 `python examples/quickstart.py`。真实影像的读取、空间校验和特征导出见 [Getting Started](docs/getting-started.md)。

## 支持的模型

| 模型 | 分析维度 | 方法 |
| --- | --- | --- |
| BIH | 最大轴向切面 | 信号加权盒计数 |
| DHI | 密度 | 灰度分布与局部变异 |
| HAB | 生境 | 带空间先验的 GMM |
| PTH | 瘤周组织 | 环带、SLIC 分块与块间差异 |
| SHI | 形状 | 几何、凸包、分形与空隙度 |
| THI | 纹理分区 | 体素特征与 KMeans |
| ITH-FS | 多尺度空间分区 | 空间谱聚类 |

各模型分数定义和范围不同，不应直接平均。ITH-FS 是独立模型，不是其余模型的分数融合器。详见 [模型与输出](docs/models.md)。

## 分析一个队列

准备 `data/demo/image_process/` 与 `data/demo/mask_process/` 下同名的 `.nii.gz` 文件后：

```python
from src.core import DataConfig
from src.pipeline import ModelRunner

config = DataConfig(dataset="demo", data_root="./data", output_root="./results")
paths = ModelRunner(config, models=["bih", "dhi"]).run(visualize=False)
```

或在仓库根目录执行：

```bash
python run_all_scores.py --dataset demo --models bih dhi --no-visualize
```

输入需要事先对齐；运行全部模型前统一到 1 mm 各向同性。当前 `compute()` 不会自动配准或重采样。目录、输出 CSV 与预处理限制见 [数据与批处理](docs/data-and-batch.md)。

## 可选模块

```bash
# 生存分析
python -m pip install -e ".[survival]"

# QuanTAV 肿瘤相关血管特征，独立包
python -m pip install -e ./QuanTAV
```

- **QuanTAV**：血管形态与组织特征，使用 `quantav_py.analyze_case()`。
- **DualCT**：双期 CT 分析脚本，尚未整合到主包公共接口。
- **analysis / R**：基因分析与探索工作流。
- **ITHscore**：独立第三方参考仓库。

## 文档与开发

```bash
python -m pip install -e ".[docs]"
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

生成 `docs/_build/html/index.html` 与英文入口 `docs/_build/html/en/index.html`，提供逐页语言切换、侧边导航和搜索。Read the Docs 构建配置已包含在仓库中。结构参考 [PyRadiomics 文档](https://pyradiomics.readthedocs.io/en/latest/)，内容与接口依据本仓库实现。

测试安装步骤见 [开发指南](docs/contributing.md)。现有代码重构记录见 [重构说明](重构说明.md)。

## 引用与许可

发布前需要补齐主包许可证、第三方代码归属与软件引用信息；当前根目录没有正式 LICENSE 文件。原版 ITHscore 的论文与许可证不能自动代表本项目全部实现。当前结果属于研究特征，尚不能据此声称经过临床验证。
