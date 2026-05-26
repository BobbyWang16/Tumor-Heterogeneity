# Python QuanTAV Core

这个目录是 MATLAB 项目核心算法的 Python 版本，目标是直接输入：

- `image`：CT 体数据，可选；当前特征计算不依赖 CT 灰度
- `tumor_mask`：肿瘤二值 mask
- `vessel_mask`：血管二值 mask
- `params`：spacing、瘤周半径、最小连通域、是否输出可视化等参数

输出：

- 91 维特征向量
- 带列名的 CSV
- 肿瘤/血管 3D 图、曲率图、扭转图

## 安装依赖

在 `python` 目录或项目根目录执行：

```bash
pip install -r python/requirements.txt
```

也可以安装为本地可编辑包：

```bash
cd python
pip install -e .
```

## Python API

```python
from quantav_py import QuantavParams, analyze_case

params = QuantavParams(
    spacing=(1.0, 1.0, 1.0),      # 必须与数组轴顺序一致
    peritumoral_radius_mm=30.0,
    min_vessel_component_size=20,
    create_visualizations=True,
)

result = analyze_case(
    image=None,
    tumor_mask=tumor_mask,        # numpy 3D array 或文件路径
    vessel_mask=vessel_mask,      # numpy 3D array 或文件路径
    params=params,
    output_dir="./output",
    case_id="case001",
)

print(result.features.shape)      # (91,)
print(result.as_dict())           # {feature_name: value}
```

`load_volume()` 使用 SimpleITK 优先读取 `.nii/.nii.gz/.nrrd/.mha`，会把 SimpleITK 的 `(x, y, z)` spacing 反转成 numpy 数组轴顺序 `(z, y, x)`。

## 命令行

从 `python` 目录运行：

```bash
python -m quantav_py.cli \
  --image ../example/ct1.nii \
  --tumor-mask ../example/seg1.nii \
  --vessel-mask ../example/vessel.nii \
  --output-dir ../python_output \
  --case-id demo \
  --radius-mm 30
```

如果只想输出特征，不生成图：

```bash
python -m quantav_py.cli \
  --tumor-mask tumor.nii.gz \
  --vessel-mask vessel.nii.gz \
  --output-dir output \
  --case-id case001 \
  --no-visualization
```

## 特征顺序

输出列顺序与 MATLAB 项目保持同一大类结构：

- 第 1-61 列：morphology，包含分支 torsion、曲率统计、直方图、血管体积/密度、feeding vessel 数量和比例。
- 第 62-91 列：organization，包含三种笛卡尔投影和三种球坐标投影的 Hough 方向统计。

注意：MATLAB 使用 Fast Marching skeleton，本版本使用 `skimage.morphology.skeletonize` 加图分支拆分；数值不会逐位等同 MATLAB，但计算定义和输出结构对齐，速度和部署成本更适合 Python。
