# SMPL-Anthropometry

Measure the SMPL/SMPLX body models and visualize the measurements and landmarks.

<p align="center">
  <img src="https://github.com/DavidBoja/SMPL-Anthropometry/blob/master/assets/measurement_visualization.png" width="950">
</p>

## 📋 项目结构

```
SMPL-Anthropometry/
├── src/                   # 源代码
│   ├── core/             # 核心测量模块
│   ├── fitting/          # SMPL拟合模块
│   └── visualization/    # 可视化模块
├── tools/                # 实用工具
├── examples/             # 示例代码
├── docs/                 # 文档
├── data/                 # SMPL模型数据
└── outputs/              # 运行输出
```

详细说明请查看 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)

<br>

## 🔨 快速开始

### 1. 安装依赖

使用 Docker（推荐）:
```bash
cd docker
sh build.sh
sh docker_run.sh /path/to/SMPL-Anthropometry
```

或手动安装:
```bash
pip install -r requirements.txt
```

详细安装说明：[docs/INSTALL.md](docs/INSTALL.md)

### 2. 下载SMPL模型

下载 SMPL/SMPLX 模型文件并放置到相应目录：
- `data/smpl/` - 放置 `SMPL_{GENDER}.pkl` 文件
- `data/smplx/` - 放置 `SMPLX_{GENDER}.pkl` 文件

详见：[docs/DOWNLOAD_SMPL.md](docs/DOWNLOAD_SMPL.md)

### 3. 运行示例

测量默认SMPL模型:
```bash
python -m src.core.measure --measure_neutral_smpl_with_mean_shape
```

从TXT文件拟合SMPL（推荐使用修复版）:
```bash
python -m src.fitting.fit_smpl_from_txt_fixed \
    --input your_data.txt \
    --output outputs/my_result \
    --visualize
```

查看3D结果:
```bash
python -m src.visualization.view_smpl_3d \
    --params outputs/my_result/smpl_params.npz \
    --save_html outputs/body_3d.html
```

<br>

## 🏃 使用方法

### 基本测量

```python
from src.core.measure import MeasureBody
from src.core.measurement_definitions import STANDARD_LABELS

# 创建测量器
measurer = MeasureBody(model_type='smpl')

# 方法1: 从shape参数创建
measurer.from_body_model(gender='NEUTRAL', shape=betas)

# 方法2: 从顶点创建
measurer.from_verts(verts=vertices)

# 执行测量
measurement_names = measurer.all_possible_measurements
measurer.measure(measurement_names)
measurer.label_measurements(STANDARD_LABELS)

# 获取结果
measurements = measurer.measurements
labeled = measurer.labeled_measurements

# 可视化
measurer.visualize(measurement_names=measurement_names)
```

### 标准测量项目

```python
STANDARD_MEASUREMENTS = {
    'A': 'head circumference',      # 头围
    'B': 'neck circumference',      # 颈围
    'C': 'shoulder to crotch height', # 肩到裆高
    'D': 'chest circumference',     # 胸围
    'E': 'waist circumference',     # 腰围
    'F': 'hip circumference',       # 臀围
    'G': 'wrist right circumference', # 右腕围
    'H': 'bicep right circumference', # 右臂围
    'I': 'forearm right circumference', # 右前臂围
    'J': 'arm right length',        # 右臂长
    'K': 'inside leg height',       # 腿内侧高
    'L': 'thigh left circumference', # 左大腿围
    'M': 'calf left circumference', # 左小腿围
    'N': 'ankle left circumference', # 左脚踝围
    'O': 'shoulder breadth',        # 肩宽
    'P': 'height'                   # 身高
}
```

所有测量单位为 **厘米(cm)**。

<br>

## 🎯 主要功能

### 1. 从TXT测量文件拟合SMPL（推荐）

```bash
python -m src.fitting.fit_smpl_from_txt_fixed \
    --input frame_1860_yolo_measure_results.txt \
    --output outputs/result \
    --keypoint_iterations 500 \
    --visualize
```

**特点：**
- ✅ 修复单位转换问题（mm → m）
- ✅ 修复坐标系对齐（YOLO Y轴向下 → SMPL Y轴向上）
- ✅ Procrustes初始对齐
- ✅ 改进的优化器和损失函数

详见：[docs/TXT_FITTING_GUIDE.md](docs/TXT_FITTING_GUIDE.md)

### 2. 从点云数据拟合

```python
from src.fitting.fit_smpl_from_data import SMPLFitterFromData

fitter = SMPLFitterFromData(model_path='data', model_type='smpl')
fitter.fit(pointcloud, num_iterations=1000)
measurements = fitter.measure_body()
```

### 3. 3D可视化查看

```bash
python -m src.visualization.view_smpl_3d \
    --betas outputs/result/betas.npy \
    --save_html outputs/body_3d.html
```

在浏览器中交互式查看3D人体模型和测量结果。

<br>

## 🛠️ 工具脚本

| 工具 | 功能 | 使用 |
|------|------|------|
| `tools/check_models.py` | 检查SMPL模型文件 | `python tools/check_models.py` |
| `tools/diagnose_keypoints.py` | 诊断关键点差异 | `python tools/diagnose_keypoints.py` |
| `tools/evaluate.py` | 评估测量误差 | `python tools/evaluate.py` |

<br>

## 📊 评估测量误差

```python
from tools.evaluate import evaluate_mae

MAE = evaluate_mae(measurer1.measurements, measurer2.measurements)
print(f"Mean Absolute Error: {MAE:.2f} cm")
```

<br>

## 📝 文档

- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - 项目结构详细说明
- [docs/INSTALL.md](docs/INSTALL.md) - 安装指南
- [docs/USAGE_GUIDE.md](docs/USAGE_GUIDE.md) - 使用指南
- [docs/TXT_FITTING_GUIDE.md](docs/TXT_FITTING_GUIDE.md) - TXT拟合详细指南
- [docs/CHANGELOG.md](docs/CHANGELOG.md) - 更新日志
- [docs/view_smpl_3d.md](docs/view_smpl_3d.md) - 3D查看工具文档

<br>

## 🎨 可视化

### 面部分割可视化
```bash
python -m src.visualization.visualize --visualize_smpl_and_smplx_face_segmentation
```

### 关节可视化
```bash
python -m src.visualization.visualize --visualize_smpl_and_smplx_joints
```

### 地标点可视化
```bash
python -m src.visualization.visualize --visualize_smpl_and_smplx_landmarks
```

<br>

## 📐 测量定义

### 长度测量
通过两个地标点之间的距离定义

### 围度测量
通过平面切割人体模型获得的周长

详见：`src/core/measurement_definitions.py`

<br>

## 🔧 自定义测量

添加新的测量定义：

1. 打开 `src/core/measurement_definitions.py`
2. 在 `MEASUREMENT_TYPES` 中添加测量类型
3. 在 `LENGTHS` 或 `CIRCUMFERENCES` 字典中定义测量
4. 如果是围度测量，在 `CIRCUMFERENCE_TO_BODYPARTS` 中指定身体部位

<br>

## 🚀 高级功能

### 身高标准化

```python
measurer.measure(all_measurement_names)
new_height = 175  # cm
measurer.height_normalize_measurements(new_height)
normalized = measurer.height_normalized_measurements
```

### 姿态模型测量

对于姿态模型的测量，请参考：[pose-independent-anthropometry](https://github.com/DavidBoja/pose-independent-anthropometry/)

<br>

## 🐳 Docker使用

```bash
cd docker
sh build.sh
sh docker_run.sh /path/to/SMPL-Anthropometry
```

容器内运行：
```bash
python -m src.core.measure --measure_neutral_smpl_with_mean_shape
```

<br>

## 🗞️ 引用

如果这个项目对您有帮助，请引用并给个星标 ⭐

```bibtex
@misc{SMPL-Anthropometry,
  author = {Bojani\'{c}, D.},
  title = {SMPL-Anthropometry},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/DavidBoja/SMPL-Anthropometry}},
}
```

<br>

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE)

<br>

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

<br>

## TODO

- [X] 实现 SMPL-X 模型支持
- [X] 添加身高标准化功能
- [X] 从TXT文件拟合SMPL
- [X] 3D浏览器查看工具
- [X] 项目结构重组
- [ ] 实现 STAR 模型支持
- [ ] 实现 SUPR 模型支持
- [ ] 允许姿态模型输入并自动取消姿态后测量
- [ ] 添加批量处理脚本
- [ ] 性能优化（GPU加速）

<br>

⭐ **如果觉得有用，请给个星标！** ⭐
