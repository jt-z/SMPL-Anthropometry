# SMPL-Anthropometry 项目结构

## 目录说明

```
SMPL-Anthropometry/
├── src/                          # 源代码（核心库）
│   ├── core/                     # 核心测量模块
│   │   ├── measure.py           # 主测量类 MeasureBody
│   │   ├── measurement_definitions.py  # 测量定义（长度、围度）
│   │   ├── joint_definitions.py        # 关节定义
│   │   ├── landmark_definitions.py     # 地标点定义
│   │   └── utils.py             # 工具函数
│   │
│   ├── fitting/                  # SMPL模型拟合
│   │   ├── fit_smpl_from_data.py       # 从点云数据拟合
│   │   ├── fit_smpl_from_keypoints.py  # 从关键点拟合
│   │   ├── fit_smpl_from_txt.py        # 从TXT文件拟合（原始版）
│   │   └── fit_smpl_from_txt_fixed.py  # 从TXT文件拟合（修复版，推荐）
│   │
│   └── visualization/            # 可视化模块
│       ├── visualize.py          # 核心可视化类 Visualizer
│       ├── view_smpl_3d.py       # 3D浏览器查看工具
│       ├── visualize_measurements.py   # 测量可视化
│       ├── visualize_smpl.py     # SMPL模型可视化
│       └── visualize_smpl_yolo_comparison.py  # SMPL与YOLO对比
│
├── tools/                        # 实用工具脚本
│   ├── check_models.py          # 检查SMPL模型文件
│   ├── diagnose_keypoints.py    # 诊断关键点差异
│   └── evaluate.py              # 评估测量误差
│
├── examples/                     # 示例代码
│   └── example_usage.py         # 基本使用示例
│
├── data/                         # 数据目录
│   ├── smpl/                    # SMPL模型文件（需手动下载）
│   │   ├── SMPL_MALE.pkl
│   │   ├── SMPL_FEMALE.pkl
│   │   ├── SMPL_NEUTRAL.pkl
│   │   └── smpl_body_parts_2_faces.json
│   └── smplx/                   # SMPLX模型文件（需手动下载）
│       └── smplx_body_parts_2_faces.json
│
├── outputs/                      # 运行输出目录
│   ├── fit_output/              # 拟合结果
│   ├── output_from_txt_fixed/   # TXT拟合输出（推荐版本）
│   └── *.html                   # 可视化HTML文件
│
├── docs/                         # 文档
│   ├── INSTALL.md               # 安装指南
│   ├── USAGE_GUIDE.md           # 使用指南
│   ├── TXT_FITTING_GUIDE.md     # TXT拟合指南
│   ├── CHANGELOG.md             # 修改日志
│   ├── DOWNLOAD_SMPL.md         # SMPL模型下载指南
│   ├── FIX_REPORT.md            # 修复报告
│   └── view_smpl_3d.md          # 3D查看工具文档
│
├── docker/                       # Docker配置
│   ├── Dockerfile
│   ├── build.sh
│   ├── run.sh
│   └── requirements.txt
│
├── assets/                       # 静态资源
│   └── measurement_visualization.png
│
├── README.md                     # 项目说明
├── requirements.txt              # Python依赖
├── .gitignore                   # Git忽略配置
└── LICENSE                       # 许可证

```

## 模块说明

### 1. 核心模块 (`src/core/`)

**功能：** SMPL/SMPLX人体模型的测量核心功能

- `MeasureBody` - 主测量类，支持从betas参数或顶点创建模型
- 16种标准人体测量（身高、胸围、腰围、臀围等）
- 可扩展的测量定义系统

**使用：**
```python
from src.core.measure import MeasureBody
from src.core.measurement_definitions import STANDARD_LABELS

measurer = MeasureBody('smpl')
measurer.from_body_model(gender='NEUTRAL', shape=betas)
measurer.measure(measurer.all_possible_measurements)
```

### 2. 拟合模块 (`src/fitting/`)

**功能：** 从不同数据源拟合SMPL模型

| 脚本 | 输入数据 | 推荐度 | 说明 |
|------|---------|-------|------|
| `fit_smpl_from_data.py` | 3D点云 | ⭐⭐⭐ | 通用点云拟合 |
| `fit_smpl_from_keypoints.py` | 3D关键点 | ⭐⭐⭐ | 从关键点拟合 |
| `fit_smpl_from_txt_fixed.py` | YOLO TXT | ⭐⭐⭐⭐⭐ | **推荐使用**，修复版 |
| `fit_smpl_from_txt.py` | YOLO TXT | ⭐⭐ | 原始版（已弃用） |

**使用：**
```bash
# 推荐：从TXT文件拟合（修复版）
python -m src.fitting.fit_smpl_from_txt_fixed \
    --input data.txt \
    --output outputs/my_fit \
    --visualize
```

### 3. 可视化模块 (`src/visualization/`)

**功能：** 3D交互式可视化

- `Visualizer` - 核心可视化引擎（Plotly）
- `view_smpl_3d.py` - 浏览器3D查看器
- 支持导出离线HTML文件

**使用：**
```bash
# 查看拟合结果
python -m src.visualization.view_smpl_3d \
    --params outputs/my_fit/smpl_params.npz \
    --save_html outputs/body_3d.html
```

### 4. 工具模块 (`tools/`)

**功能：** 辅助诊断和评估工具

- `check_models.py` - 验证SMPL模型文件完整性
- `diagnose_keypoints.py` - 分析关键点坐标系和尺度差异
- `evaluate.py` - 计算测量误差（MAE）

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 下载SMPL模型
参考 [docs/DOWNLOAD_SMPL.md](docs/DOWNLOAD_SMPL.md)

### 3. 运行示例
```bash
# 测量默认SMPL模型
python -m src.core.measure --measure_neutral_smpl_with_mean_shape

# 从TXT拟合
python -m src.fitting.fit_smpl_from_txt_fixed \
    --input your_data.txt \
    --output outputs/result \
    --visualize
```

## 工作流程

```
输入数据 (TXT/点云/关键点)
    ↓
拟合SMPL模型 (src/fitting/)
    ↓
生成 smpl_params.npz + betas.npy
    ↓
测量人体 (src/core/measure.py)
    ↓
可视化结果 (src/visualization/)
    ↓
输出 HTML + measurements.txt
```

## 输出文件说明

| 文件 | 说明 |
|------|------|
| `smpl_params.npz` | SMPL模型参数（betas, pose, 性别） |
| `betas.npy` | 形状参数（10维向量） |
| `measurements.txt` | 人体测量结果（文本） |
| `body_3d.html` | 交互式3D可视化（可在浏览器打开） |

## 模块依赖关系

```
src/core/measure.py (核心)
    ↑
    ├─── src/fitting/* (拟合模块依赖测量)
    └─── src/visualization/* (可视化依赖测量)
```

## Git忽略说明

以下目录已在 `.gitignore` 中忽略：
- `outputs/*` - 运行时生成的输出
- `*.html` - 大型可视化文件
- `__pycache__/` - Python缓存

保留在仓库中的：
- `data/*/body_parts_2_faces.json` - 配置文件
- `src/` - 源代码
- `docs/` - 文档

## 开发指南

### 添加新的测量定义

1. 编辑 `src/core/measurement_definitions.py`
2. 在 `MEASUREMENT_TYPES` 添加类型
3. 在对应的 `LENGTHS` 或 `CIRCUMFERENCES` 字典添加定义

### 添加新的拟合方法

1. 在 `src/fitting/` 创建新的脚本
2. 继承或使用 `SMPLFitterFromData` 基类
3. 更新 `src/fitting/__init__.py`

## 常见问题

参考 [docs/INSTALL.md](docs/INSTALL.md) 的常见问题部分。

## 版本历史

参考 [docs/CHANGELOG.md](docs/CHANGELOG.md)
