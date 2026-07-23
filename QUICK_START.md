# SMPL-Anthropometry 快速入门

## 5分钟快速开始

### 1. 安装依赖（首次运行）

```bash
pip install -r requirements.txt
```

### 2. 验证安装

```bash
# 检查SMPL模型文件
python3 tools/check_models.py
```

### 3. 运行示例

#### 测量默认SMPL模型
```bash
python3 -m src.core.measure --measure_neutral_smpl_with_mean_shape
```

#### 从TXT文件拟合SMPL（如果有数据）
```bash
python3 -m src.fitting.fit_smpl_from_txt_fixed \
    --input your_data.txt \
    --output outputs/my_result \
    --visualize
```

#### 查看3D结果
```bash
python3 -m src.visualization.view_smpl_3d \
    --params outputs/my_result/smpl_params.npz \
    --save_html outputs/body_3d.html
```

## 常用命令

| 功能 | 命令 |
|------|------|
| 检查模型 | `python3 tools/check_models.py` |
| 测量默认模型 | `python3 -m src.core.measure --measure_neutral_smpl_with_mean_shape` |
| TXT拟合 | `python3 -m src.fitting.fit_smpl_from_txt_fixed --input data.txt --output outputs/result` |
| 3D查看 | `python3 -m src.visualization.view_smpl_3d --betas outputs/result/betas.npy` |
| 诊断关键点 | `python3 tools/diagnose_keypoints.py` |

## 项目结构

```
SMPL-Anthropometry/
├── src/              # 源代码（核心功能）
│   ├── core/        # 测量模块
│   ├── fitting/     # 拟合模块
│   └── visualization/ # 可视化模块
├── tools/           # 实用工具
├── examples/        # 示例代码
├── docs/            # 文档
└── outputs/         # 运行输出
```

## 下一步

- 📖 阅读完整文档：[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
- 🔧 安装指南：[docs/INSTALL.md](docs/INSTALL.md)
- 📝 使用指南：[docs/USAGE_GUIDE.md](docs/USAGE_GUIDE.md)
- 🎯 TXT拟合：[docs/TXT_FITTING_GUIDE.md](docs/TXT_FITTING_GUIDE.md)

## 需要帮助？

运行快速测试脚本：
```bash
bash quickstart.sh
```

查看项目结构：
```bash
cat PROJECT_STRUCTURE.md
```
