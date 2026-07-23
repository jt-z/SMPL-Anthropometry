# SMPL-Anthropometry 修改说明

## 更新时间

2026年7月17日

---

## 一、核心功能增强

### 1. 可视化保存功能

**修改文件：**
- [measure.py](file:///home/zjt/dev/On_Git_Projects/SMPL-Anthropometry/measure.py)
- [visualize.py](file:///home/zjt/dev/On_Git_Projects/SMPL-Anthropometry/visualize.py)

**修改内容：**
- 在 `Measurer.visualize()` 方法中新增 `save_html` 参数
- 在 `Visualizer.visualize()` 方法中新增 `save_html` 参数
- 当指定 `save_html` 参数时，将3D可视化结果保存为HTML文件
- 画布尺寸从 1000x700 调整为 1200x800，提升查看体验

**使用示例：**
```python
measurer.visualize(
    measurement_names=measurement_names,
    title="SMPL Body Measurement",
    save_html="./body_3d.html"
)
```

---

## 二、新增脚本文件

### 2.1 从TXT测量文件拟合SMPL

**文件：** [fit_smpl_from_txt.py](file:///home/zjt/dev/On_Git_Projects/SMPL-Anthropometry/fit_smpl_from_txt.py)

**功能：**
- 解析YOLO检测输出的TXT格式测量结果文件
- 提取17个COCO关键点的3D坐标
- 提取11项骨架测量数据（肩宽、臂长、腿长等）
- 双阶段拟合：关键点拟合 + 测量数据精细拟合
- 生成模拟点云数据用于后续处理

**使用示例：**
```bash
python fit_smpl_from_txt.py --input ./frame_1860_yolo_measure_results.txt --output ./output_from_txt --visualize
```

### 2.2 从TXT测量文件拟合SMPL（修复版）

**文件：** [fit_smpl_from_txt_fixed.py](file:///home/zjt/dev/On_Git_Projects/SMPL-Anthropometry/fit_smpl_from_txt_fixed.py)

**功能：**
- 修复原始版本中的单位转换问题（mm → m）
- 修复坐标系对齐问题（YOLO向下Y轴 → SMPL向上Y轴）
- 添加Procrustes初始对齐，提升拟合精度
- 改进优化器和损失函数
- 以髋部为中心进行坐标中心化

**关键改进：**

| 改进项 | 原始版 | 修复版 |
|--------|--------|--------|
| 单位转换 | 部分转换 | 完整 mm→m |
| 坐标系 | 未处理 | YOLO向下Y轴 → SMPL向上Y轴 |
| 初始对齐 | 无 | Procrustes刚性对齐 |
| 优化器 | Adam | Adam + 更好的正则化 |

**使用示例：**
```bash
python fit_smpl_from_txt_fixed.py --input ./frame_1860_yolo_measure_results.txt --output ./output_from_txt_fixed --visualize
```

### 2.3 关键点诊断工具

**文件：** [diagnose_keypoints.py](file:///home/zjt/dev/On_Git_Projects/SMPL-Anthropometry/diagnose_keypoints.py)

**功能：**
- 分析YOLO关键点与SMPL关键点的差异
- 计算尺度对比（YOLO vs SMPL）
- 检查坐标系方向
- 对比肩宽等关键测量

**使用示例：**
```bash
python diagnose_keypoints.py
```

### 2.4 3D浏览器查看工具

**文件：** [view_smpl_3d.py](file:///home/zjt/dev/On_Git_Projects/SMPL-Anthropometry/view_smpl_3d.py)

**功能：**
- 在浏览器中交互式查看SMPL人体模型
- 支持加载 `betas.npy` 或 `smpl_params.npz` 文件
- 叠加显示体型测量结果（长度、围度）
- 可选择不显示测量线、关键点或关节
- 支持将3D模型保存为HTML文件

**使用示例：**
```bash
# 使用 betas.npy
python view_smpl_3d.py --betas ./fit_output/betas.npy

# 使用 smpl_params.npz
python view_smpl_3d.py --params ./output_smpl_fit/smpl_params.npz

# 保存HTML文件
python view_smpl_3d.py --betas ./fit_output/betas.npy --save_html ./body_3d.html
```

### 2.5 SMPL模型可视化

**文件：** [visualize_smpl.py](file:///home/zjt/dev/On_Git_Projects/SMPL-Anthropometry/visualize_smpl.py)

**功能：**
- 使用Plotly创建SMPL模型的3D可视化
- 显示模型网格和关节点
- 输出HTML文件，支持离线查看

**使用示例：**
```bash
python visualize_smpl.py --betas ./fit_output/betas.npy
```

### 2.6 SMPL与YOLO对比可视化

**文件：** [visualize_smpl_yolo_comparison.py](file:///home/zjt/dev/On_Git_Projects/SMPL-Anthropometry/visualize_smpl_yolo_comparison.py)

**功能：**
- 同时展示SMPL模型网格和YOLO检测的3D关键点
- 显示骨架连线
- 蓝色标记YOLO关键点，红色标记SMPL关节
- 用于调试拟合结果，查看关键点对齐情况

**使用示例：**
```bash
python visualize_smpl_yolo_comparison.py
```

---

## 三、新增文档

### 3.1 TXT拟合指南

**文件：** [TXT_FITTING_GUIDE.md](file:///home/zjt/dev/On_Git_Projects/SMPL-Anthropometry/TXT_FITTING_GUIDE.md)

**内容：**
- 数据来源和格式分析
- 完整使用方法和参数说明
- 代码流程详细解释
- 与现有代码的对比
- 故障排除指南
- 扩展建议

### 3.2 3D查看工具文档

**文件：** [view_smpl_3d.md](file:///home/zjt/dev/On_Git_Projects/SMPL-Anthropometry/view_smpl_3d.md)

**内容：**
- 功能概述和在整个流程中的位置
- 依赖关系说明
- 代码逻辑详细解释
- 参数说明
- 使用示例
- 浏览器交互操作指南

---

## 四、新增可视化输出

**文件列表：**
- [fit_output/body_3d.html](file:///home/zjt/dev/On_Git_Projects/SMPL-Anthropometry/fit_output/body_3d.html) - 3D人体模型可视化（包含测量线）
- [smpl_visualization.html](file:///home/zjt/dev/On_Git_Projects/SMPL-Anthropometry/smpl_visualization.html) - SMPL模型3D可视化
- [smpl_yolo_comparison.html](file:///home/zjt/dev/On_Git_Projects/SMPL-Anthropometry/smpl_yolo_comparison.html) - SMPL与YOLO关键点对比可视化

---

## 五、配置更新

### .gitignore

**修改文件：** [.gitignore](file:///home/zjt/dev/On_Git_Projects/SMPL-Anthropometry/.gitignore)

**新增忽略规则：**
```
output_frame_1860/
output_from_txt/
output_from_txt_fixed/
```

**说明：** 这三个目录是运行时生成的输出目录，不应纳入版本控制。

---

## 六、完整工作流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                         数据来源                                     │
│   3D-Human-Measure/frame_output/frame_1860_yolo_measure_results.txt │
└─────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────┐         ┌─────────────────┐         ┌───────────────────────┐
│ diagnose_     │         │ fit_smpl_from_  │         │ fit_smpl_from_        │
│ keypoints.py  │         │ txt.py          │         │ txt_fixed.py          │
│ (诊断分析)     │         │ (原始版本)       │         │ (修复版本)             │
└───────────────┘         └─────────────────┘         └───────────────────────┘
                                    │                           │
                                    ▼                           ▼
                          ┌───────────────┐             ┌───────────────┐
                          │ output_from_  │             │ output_from_  │
                          │ txt/          │             │ txt_fixed/    │
                          └───────────────┘             └───────────────┘
                                    │                           │
                                    └───────────────┬───────────┘
                                                    ▼
                                        ┌───────────────────┐
                                        │ view_smpl_3d.py   │
                                        │ (3D浏览器查看)      │
                                        └───────────────────┘
                                                    │
                                                    ▼
                                        ┌───────────────────┐
                                        │ fit_output/       │
                                        │ body_3d.html      │
                                        └───────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                     可视化对比工具                                    │
│   visualize_smpl_yolo_comparison.py → smpl_yolo_comparison.html     │
│   visualize_smpl.py → smpl_visualization.html                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 七、提交历史

| 序号 | Commit Hash | 描述 |
|------|-------------|------|
| 1 | `1655f51` | feat: 为可视化添加save_html保存功能 |
| 2 | `0c89dca` | feat: 添加从TXT测量文件拟合SMPL的脚本 |
| 3 | `49d773a` | feat: 添加诊断工具和3D查看工具 |
| 4 | `92df766` | feat: 添加SMPL与YOLO对比可视化脚本 |
| 5 | `11b84a8` | docs: 添加TXT拟合和3D查看的文档 |
| 6 | `4188789` | chore: 添加可视化HTML输出文件 |
| 7 | `ad5f41c` | docs: 添加测量结果文档 |
| 8 | `7e795de` | chore: 在.gitignore中忽略输出目录 |

---

## 八、使用建议

### 推荐使用修复版脚本

对于从TXT测量文件拟合SMPL模型的任务，推荐使用 `fit_smpl_from_txt_fixed.py`，因为它修复了以下关键问题：
- 单位转换问题
- 坐标系对齐问题
- 添加了Procrustes初始对齐

### 调试流程

如果拟合结果不理想，可以按照以下流程进行调试：
1. 运行 `diagnose_keypoints.py` 分析关键点差异
2. 运行 `visualize_smpl_yolo_comparison.py` 查看关键点对齐情况
3. 根据诊断结果调整拟合参数
4. 使用 `view_smpl_3d.py` 查看最终拟合效果

### 输出目录

运行脚本后，结果会保存在以下目录：
- `output_from_txt/` - 原始版脚本输出
- `output_from_txt_fixed/` - 修复版脚本输出
- `fit_output/` - 3D可视化输出

这些目录已添加到 `.gitignore`，不会被Git跟踪。