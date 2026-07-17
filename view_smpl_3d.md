# view\_smpl\_3d.py — SMPL 人体模型 3D 浏览器查看工具

Resume this session with:──────────────────────────────────────────────────────────────────────────────────────────
claude --resume e238f361-2db7-4019-ac2b-31ee1b78dca6

## 功能概述

将 SMPL 拟合结果（beta 形状参数）重建为三维人体网格，在浏览器中交互式查看，并叠加显示体型测量结果。

输出为基于 **Plotly** 的交互式 HTML 页面，支持旋转、缩放、平移，以及按测量项开关显示。

***

## 在整个流程中的位置

```
深度相机数据 (smpl_input.npz)
        │
        ▼
fit_smpl_from_keypoints.py   ←── 3D 关键点拟合 SMPL
or
fit_smpl_from_data.py        ←── 关键点 + 点云联合拟合
        │
        │  输出
        ├─ fit_output/betas.npy          ← beta 形状参数
        ├─ output_smpl_fit/smpl_params.npz
        └─ fit_output/measurements.txt
        │
        ▼
view_smpl_3d.py              ←── 本脚本：3D 可视化
        │
        │  输出
        ├─ 浏览器交互式 3D 窗口（自动弹出）
        └─ fit_output/body_3d.html（可选保存）
```

***

## 依赖关系

| 模块                            | 作用                         |
| ----------------------------- | -------------------------- |
| `measure.py` → `MeasureBody`  | 重建 SMPL 网格并执行体型测量          |
| `visualize.py` → `Visualizer` | 将网格、关节、测量线组装为 Plotly 3D 图形 |
| `smplx`                       | 从 beta 参数生成 SMPL 顶点和关节     |
| `plotly`                      | 渲染交互式 3D HTML 页面           |

***

## 代码逻辑

### 1. 加载 beta 参数 — `load_betas()`

```
betas.npy          →  np.load()  →  torch.tensor (1, 10)
smpl_params.npz    →  np.load()['betas']  →  torch.tensor (1, 10)
```

两种输入格式均支持：

- `betas.npy`：由 `fit_smpl_from_keypoints.py` 输出，直接存储 10 维 beta 数组
- `smpl_params.npz`：由 `fit_smpl_from_data.py` 输出，包含 `betas` 和 `pose` 字段

两者互斥，不能同时指定。

***

### 2. 重建 SMPL 模型

```python
measurer = MeasureBody(model_type)          # 创建 SMPL 或 SMPLX 测量器
measurer.from_body_model(gender, betas)     # 用 beta 驱动模型，得到 6890 个顶点和关节坐标
```

此步骤使用**零姿势（T-pose）**，体型完全由 beta 决定：

- `beta[0]` 主要控制身高/胖瘦的整体尺度
- `beta[1..9]` 控制更细节的体型变化（肩宽、腿长比例等）

***

### 3. 自动测量

```python
measurer.measure(measurer.all_possible_measurements)
```

调用 `MeasureBody.measure()` 对所有定义的测量项逐一计算：

| 测量类型           | 实现方式                |
| -------------- | ------------------- |
| **长度**（身高、臂长等） | 两个 landmark 顶点的欧式距离 |
| **围度**（胸围、腰围等） | 用平面切割网格，取切面轮廓的凸包周长  |

结果在终端打印，单位 cm。

***

### 4. 3D 可视化

```python
measurer.visualize(
    measurement_names=...,
    visualize_body=True,
    visualize_landmarks=...,
    visualize_joints=...,
    visualize_measurements=...,
    title=...,
    save_html=...,
)
```

内部调用链：

```
measurer.visualize()
    └─> Visualizer.__init__()      # 聚合所有图形数据
    └─> Visualizer.visualize()
            ├─ create_mesh_plot()               # SMPL 三角网格（半透明）
            ├─ create_wireframe_plot()          # 网格框架线
            ├─ create_joint_plot()              # 24 个骨架关节点
            ├─ create_landmarks_plot()          # 体型特征点
            ├─ create_measurement_length_plot() # 长度测量线段（循环）
            ├─ create_measurement_circumference_plot() # 围度测量圈（循环）
            ├─ fig.write_html()     # 保存 HTML（若指定 --save_html）
            └─ fig.show()           # 打开浏览器
```

标题栏自动显示关键测量值（身高、胸围、腰围）。

***

## 参数说明

| 参数                  | 默认值                      | 说明                                     |
| ------------------- | ------------------------ | -------------------------------------- |
| `--betas`           | `./fit_output/betas.npy` | beta 参数文件路径（与 `--params` 互斥）           |
| `--params`          | —                        | `smpl_params.npz` 文件路径（与 `--betas` 互斥） |
| `--gender`          | `NEUTRAL`                | 性别：`NEUTRAL` / `MALE` / `FEMALE`       |
| `--model_type`      | `smpl`                   | 模型类型：`smpl`（6890顶点）或 `smplx`（10475顶点）  |
| `--no_measurements` | False                    | 不绘制测量线，加载速度更快                          |
| `--no_landmarks`    | False                    | 不显示体型特征点                               |
| `--no_joints`       | False                    | 不显示骨架关节点                               |
| `--save_html`       | —                        | 将 3D 模型保存为 HTML 文件，可离线打开               |

***

## 使用示例

### 基础使用（自动读取默认路径）

```bash
cd /home/zjt/dev/On_Git_Projects/SMPL-Anthropometry
python view_smpl_3d.py
```

### 指定 betas 文件

```bash
python view_smpl_3d.py --betas ./fit_output/betas.npy
```

### 使用 fit\_smpl\_from\_data.py 的输出

```bash
python view_smpl_3d.py --params ./output_smpl_fit/smpl_params.npz --gender MALE
```

### 只看体型网格，不画测量线（加载更快）

```bash
python view_smpl_3d.py --no_measurements --no_landmarks --no_joints
```

### 保存 HTML 文件（WSL / 无图形界面环境）

```bash
python view_smpl_3d.py --save_html ./fit_output/body_3d.html
```

WSL 环境下用 Windows 浏览器打开：

```bash
explorer.exe $(wslpath -w ./fit_output/body_3d.html)
```

***

## 浏览器中的交互操作

| 操作     | 方式        |
| ------ | --------- |
| 旋转模型   | 左键拖动      |
| 缩放     | 鼠标滚轮      |
| 平移     | 右键拖动      |
| 开关某项测量 | 点击右侧图例对应项 |
| 悬停查看数值 | 鼠标悬停在测量线上 |
| 重置视角   | 双击空白区域    |

***

## 输出文件

| 文件                        | 说明                                 |
| ------------------------- | ---------------------------------- |
| `fit_output/body_3d.html` | 独立 HTML 文件，包含完整 Plotly JS，无需联网即可打开 |

***

## 前置条件

运行本脚本前需要：

1. **SMPL 模型文件**已放置到 `data/smpl/` 目录：
   ```
   data/smpl/SMPL_NEUTRAL.pkl
   ```
   下载地址：<https://smpl.is.tue.mpg.de/>
2. **已完成 SMPL 拟合**，存在以下文件之一：
   ```
   fit_output/betas.npy          ← fit_smpl_from_keypoints.py 输出
   output_smpl_fit/smpl_params.npz  ← fit_smpl_from_data.py 输出
   ```
3. **Python 依赖**：
   ```bash
   pip install smplx torch numpy plotly trimesh
   ```

