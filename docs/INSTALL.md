# SMPL-Anthropometry 安装指南

本文档提供项目依赖的安装说明。

## 环境要求

- Python 3.7+
- 推荐使用虚拟环境（conda 或 venv）

## 快速安装

### 方法1：使用 requirements.txt

```bash
pip install -r requirements.txt
```

### 方法2：使用 Docker（推荐）

```bash
cd docker
sh build.sh
sh docker_run.sh /path/to/SMPL-Anthropometry
```

## 依赖说明

### 核心库

- **numpy** (>=1.18.5) - 数值计算
- **torch** (>=1.6.0) - 深度学习框架，用于SMPL模型
- **scipy** (>=1.10.0) - 科学计算（Procrustes对齐、优化）
- **scikit-learn** (>=1.0.2) - 机器学习工具

### SMPL模型

- **smplx** (>=0.1.26) - SMPL/SMPLX人体模型库

### 3D处理

- **trimesh** (>=3.15.1) - 3D网格处理

### 可视化

- **plotly** (>=5.10.0) - 交互式3D可视化
- **matplotlib** (>=3.3.0) - 2D绘图

### 数据处理

- **pandas** (>=1.3.5) - 数据分析
- **tqdm** (>=4.66.1) - 进度条

## GPU 支持（可选）

如果您有 NVIDIA GPU，建议安装支持 CUDA 的 PyTorch 版本以加速计算：

### CUDA 11.8
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### CUDA 12.1
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### CPU 版本
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## 验证安装

运行以下命令检查安装是否成功：

```bash
python check_models.py
```

或者运行示例：

```bash
python measure.py --measure_neutral_smpl_with_mean_shape
```

## 常见问题

### 1. torch 安装失败

如果直接安装 torch 失败，请先访问 [PyTorch官网](https://pytorch.org/get-started/locally/) 选择适合您系统的安装命令。

### 2. smplx 模型文件缺失

需要手动下载 SMPL/SMPLX 模型文件：

1. 访问 [SMPL官网](https://smpl.is.tue.mpg.de/) 注册并下载
2. 将 `SMPL_{GENDER}.pkl` 文件放入 `data/smpl/` 目录
3. 将 `SMPLX_{GENDER}.pkl` 文件放入 `data/smplx/` 目录

详见 [DOWNLOAD_SMPL.md](DOWNLOAD_SMPL.md)

### 3. trimesh 渲染问题

如果遇到渲染问题，可能需要安装额外的依赖：

```bash
pip install pyglet
```

### 4. plotly 可视化不显示

确保浏览器支持 WebGL。如果在服务器上运行，使用 `--save_html` 参数保存为HTML文件后下载到本地查看。

## 开发环境（可选）

如果需要开发或调试，可以安装额外的工具：

```bash
pip install jupyter ipython pytest black flake8
```

## 版本兼容性

本项目已在以下环境测试通过：

- Python 3.8 / 3.9 / 3.10
- Ubuntu 20.04 / 22.04
- Windows 10/11 (WSL2)
- macOS 12+

## 更新依赖

更新所有依赖到最新版本：

```bash
pip install --upgrade -r requirements.txt
```

## 卸载

```bash
pip uninstall -r requirements.txt -y
```
