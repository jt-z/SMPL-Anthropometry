#!/bin/bash
# SMPL-Anthropometry 快速测试脚本

echo "=========================================="
echo "SMPL-Anthropometry 快速测试"
echo "=========================================="
echo ""

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未安装"
    exit 1
fi

echo "✓ Python3: $(python3 --version)"
echo ""

# 检查SMPL模型
echo "1. 检查SMPL模型文件..."
python3 tools/check_models.py
echo ""

# 测量默认SMPL模型
echo "2. 测量默认SMPL模型..."
python3 -m src.core.measure --measure_neutral_smpl_with_mean_shape
echo ""

# 检查输出
echo "3. 检查输出目录..."
if [ -d "outputs" ]; then
    echo "✓ outputs/ 目录存在"
    ls -lh outputs/ 2>/dev/null || echo "  (空目录)"
else
    echo "✓ outputs/ 目录将在运行时创建"
fi
echo ""

echo "=========================================="
echo "测试完成！"
echo "=========================================="
echo ""
echo "下一步："
echo "  1. 查看项目结构: cat PROJECT_STRUCTURE.md"
echo "  2. 查看文档: ls docs/"
echo "  3. 运行拟合: python3 -m src.fitting.fit_smpl_from_txt_fixed --help"
echo "  4. 3D查看: python3 -m src.visualization.view_smpl_3d --help"
