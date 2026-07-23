#!/bin/bash
# 点云拟合测试脚本

echo "════════════════════════════════════════════════════════════════"
echo "           点云SMPL拟合测试"
echo "════════════════════════════════════════════════════════════════"
echo ""

# 输入文件
INPUT_FILE="data/input_points/case1/rail_scan_20260717_014654_replay.ply"
OUTPUT_DIR="outputs/from_pointcloud_test"

echo "输入文件: $INPUT_FILE"
echo "输出目录: $OUTPUT_DIR"
echo ""

# 1. 检查文件
echo "1. 检查点云数据..."
echo "----------------------------------------"
if [ -f "$INPUT_FILE" ]; then
    echo "✓ 文件存在"
    ls -lh "$INPUT_FILE" | awk '{print "  大小:", $5}'
else
    echo "✗ 文件不存在: $INPUT_FILE"
    exit 1
fi
echo ""

# 2. 检查依赖
echo "2. 检查依赖..."
echo "----------------------------------------"
python3 -c "import trimesh; print('✓ trimesh')" 2>/dev/null || echo "✗ trimesh"
python3 -c "import smplx; print('✓ smplx')" 2>/dev/null || echo "✗ smplx"
python3 -c "import torch; print('✓ torch')" 2>/dev/null || echo "✗ torch"
echo ""

# 3. 运行拟合
echo "3. 运行SMPL拟合..."
echo "----------------------------------------"
echo "命令: python3 -m src.fitting.fit_smpl_from_data \\"
echo "        --input $INPUT_FILE \\"
echo "        --output $OUTPUT_DIR"
echo ""
echo "开始拟合（这可能需要几分钟）..."
echo ""

python3 -m src.fitting.fit_smpl_from_data \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_DIR"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ 拟合完成"
else
    echo ""
    echo "✗ 拟合失败"
    exit 1
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "           拟合结果"
echo "════════════════════════════════════════════════════════════════"
echo ""

# 4. 查看结果
if [ -f "$OUTPUT_DIR/measurements.txt" ]; then
    echo "测量结果:"
    echo "----------------------------------------"
    head -30 "$OUTPUT_DIR/measurements.txt"
    echo ""
else
    echo "⚠️  未找到测量结果文件"
fi

# 5. 查看SMPL参数
if [ -f "$OUTPUT_DIR/smpl_params.npz" ]; then
    echo "SMPL参数:"
    echo "----------------------------------------"
    python3 << 'PYEOF'
import numpy as np
data = np.load('outputs/from_pointcloud_test/smpl_params.npz')
print(f"Betas: {data['betas']}")
print(f"是否全零: {np.allclose(data['betas'], 0)}")
PYEOF
    echo ""
else
    echo "⚠️  未找到SMPL参数文件"
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "完成！"
echo ""
echo "查看详细结果："
echo "  $ cat $OUTPUT_DIR/measurements.txt"
echo ""
echo "3D可视化："
echo "  $ python3 -m src.visualization.view_smpl_3d \\"
echo "      --params $OUTPUT_DIR/smpl_params.npz"
echo "════════════════════════════════════════════════════════════════"

