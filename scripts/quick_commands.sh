#!/bin/bash
# 常用命令快捷方式

show_help() {
    cat << 'HELP'
SMPL-Anthropometry 快捷命令
====================================

使用方法：./quick_commands.sh <命令>

可用命令：
  view <file>      - 3D查看SMPL结果
  export <file>    - 导出为HTML
  measure <file>   - 只显示测量数据
  compare          - 对比所有结果
  list             - 列出所有输出
  check            - 检查模型文件

示例：
  ./quick_commands.sh view outputs/output_from_txt_fixed/smpl_params.npz
  ./quick_commands.sh export outputs/output_from_txt_fixed/smpl_params.npz
  ./quick_commands.sh compare

====================================
HELP
}

case "$1" in
    view)
        if [ -z "$2" ]; then
            echo "错误：需要指定文件路径"
            echo "用法：./quick_commands.sh view <smpl_params.npz路径>"
            exit 1
        fi
        python3 -m src.visualization.view_smpl_3d --params "$2"
        ;;
    export)
        if [ -z "$2" ]; then
            echo "错误：需要指定文件路径"
            exit 1
        fi
        output="/mnt/d/smpl_export_$(date +%Y%m%d_%H%M%S).html"
        python3 -m src.visualization.view_smpl_3d --params "$2" --save_html "$output"
        echo "✓ 已导出到: $output"
        ;;
    measure)
        if [ -z "$2" ]; then
            echo "错误：需要指定文件路径"
            exit 1
        fi
        dir=$(dirname "$2")
        cat "$dir/measurements.txt"
        ;;
    compare)
        ./compare_results.sh
        ;;
    list)
        echo "可用的SMPL结果："
        find outputs -name "smpl_params.npz" -o -name "betas.npy" | sort
        ;;
    check)
        python3 tools/check_models.py
        ;;
    *)
        show_help
        ;;
esac
