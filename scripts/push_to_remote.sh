#!/bin/bash
# 推送到远程仓库脚本

echo "════════════════════════════════════════════════════════════════"
echo "           准备推送到远程仓库"
echo "════════════════════════════════════════════════════════════════"
echo ""

echo "当前分支状态："
git status --short
echo ""

echo "待推送的提交："
git log --oneline origin/master..HEAD 2>/dev/null || git log --oneline -10
echo ""

echo "════════════════════════════════════════════════════════════════"
echo "准备执行："
echo "  git push origin master"
echo ""
read -p "是否继续推送？(y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "正在推送..."
    git push origin master
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ 推送成功！"
        echo ""
        echo "可选操作："
        echo "  创建标签: git tag -a v1.0.0-restructured -m 'Project restructuring complete'"
        echo "  推送标签: git push origin v1.0.0-restructured"
    else
        echo ""
        echo "✗ 推送失败，请检查错误信息"
    fi
else
    echo "已取消推送"
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
