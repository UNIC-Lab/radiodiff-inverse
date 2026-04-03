#!/bin/bash

# FPS-SMC高级指标计算脚本
# 该脚本用于计算全面的图像质量指标，包括基本指标、发射机指标和建筑物指标

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# 显示帮助信息
show_help() {
    echo "使用方法: $0 [选项]"
    echo "选项:"
    echo "  --base_dir DIR    指定实验结果的基础目录 (默认: $SCRIPT_DIR/results)"
    echo "  --exp_dir DIR     仅处理指定的实验目录"
    echo "  --gpu NUM         使用的GPU ID (默认: 0)"
    echo "  --skip_summary    跳过汇总步骤"
    echo "  --help            显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 --base_dir /path/to/results"
    echo "  $0 --exp_dir /path/to/specific/experiment"
    echo "  $0 --gpu 1"
    echo "  $0 --skip_summary"
}

# 默认参数
BASE_DIR="$SCRIPT_DIR/results"
EXP_DIR=""
GPU=0
SKIP_SUMMARY=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --base_dir)
            BASE_DIR="$2"
            shift 2
            ;;
        --exp_dir)
            EXP_DIR="$2"
            shift 2
            ;;
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --skip_summary)
            SKIP_SUMMARY=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

# 确认运行参数
echo "运行参数:"
echo "  基础目录: $BASE_DIR"
if [ -n "$EXP_DIR" ]; then
    echo "  实验目录: $EXP_DIR"
else
    echo "  处理所有实验"
fi
echo "  GPU ID: $GPU"
if [ "$SKIP_SUMMARY" = true ]; then
    echo "  跳过汇总: 是"
else
    echo "  跳过汇总: 否"
fi
echo ""

# 确认是否继续
read -p "是否继续? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

# 构建命令
CMD="python3 advanced_metric_calculator.py --gpu $GPU"
if [ -n "$BASE_DIR" ]; then
    CMD="$CMD --base_dir $BASE_DIR"
fi
if [ -n "$EXP_DIR" ]; then
    CMD="$CMD --exp_dir $EXP_DIR"
fi
if [ "$SKIP_SUMMARY" = true ]; then
    CMD="$CMD --skip_summary"
fi

# 运行命令
echo "执行: $CMD"
eval "$CMD" 