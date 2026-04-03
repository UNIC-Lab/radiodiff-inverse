#!/bin/bash

# 脚本用于计算指标并生成汇总结果

# 获取脚本所在目录的上级目录作为项目根目录
PROJECT_ROOT="$(dirname "$(dirname "$(realpath "$0")")")"
# 默认目录
DEFAULT_IMAGES_DIR="${PROJECT_ROOT}/result/images"
DEFAULT_METRICS_DIR="${PROJECT_ROOT}/result/metrics"
# 默认脚本路径
DEFAULT_SCRIPT_PATH="${PROJECT_ROOT}/advanced_metric_calculator.py"

# 显示帮助信息
function show_help {
    echo "用法: $(basename $0) [选项]"
    echo "选项:"
    echo "  -h, --help                显示此帮助信息"
    echo "  -i, --images DIR          指定图像目录 (默认: ${DEFAULT_IMAGES_DIR})"
    echo "  -m, --metrics DIR         指定指标结果目录 (默认: ${DEFAULT_METRICS_DIR})"
    echo "  -s, --script PATH         指定高级指标计算脚本路径 (默认: ${DEFAULT_SCRIPT_PATH})"
    echo "  -g, --gpu ID              指定GPU ID (默认: 自动检测)"
    echo "  -n, --no-summary          不生成汇总文件"
    echo "示例:"
    echo "  $(basename $0) --images /path/to/images --metrics /path/to/metrics"
}

# 解析命令行参数
IMAGES_DIR=${DEFAULT_IMAGES_DIR}
METRICS_DIR=${DEFAULT_METRICS_DIR}
SCRIPT_PATH=${DEFAULT_SCRIPT_PATH}
GPU_ID=""
SKIP_SUMMARY=false

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -h|--help)
            show_help
            exit 0
            ;;
        -i|--images)
            IMAGES_DIR="$2"
            shift 2
            ;;
        -m|--metrics)
            METRICS_DIR="$2"
            shift 2
            ;;
        -s|--script)
            SCRIPT_PATH="$2"
            shift 2
            ;;
        -g|--gpu)
            GPU_ID="$2"
            shift 2
            ;;
        -n|--no-summary)
            SKIP_SUMMARY=true
            shift
            ;;
        *)
            echo "错误: 未知选项 $1"
            show_help
            exit 1
            ;;
    esac
done

# 检查目录是否存在
if [ ! -d "${IMAGES_DIR}" ]; then
    echo "错误: 图像目录不存在: ${IMAGES_DIR}"
    exit 1
fi

if [ ! -f "${SCRIPT_PATH}" ]; then
    echo "错误: 指标计算脚本不存在: ${SCRIPT_PATH}"
    echo "请检查脚本路径或使用 -s 选项指定正确的脚本路径"
    exit 1
fi

# 确保指标结果目录存在
mkdir -p "${METRICS_DIR}"

echo "配置信息:"
echo "图像目录: ${IMAGES_DIR}"
echo "指标结果目录: ${METRICS_DIR}"
echo "指标计算脚本: ${SCRIPT_PATH}"

# 自动检测GPU设备
if [ -z "${GPU_ID}" ]; then
    if [ -e "/dev/nvidia0" ]; then
        GPU_ID=0
    elif [ -e "/dev/nvidia1" ]; then
        GPU_ID=1
    else
        echo "警告: 未检测到NVIDIA GPU设备，将使用CPU进行计算"
        GPU_ID=0
    fi
fi

echo "使用GPU ID: ${GPU_ID}"
echo "开始计算指标..."

# 运行指标计算脚本
python3 ${SCRIPT_PATH} --base_dir ${IMAGES_DIR} --gpu ${GPU_ID} --skip_summary

# 创建符号链接，将所有指标JSON文件链接到metrics目录
echo "创建指标文件的符号链接..."

# 复制并链接所有实验的指标文件到metrics目录
find "${IMAGES_DIR}" -name "metrics.json" | while read metrics_file; do
    # 获取相对路径，例如: cond/random/mask0.5_noise0.01
    rel_path=$(echo ${metrics_file} | sed "s|${IMAGES_DIR}/||" | sed "s|/metrics.json||")
    # 创建目标目录
    mkdir -p "${METRICS_DIR}/$(dirname ${rel_path})"
    # 复制指标文件
    cp "${metrics_file}" "${METRICS_DIR}/${rel_path}_metrics.json"
done

# 创建汇总文件
if [ "${SKIP_SUMMARY}" = false ]; then
    echo "生成汇总指标文件..."
    python3 ${SCRIPT_PATH} --base_dir ${IMAGES_DIR} --gpu ${GPU_ID}

    # 将汇总文件复制到metrics目录
    if [ -f "${IMAGES_DIR}/all_result.json" ]; then
        cp "${IMAGES_DIR}/all_result.json" "${METRICS_DIR}/all_result.json"
        echo "汇总文件: ${METRICS_DIR}/all_result.json"
    fi
    
    latest_summary=$(ls -t ${IMAGES_DIR}/summary_metrics_v*.json 2>/dev/null | head -1)
    if [ -n "${latest_summary}" ]; then
        cp "${latest_summary}" "${METRICS_DIR}/"
        echo "详细汇总文件: ${METRICS_DIR}/$(basename ${latest_summary})"
    fi
else
    echo "跳过汇总文件生成"
fi

echo "指标计算和汇总完成！"
echo "结果保存在: ${METRICS_DIR}" 
