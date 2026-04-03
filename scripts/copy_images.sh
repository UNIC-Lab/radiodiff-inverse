#!/bin/bash

# 获取脚本所在目录的上级目录作为项目根目录
PROJECT_ROOT="$(dirname "$(dirname "$(realpath "$0")")")"
# 默认目录
DEFAULT_SOURCE_DIR1="${PROJECT_ROOT}/result/cond"
DEFAULT_SOURCE_DIR2="${PROJECT_ROOT}/result/uncond"
DEFAULT_TARGET_DIR="${PROJECT_ROOT}/result/images"

# 显示帮助信息
function show_help {
    echo "用法: $(basename $0) [选项]"
    echo "选项:"
    echo "  -h, --help                显示此帮助信息"
    echo "  -s1, --source1 DIR        指定第一个源目录 (默认: ${DEFAULT_SOURCE_DIR1})"
    echo "  -s2, --source2 DIR        指定第二个源目录 (默认: ${DEFAULT_SOURCE_DIR2})"
    echo "  -t, --target DIR          指定目标目录 (默认: ${DEFAULT_TARGET_DIR})"
    echo "  -n, --no-metrics          不自动运行指标计算脚本"
    echo "示例:"
    echo "  $(basename $0) --source1 /path/to/cond --source2 /path/to/uncond --target /path/to/output"
}

# 解析命令行参数
SOURCE_DIR1=${DEFAULT_SOURCE_DIR1}
SOURCE_DIR2=${DEFAULT_SOURCE_DIR2}
TARGET_DIR=${DEFAULT_TARGET_DIR}
RUN_METRICS=true

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -h|--help)
            show_help
            exit 0
            ;;
        -s1|--source1)
            SOURCE_DIR1="$2"
            shift 2
            ;;
        -s2|--source2)
            SOURCE_DIR2="$2"
            shift 2
            ;;
        -t|--target)
            TARGET_DIR="$2"
            shift 2
            ;;
        -n|--no-metrics)
            RUN_METRICS=false
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
if [ ! -d "${SOURCE_DIR1}" ]; then
    echo "错误: 源目录1不存在: ${SOURCE_DIR1}"
    exit 1
fi

if [ ! -d "${SOURCE_DIR2}" ]; then
    echo "错误: 源目录2不存在: ${SOURCE_DIR2}"
    exit 1
fi

# 确保目标目录存在
mkdir -p "${TARGET_DIR}/cond"
mkdir -p "${TARGET_DIR}/uncond"

echo "配置信息:"
echo "源目录1: ${SOURCE_DIR1}"
echo "源目录2: ${SOURCE_DIR2}"
echo "目标目录: ${TARGET_DIR}"
echo "自动运行指标计算: $([ "${RUN_METRICS}" = true ] && echo "是" || echo "否")"
echo ""

echo "开始复制图像文件..."

# 复制 cond 目录下的文件
echo "复制 ${SOURCE_DIR1} 到 ${TARGET_DIR}/cond..."
for experiment_dir in $(find "${SOURCE_DIR1}" -maxdepth 1 -type d -not -path "${SOURCE_DIR1}"); do
    experiment_name=$(basename "${experiment_dir}")
    echo "处理实验: ${experiment_name}"
    
    # 遍历所有子目录（不同的噪声和掩码配置）
    find "${experiment_dir}" -mindepth 1 -maxdepth 1 -type d | while read config_dir; do
        config_name=$(basename "${config_dir}")
        target_config_dir="${TARGET_DIR}/cond/${experiment_name}/${config_name}"
        
        # 创建目标目录结构
        mkdir -p "${target_config_dir}/input"
        mkdir -p "${target_config_dir}/label"
        mkdir -p "${target_config_dir}/recon"
        
        # 复制 input 图像
        if [ -d "${config_dir}/input" ]; then
            echo "复制 ${config_dir}/input 到 ${target_config_dir}/input"
            cp -n "${config_dir}/input"/* "${target_config_dir}/input/" 2>/dev/null || true
        fi
        
        # 复制 label 图像
        if [ -d "${config_dir}/label" ]; then
            echo "复制 ${config_dir}/label 到 ${target_config_dir}/label"
            cp -n "${config_dir}/label"/* "${target_config_dir}/label/" 2>/dev/null || true
        fi
        
        # 复制 recon 图像
        if [ -d "${config_dir}/recon" ]; then
            echo "复制 ${config_dir}/recon 到 ${target_config_dir}/recon"
            cp -n "${config_dir}/recon"/* "${target_config_dir}/recon/" 2>/dev/null || true
        fi
    done
done

# 复制 uncond 目录下的文件
echo "复制 ${SOURCE_DIR2} 到 ${TARGET_DIR}/uncond..."
for experiment_dir in $(find "${SOURCE_DIR2}" -maxdepth 1 -type d -not -path "${SOURCE_DIR2}"); do
    experiment_name=$(basename "${experiment_dir}")
    echo "处理实验: ${experiment_name}"
    
    # 遍历所有子目录（不同的噪声和掩码配置）
    find "${experiment_dir}" -mindepth 1 -maxdepth 1 -type d | while read config_dir; do
        config_name=$(basename "${config_dir}")
        target_config_dir="${TARGET_DIR}/uncond/${experiment_name}/${config_name}"
        
        # 创建目标目录结构
        mkdir -p "${target_config_dir}/input"
        mkdir -p "${target_config_dir}/label"
        mkdir -p "${target_config_dir}/recon"
        
        # 复制 input 图像
        if [ -d "${config_dir}/input" ]; then
            echo "复制 ${config_dir}/input 到 ${target_config_dir}/input"
            cp -n "${config_dir}/input"/* "${target_config_dir}/input/" 2>/dev/null || true
        fi
        
        # 复制 label 图像
        if [ -d "${config_dir}/label" ]; then
            echo "复制 ${config_dir}/label 到 ${target_config_dir}/label"
            cp -n "${config_dir}/label"/* "${target_config_dir}/label/" 2>/dev/null || true
        fi
        
        # 复制 recon 图像
        if [ -d "${config_dir}/recon" ]; then
            echo "复制 ${config_dir}/recon 到 ${target_config_dir}/recon"
            cp -n "${config_dir}/recon"/* "${target_config_dir}/recon/" 2>/dev/null || true
        fi
    done
done

echo "图像复制完成！"
echo "所有图像已复制到: ${TARGET_DIR}"

# 执行指标计算脚本
if [ "${RUN_METRICS}" = true ]; then
    echo "开始计算指标..."
    SCRIPT_DIR="$(dirname "$(realpath "$0")")"
    "${SCRIPT_DIR}/calculate_metrics.sh" --images "${TARGET_DIR}" --metrics "${TARGET_DIR}/../metrics"
else
    echo "跳过指标计算。如需计算指标，请运行:"
    echo "$(dirname "$(realpath "$0")")/calculate_metrics.sh --images \"${TARGET_DIR}\" --metrics \"${TARGET_DIR}/../metrics\""
fi 
