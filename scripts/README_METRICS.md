# 图像复制和指标计算脚本使用说明

本文档提供了关于如何使用图像复制和指标计算脚本的说明。这些脚本支持命令行参数，可以灵活指定源目录和目标目录。

## 脚本概述

1. `scripts/copy_images.sh` - 复制图像文件并保持目录结构
2. `scripts/calculate_metrics.sh` - 计算图像指标并生成汇总结果

## 使用方法

### 复制图像并计算指标（一步完成）

执行以下命令将自动完成图像复制和指标计算：

```bash
# 使用默认目录
./scripts/copy_images.sh

# 指定自定义目录
./scripts/copy_images.sh --source1 /path/to/cond --source2 /path/to/uncond --target /path/to/output
```

复制图像脚本的完整参数列表：

```
选项:
  -h, --help                显示帮助信息
  -s1, --source1 DIR        指定第一个源目录（cond目录）
  -s2, --source2 DIR        指定第二个源目录（uncond目录）
  -t, --target DIR          指定目标目录
  -n, --no-metrics          不自动运行指标计算脚本
```

默认设置:
- 源目录1: `/home/Users_Work_Space/zsfang/result/ImageNet_v1/cond`
- 源目录2: `/home/Users_Work_Space/zsfang/result/ImageNet_v1/uncond`
- 目标目录: `/home/Users_Work_Space/zsfang/radiodiff-inverse/result/images`

### 仅计算指标（如果图像已经复制）

如果您已经复制了图像，只需要计算指标，可以执行：

```bash
# 使用默认目录
./scripts/calculate_metrics.sh

# 指定自定义目录
./scripts/calculate_metrics.sh --images /path/to/images --metrics /path/to/metrics
```

指标计算脚本的完整参数列表：

```
选项:
  -h, --help                显示帮助信息
  -i, --images DIR          指定图像目录
  -m, --metrics DIR         指定指标结果目录
  -s, --script PATH         指定高级指标计算脚本路径
  -g, --gpu ID              指定GPU ID
  -n, --no-summary          不生成汇总文件
```

默认设置:
- 图像目录: `/home/Users_Work_Space/zsfang/radiodiff-inverse/result/images`
- 指标结果目录: `/home/Users_Work_Space/zsfang/radiodiff-inverse/result/metrics`
- 指标计算脚本: 项目根目录下的 `advanced_metric_calculator.py`

## 目录结构

复制后的目录结构将保持如下格式：

```
<目标目录>/
├── cond/
│   └── [experiment_name]/
│       └── [config_name]/
│           ├── input/
│           ├── label/
│           └── recon/
└── uncond/
    └── [experiment_name]/
        └── [config_name]/
            ├── input/
            ├── label/
            └── recon/
```

指标结果将保存在：

```
<指标目录>/
├── cond/
│   └── [experiment_name]/
│       └── [config_name]_metrics.json
├── uncond/
│   └── [experiment_name]/
│       └── [config_name]_metrics.json
└── all_result.json
```

## 汇总结果

所有实验的指标将汇总在 `all_result.json` 文件中，按照掩码比例和噪声级别组织，便于后续分析和可视化。

## 实用示例

### 示例1: 使用不同的源目录

```bash
./scripts/copy_images.sh --source1 /home/data/experiment1/cond --source2 /home/data/experiment1/uncond
```

### 示例2: 更改目标目录并跳过指标计算

```bash
./scripts/copy_images.sh --target /home/output/experiment_results --no-metrics
```

### 示例3: 仅计算特定目录的指标并指定GPU

```bash
./scripts/calculate_metrics.sh --images /home/data/images --metrics /home/results/metrics --gpu 1
```

### 示例4: 指定自定义的指标计算脚本路径

```bash
./scripts/calculate_metrics.sh --script /path/to/custom/metric_calculator.py
```

## 故障排除

如果脚本运行过程中遇到问题：

1. 确保源目录存在并包含所需的图像文件
2. 确保目标目录具有足够的磁盘空间
3. 检查权限问题，确保脚本有执行权限
4. 如果指标计算失败，检查GPU是否可用
5. 如果需要重新计算指标，可以删除目标目录中的指标文件然后重新运行 `calculate_metrics.sh`
6. 检查指标计算脚本路径是否正确，默认会使用项目目录下的 `advanced_metric_calculator.py`
7. 使用 `-h` 或 `--help` 参数查看帮助信息和可用选项 