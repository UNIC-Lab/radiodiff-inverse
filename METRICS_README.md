# FPS-SMC 高级指标计算工具

本工具用于计算FPS-SMC实验的全面指标，包括基本指标、发射机区域指标和建筑物区域指标。

## 功能特点

- 计算全面的图像质量指标：
  - 基本指标：psnr, ssim, nmse, rmse, lpips
  - 发射机区域指标：tx_psnr, tx_ssim, tx_nmse, tx_rmse, tx_lpips
  - 建筑物区域指标：building_psnr, building_ssim, building_nmse, building_rmse, building_lpips
- 自动识别发射机位置（最亮点）和建筑物区域（黑色区域）
- 生成单个实验的详细指标和所有实验的汇总指标
- 计算每个指标的平均值和标准差
- 支持处理单个实验或所有实验
- 自动生成有序汇总文件，便于结果分析

## 目录结构

```
FPS-SMC-2023/
├── advanced_metric_calculator.py  # 主要的指标计算脚本
├── run_calculate_metrics.sh       # 便捷运行脚本
├── METRICS_README.md              # 本文档
└── old_scripts/                   # 旧版本脚本（已弃用）
    ├── compute_metric.py
    ├── recalculate_metrics.py
    └── summary_metrics.py
```

## 使用方法

### 方法1：使用便捷运行脚本

```bash
./run_calculate_metrics.sh --base_dir /path/to/results
```

### 方法2：直接运行Python脚本

```bash
python advanced_metric_calculator.py --base_dir /path/to/results
```

## 参数说明

- `--base_dir`：指定实验结果的基础目录（默认：`./results`）
- `--exp_dir`：仅处理指定的单个实验目录
- `--gpu`：使用的GPU ID（默认：0）
- `--skip_summary`：跳过汇总步骤，不生成汇总文件

## 输出说明

该工具会在每个实验目录下生成以下文件：

- `metrics.json`：该实验的平均指标和标准差
- `per_sample_metrics.json`：该实验中每个样本的详细指标

同时，在基础目录下生成汇总文件：

- `all_result.json`：固定名称的汇总文件，包含所有实验的指标，按掩码率和噪声级别有序排列
- `summary_metrics_vMMDD_HHMM.json`：带时间戳的汇总文件，内容与`all_result.json`相同

## 汇总文件格式

汇总文件采用多层嵌套的JSON格式组织，按照以下层级结构：

```
{
  "cond": {                     // 条件类型（cond/uncond）
    "random": {                 // 掩码类型（random/sampler）
      "mask0.50": {             // 掩码率（按照数值排序）
        "noise0.010": {         // 噪声水平（按照数值排序）
          "psnr": 28.5,         // 各项指标...
          "ssim": 0.92,
          ...
        },
        "noise0.030": { ... },
        ...
      },
      "mask0.60": { ... },
      ...
    },
    "sampler": { ... }
  },
  "uncond": { ... }
}
```

这种结构使得数据便于查询和可视化，特别是当需要按照掩码率或噪声水平进行比较时。

## 指标说明

### 基本指标

- `psnr`：峰值信噪比，越高越好
- `ssim`：结构相似性，越高越好
- `nmse`：归一化均方误差，越低越好
- `rmse`：均方根误差，越低越好
- `lpips`：学习的感知图像块相似性，越低越好

### 发射机区域指标

- `tx_psnr`：发射机区域的峰值信噪比
- `tx_ssim`：发射机区域的结构相似性
- `tx_nmse`：发射机区域的归一化均方误差
- `tx_rmse`：发射机区域的均方根误差
- `tx_lpips`：发射机区域的感知相似性

### 建筑物区域指标

- `building_psnr`：建筑物区域的峰值信噪比
- `building_ssim`：建筑物区域的结构相似性
- `building_nmse`：建筑物区域的归一化均方误差
- `building_rmse`：建筑物区域的均方根误差
- `building_lpips`：建筑物区域的感知相似性

## 注意事项

1. 运行前需要安装必要的依赖：
   - torch
   - numpy
   - PIL
   - skimage
   - lpips
   - tqdm

2. 确保图像目录结构符合要求：
   - `exp_dir/label/`：包含标签图像，命名格式为 `xxxx_label.png`
   - `exp_dir/recon/`：包含重建图像，命名格式为 `xxxx_recon.png`

3. 为了更好的性能，建议在GPU上运行。

4. 当处理单个实验目录时，工具会自动尝试查找基础目录（包含cond/uncond的目录），以便生成正确的汇总文件。

5. 如果只想计算指标而不生成汇总文件，可以使用`--skip_summary`参数。

## 示例

### 处理单个实验

```bash
./run_calculate_metrics.sh --exp_dir /path/to/specific/experiment --gpu 0
```

### 处理所有实验

```bash
./run_calculate_metrics.sh --base_dir /path/to/results --gpu 0
```

### 仅计算指标，不生成汇总

```bash
./run_calculate_metrics.sh --base_dir /path/to/results --skip_summary
``` 