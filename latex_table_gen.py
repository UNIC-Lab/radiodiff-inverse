import json
import os
import numpy as np

# 定义输出目录
project_root = os.path.abspath(os.path.dirname(__file__))
output_dir = os.environ.get(
    "LATEX_TABLE_OUTPUT_DIR",
    os.path.join(project_root, "table", "imagenet", "v3"),
)
os.makedirs(output_dir, exist_ok=True)

# 加载JSON数据 - 读取完整数据
summary_json = os.environ.get(
    "LATEX_TABLE_SUMMARY_JSON",
    os.path.join(project_root, "result", "summary_metrics_v0224_0942.json"),
)
with open(summary_json, "r") as f:
    data = json.load(f)  # 读取完整数据结构

# 实验类型和参数
experiment_types = ["random", "sampler"]  # 在JSON中的键
exp_display_names = {
    "random": "Random Sampling",
    "sampler": "Regular Sampling"
}

conditions = ["cond", "uncond"]  # 有无建筑物信息
cond_display_names = {
    "cond": "With Building Information",
    "uncond": "Without Building Information"
}

mask_ratios = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
noise_levels = [0.01, 0.03, 0.05, 0.07, 0.09]

metrics = {
    "psnr": "PSNR (dB)",
    "ssim": "SSIM",
    "lpips": "LPIPS",
    "building_psnr": "Building Area PSNR (dB)",
    "building_ssim": "Building Area SSIM",
    "building_lpips": "Building Area LPIPS"
}

# 为每种条件和指标创建LaTeX表格
for condition in conditions:
    for exp_type in experiment_types:
        for metric, metric_name in metrics.items():
            # 表格文件名
            filename = f"table_{condition}_{exp_type}_{metric}.tex"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, "w") as f:
                # 写入LaTeX表格头部
                f.write("\\begin{table}[htbp]\n")
                f.write("\\centering\n")
                f.write("\\caption{" + f"{metric_name} for {cond_display_names[condition]} with {exp_display_names[exp_type]}" + "}\n")
                f.write("\\label{tab:" + f"{condition}_{exp_type}_{metric}" + "}\n")
                
                # 表格开始
                f.write("\\begin{tabular}{l|" + "c"*len(noise_levels) + "}\n")
                f.write("\\hline\n")
                
                # 表头
                f.write("Mask Ratio & " + " & ".join([f"Noise {noise:.2f}" for noise in noise_levels]) + " \\\\\n")
                f.write("\\hline\n")
                
                # 表格内容
                for mask_ratio in mask_ratios:
                    row = [f"{mask_ratio:.2f}"]
                    for noise_level in noise_levels:
                        key = f"mask{mask_ratio:.2f}_noise{noise_level:.3f}"
                        try:
                            # 修正数据访问路径
                            value = data[condition][exp_type].get(key, {}).get(metric, "-")
                            
                            # 根据指标类型格式化数值
                            if isinstance(value, (int, float)):
                                if metric.startswith("psnr"):
                                    # PSNR保留2位小数
                                    row.append(f"{value:.2f}")
                                elif metric.startswith("ssim"):
                                    # SSIM保留3位小数
                                    row.append(f"{value:.3f}")
                                elif metric.startswith("lpips"):
                                    # LPIPS保留3位小数
                                    row.append(f"{value:.3f}")
                                else:
                                    row.append(f"{value:.3f}")
                            else:
                                row.append("-")
                        except (KeyError, TypeError):
                            row.append("-")
                    
                    f.write(" & ".join(row) + " \\\\\n")
                
                # 表格结束
                f.write("\\hline\n")
                f.write("\\end{tabular}\n")
                f.write("\\end{table}\n")
            
            print(f"创建表格: {filepath}")

# 创建汇总表格 - 仅包含PSNR、SSIM和LPIPS
summary_metrics = ["psnr", "ssim", "lpips"]
representative_masks = [0.5, 0.8, 0.95]

for metric in summary_metrics:
    # 创建汇总表格
    filename = f"table_summary_{metric}.tex"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, "w") as f:
        # 写入LaTeX表格头部
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Comparison of " + f"{metrics[metric]}" + " under Different Experimental Conditions}\n")
        f.write("\\label{tab:summary_" + metric + "}\n")
        
        # 表格开始 - 复杂表头
        f.write("\\begin{tabular}{l|c|" + "c"*len(noise_levels) + "}\n")
        f.write("\\hline\n")
        
        # 主表头
        f.write("Experiment Type & Mask Ratio & " + " & ".join([f"Noise {noise:.2f}" for noise in noise_levels]) + " \\\\\n")
        f.write("\\hline\n")
        
        for condition in conditions:
            for exp_type in experiment_types:
                # 添加实验类型作为大分组 - 修复花括号问题
                num_rows = str(len(representative_masks))
                display_name = cond_display_names[condition] + "+" + exp_display_names[exp_type]
                f.write(f"\\multirow{{{num_rows}}}{{*}}{{{display_name}}}")
                
                first_row = True
                for mask_ratio in representative_masks:
                    if not first_row:
                        f.write(" & ")
                    first_row = False
                    
                    f.write(f" & {mask_ratio:.2f}")
                    
                    for noise_level in noise_levels:
                        key = f"mask{mask_ratio:.2f}_noise{noise_level:.3f}"
                        try:
                            # 修正数据访问路径
                            value = data[condition][exp_type].get(key, {}).get(metric, "-")
                            if isinstance(value, (int, float)):
                                if metric == "psnr":
                                    f.write(f" & {value:.2f}")
                                else:
                                    f.write(f" & {value:.3f}")
                            else:
                                f.write(" & -")
                        except (KeyError, TypeError):
                            f.write(" & -")
                    
                    f.write(" \\\\\n")
                
                # 添加组间分隔线
                f.write("\\hline\n")
        
        # 表格结束
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"创建汇总表格: {filepath}")

# 创建对比表格 - 将条件和无条件结果放在一起比较
for condition in conditions:  # 添加外层循环，分别处理cond和uncond
    for exp_type in experiment_types:
        for metric in summary_metrics:
            filename = f"table_comparison_{condition}_{exp_type}_{metric}.tex"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, "w") as f:
                # 写入LaTeX表格头部
                f.write("\\begin{table}[htbp]\n")
                f.write("\\centering\n")
                # 根据condition和exp_type正确生成标题
                title = f"{metrics[metric]} {cond_display_names[condition]} under {exp_display_names[exp_type]}"
                f.write("\\caption{" + title + "}\n")
                f.write("\\label{tab:comparison_" + f"{condition}_{exp_type}_{metric}" + "}\n")
                
                # 使用新的表格格式
                f.write("\\begin{tabular}{c|l|ccccc}\n")
                f.write("\\hline\n")
                
                # 新的表头格式
                f.write("\\multirow{2}{*}{\\shortstack{Mask\\\\Ratio}} & \\multirow{2}{*}{Metric} & \\multicolumn{5}{c}{Noise Level} \\\\\n")
                f.write("\\cline{3-7}\n")
                f.write(" & & " + " & ".join([f"{noise:.2f}" for noise in noise_levels]) + " \\\\\n")
                f.write("\\hline\n")
                
                # 表格内容
                for mask_ratio in representative_masks:
                    # 写入掩码比例行
                    f.write(f"\\multirow{{2}}{{*}}{{{mask_ratio:.2f}}} & Global")
                    
                    # Global指标
                    for noise_level in noise_levels:
                        key = f"mask{mask_ratio:.2f}_noise{noise_level:.3f}"
                        try:
                            value = data[condition][exp_type].get(key, {}).get(metric, "-")
                            if isinstance(value, (int, float)):
                                f.write(f" & {value:.3f}")
                            else:
                                f.write(" & -")
                        except (KeyError, TypeError):
                            f.write(" & -")
                    f.write(" \\\\\n")
                    
                    # Building指标
                    f.write(" & Building")
                    for noise_level in noise_levels:
                        key = f"mask{mask_ratio:.2f}_noise{noise_level:.3f}"
                        try:
                            value = data[condition][exp_type].get(key, {}).get(f"building_{metric}", "-")
                            if isinstance(value, (int, float)):
                                f.write(f" & {value:.3f}")
                            else:
                                f.write(" & -")
                        except (KeyError, TypeError):
                            f.write(" & -")
                    f.write(" \\\\\n")
                    f.write("\\hline\n")
                
                # 表格结束
                f.write("\\end{tabular}\n")
                f.write("\\end{table}\n")
            
            print(f"创建对比表格: {filepath}")

print("All LaTeX tables have been generated successfully!")
