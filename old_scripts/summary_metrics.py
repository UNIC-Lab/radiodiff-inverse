import os
import json
from collections import defaultdict

def collect_metrics(base_dir):
    """
    收集所有实验的指标数据
    """
    metrics_data = defaultdict(lambda: {
        'cond': {'random': defaultdict(dict), 'sampler': defaultdict(dict)},
        'uncond': {'random': defaultdict(dict), 'sampler': defaultdict(dict)}
    })

    # 遍历目录结构
    for root, dirs, files in os.walk(os.path.join(base_dir, "ImageNet_v1")):
        if 'metrics.json' in files:
            # 解析路径信息
            path_parts = root.split('/')
            try:
                # 获取cond_mode (cond/uncond)
                cond_idx = path_parts.index("ImageNet_v1") + 1
                cond_mode = path_parts[cond_idx]
                # 获取mask_mode (random/sampler)
                mask_mode = path_parts[cond_idx + 1]
                # 获取实验目录名
                exp_dir = path_parts[cond_idx + 2]
            except IndexError:
                print(f"跳过非标准目录: {root}")
                continue

            # 提取掩码率和噪声水平
            try:
                mask_part, noise_part = exp_dir.split('_')
                mask_ratio = float(mask_part.replace('mask', ''))
                noise_sigma = float(noise_part.replace('noise', ''))
            except Exception as e:
                print(f"目录名解析失败: {exp_dir} ({str(e)})")
                continue

            # 加载指标数据
            metrics_path = os.path.join(root, 'metrics.json')
            try:
                with open(metrics_path, 'r') as f:
                    data = json.load(f)
                    # 按区域存储指标
                    metrics_data[cond_mode][mask_mode][(mask_ratio, noise_sigma)] = {
                        'overall': data['overall'],
                        'building': data['building'],
                        'non_building': data['non_building'],
                        'tx': data['tx']
                    }
            except Exception as e:
                print(f"加载 {metrics_path} 失败: {str(e)}")
    
    return metrics_data

def save_summary(data, output_dir):
    """
    保存汇总结果（保持不变）
    """
    # [与之前相同的保存逻辑...]

if __name__ == "__main__":
    # 基础目录默认为仓库根目录下的results
    base_dir = "results"
    output_dir = os.path.join(base_dir, "ImageNet_v1", "result")
    
    print("开始收集指标数据...")
    metrics = collect_metrics(base_dir)
    
    print("\n保存汇总结果...")
    save_summary(metrics, output_dir) 
