import os
from datetime import datetime
import json

def create_experiment_dir(base_dir, mask_type, mask_ratio, noise_level, version):
    """创建实验目录
    Args:
        base_dir: 基础目录
        mask_type: 掩码类型 (random_cond, random_uncond, sensor_rect, sensor_rect_uncond, corner_box_random, single_box)
        mask_ratio: 掩码比例
        noise_level: 噪声水平
        version: 版本号
    Returns:
        str: 实验目录路径
    """
    # 首先创建版本目录
    version_dir = os.path.join(base_dir, version)
    
    # 根据mask_type确定目录名
    if mask_type == 'random_cond':
        exp_type = 'cond_random'
    elif mask_type == 'random_uncond':
        exp_type = 'uncond_random'
    elif mask_type == 'sensor_rect':
        exp_type = 'cond_sampler'
    elif mask_type == 'sensor_rect_uncond':
        exp_type = 'uncond_sampler'
    elif mask_type == 'corner_box_random':
        # 根据调用脚本判断是条件还是无条件
        if 'uncond' in base_dir:
            exp_type = 'uncond_corner_box'
        else:
            exp_type = 'cond_corner_box'
    elif mask_type == 'single_box':
        # 根据调用脚本判断是条件还是无条件
        if 'uncond' in base_dir:
            exp_type = 'uncond_single_box'
        else:
            exp_type = 'cond_single_box'
    else:
        raise ValueError(f"Unsupported mask type: {mask_type}")
    
    # 创建实验类型目录
    exp_type_dir = os.path.join(version_dir, exp_type)
    
    # 最后创建具体实验目录
    exp_name = f"mask{mask_ratio:.2f}_noise{noise_level:.3f}"
    exp_dir = os.path.join(exp_type_dir, exp_name)
    
    # 创建必要的子目录
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'input'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'label'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'recon'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'building_mask'), exist_ok=True)
    
    return exp_dir

def save_experiment_results(exp_dir, all_results):
    """保存实验结果
    Args:
        exp_dir: 实验目录
        all_results: 实验结果字典
    """
    # 获取版本目录（回溯两级）
    version_dir = os.path.dirname(os.path.dirname(os.path.dirname(exp_dir)))
    
    # 在版本目录下保存结果
    results_file = os.path.join(version_dir, 'all_results.json')
    
    # 如果文件已存在，读取并更新
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            existing_results = json.load(f)
        existing_results.update(all_results)
        all_results = existing_results
    
    # 保存结果
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4) 