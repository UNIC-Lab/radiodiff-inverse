import os
import json
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from util.metrics import evaluate_metrics
from lpips import LPIPS
from datetime import datetime

def load_image(path):
    """加载图像并转换为tensor"""
    img = Image.open(path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),  # 先调整大小
        transforms.CenterCrop((256, 256))  # 再裁剪
    ])
    return transform(img)

def get_image_pairs(exp_dir):
    """获取实验目录下所有配对的图像路径
    Args:
        exp_dir: 实验目录路径
    Returns:
        list of tuples: [(recon_path, label_path), ...]
    """
    # 获取recon目录下的所有图片
    recon_dir = os.path.join(exp_dir, 'recon')
    if not os.path.exists(recon_dir):
        return []
    
    recon_files = sorted([f for f in os.listdir(recon_dir) if f.endswith('_recon.png')])
    pairs = []
    
    for recon_file in recon_files:
        # 从recon文件名中提取序号（如从"0000_recon.png"中提取"0000"）
        img_id = recon_file.split('_')[0]
        # 构建对应的label文件名（如"0000_label.png"）
        label_file = f"{img_id}_label.png"
        
        recon_path = os.path.join(recon_dir, recon_file)
        label_path = os.path.join(exp_dir, 'label', label_file)
        
        # 检查文件是否存在
        if os.path.exists(label_path):
            pairs.append((recon_path, label_path))
    
    return pairs

def recalculate_metrics(base_dir, device='cuda'):
    """重新计算所有实验的指标"""
    # 生成版本标识
    timestamp = datetime.now().strftime("%m%d_%H%M")
    metrics_version = f"metrics_v{timestamp}"
    
    # 初始化 LPIPS 模型（只需要一次）
    lpips_fn = LPIPS(net='alex').to(device)
    print(f"Using metrics version: {metrics_version}")
    
    # 遍历条件类型
    for cond_type in ['cond', 'uncond']:
        cond_dir = os.path.join(base_dir, cond_type)
        if not os.path.exists(cond_dir):
            print(f"Directory not found: {cond_dir}")
            continue
        
        # 遍历掩码类型
        for mask_type in ['random', 'sampler']:
            mask_dir = os.path.join(cond_dir, mask_type)
            if not os.path.exists(mask_dir):
                print(f"Directory not found: {mask_dir}")
                continue
            
            # 遍历每个实验目录
            for exp_name in os.listdir(mask_dir):
                exp_dir = os.path.join(mask_dir, exp_name)
                if not os.path.isdir(exp_dir):
                    continue
                
                print(f"Processing {cond_type}/{mask_type}/{exp_name}")
                
                # 获取所有配对的图像路径
                image_pairs = get_image_pairs(exp_dir)
                if not image_pairs:
                    print(f"No valid image pairs found in {exp_dir}")
                    continue
                
                print(f"Found {len(image_pairs)} image pairs")
                
                # 计算每对图像的指标
                metrics_list = []
                for recon_path, label_path in image_pairs:
                    try:
                        # 加载图像
                        recon_img = load_image(recon_path).to(device)
                        gt_img = load_image(label_path).to(device)
                        
                        # 计算指标
                        metrics = evaluate_metrics(
                            recon_img, 
                            gt_img, 
                            recon_img,
                            device=device,
                            lpips_fn=lpips_fn  # 传入已初始化的LPIPS模型
                        )
                        metrics_list.append(metrics)
                    except Exception as e:
                        print(f"Error processing images: {e}")
                        print(f"Recon: {recon_path}")
                        print(f"Label: {label_path}")
                        continue
                
                if metrics_list:
                    # 计算平均指标
                    avg_metrics = {k: float(np.mean([m[k] for m in metrics_list])) for k in metrics_list[0].keys()}
                    
                    # 保存新的指标（使用版本控制的文件名）
                    metrics_file = f"{metrics_version}.json"
                    save_path = os.path.join(exp_dir, metrics_file)
                    with open(save_path, 'w') as f:
                        json.dump(avg_metrics, f, indent=4)
                    print(f"Saved metrics to {save_path}")
    
    # 创建汇总文件（也使用版本控制）
    create_summary(base_dir, metrics_version)

def create_summary(base_dir, metrics_version):
    """创建所有实验结果的汇总，并按照mask和noise水平排序
    Args:
        base_dir: 基础目录
    """
    all_results = {}
    metrics_file = f"{metrics_version}.json"
    
    # 遍历所有实验目录
    for cond_type in ['cond', 'uncond']:
        all_results[cond_type] = {}
        for mask_type in ['random', 'sampler']:
            # 用于临时存储结果的字典
            temp_results = {}
            
            mask_dir = os.path.join(base_dir, cond_type, mask_type)
            if not os.path.exists(mask_dir):
                continue
                
            for exp_name in os.listdir(mask_dir):
                if not os.path.isdir(os.path.join(mask_dir, exp_name)):
                    continue
                    
                # 解析实验名称中的mask和noise值
                try:
                    # 从形如 "mask0.70_noise0.070" 的名称中提取值
                    parts = exp_name.split('_')
                    mask_val = float(parts[0].replace('mask', ''))
                    noise_val = float(parts[1].replace('noise', ''))
                    
                    metrics_path = os.path.join(mask_dir, exp_name, metrics_file)
                    if os.path.exists(metrics_path):
                        with open(metrics_path, 'r') as f:
                            metrics = json.load(f)
                            # 将mask和noise值添加到metrics中
                            metrics['mask_ratio'] = mask_val
                            metrics['noise_level'] = noise_val
                            temp_results[exp_name] = metrics
                except:
                    print(f"Warning: Could not parse experiment name: {exp_name}")
                    continue
            
            # 按照mask比例和noise水平排序
            sorted_results = {}
            
            if temp_results:  # 确保有结果再处理
                # 获取所有唯一的mask值并排序
                mask_values = sorted(set(result['mask_ratio'] for result in temp_results.values()))
                
                for mask_val in mask_values:
                    mask_key = f"mask_{mask_val:.2f}"
                    sorted_results[mask_key] = {}
                    
                    # 获取当前mask值对应的所有结果
                    mask_results = {k: v for k, v in temp_results.items() 
                                  if v['mask_ratio'] == mask_val}
                    
                    # 按noise水平排序
                    for exp_name in sorted(mask_results.keys(), 
                                         key=lambda x: mask_results[x]['noise_level']):
                        metrics = mask_results[exp_name]
                        noise_key = f"noise_{metrics['noise_level']:.3f}"
                        # 移除额外添加的字段
                        metrics.pop('mask_ratio')
                        metrics.pop('noise_level')
                        sorted_results[mask_key][noise_key] = metrics
            
            all_results[cond_type][mask_type] = sorted_results
    
    # 保存汇总结果
    summary_file = f"summary_{metrics_version}.json"
    summary_path = os.path.join(base_dir, summary_file)
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"Saved summary to {summary_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='results')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 重新计算指标
    recalculate_metrics(args.base_dir, device) 
