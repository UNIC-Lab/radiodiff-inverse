import os
import json
from functools import partial
import argparse
import yaml

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata, Rbf

from guided_diffusion.measurements import get_noise, get_operator
from data.dataloader import get_dataset, get_dataloader
from util.img_utils import clear_color, mask_generator
from util.metrics import evaluate_metrics
from util.experiment_utils import create_experiment_dir, save_experiment_results
from lpips import LPIPS

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def interpolate_image(y_n, mask, method='rbf', device='cuda'):
    """使用不同的插值方法重建图像
    
    Args:
        y_n: 带噪声的测量值 [B,C,H,W]
        mask: 采样掩码 [B,C,H,W]，1表示已知点，0表示未知点
        method: 插值方法，可选['nearest', 'linear', 'cubic', 'rbf']
    """
    B, C, H, W = y_n.shape
    result = torch.zeros_like(y_n)
    
    # 将数据移到CPU进行插值计算
    y_n = y_n.cpu().numpy()
    mask = mask.cpu().numpy()
    
    # 为每个batch和通道进行插值
    for b in range(B):
        for c in range(C):
            # 获取已知点的坐标和值
            known_points = np.where(mask[b,c] > 0)
            points = np.array(list(zip(known_points[0], known_points[1])))
            values = y_n[b,c][known_points]
            
            # 创建需要插值的网格点
            grid_y, grid_x = np.mgrid[0:H, 0:W]
            
            if method == 'rbf':
                # 使用RBF插值
                rbf = Rbf(points[:,0], points[:,1], values, function='multiquadric')
                interpolated = rbf(grid_y, grid_x)
            else:
                # 使用griddata进行插值
                grid_points = np.array(list(zip(grid_y.ravel(), grid_x.ravel())))
                interpolated = griddata(points, values, grid_points, method=method)
                interpolated = interpolated.reshape(H, W)
            
            # 将插值结果放回原始位置
            result[b,c] = torch.from_numpy(interpolated)
    
    return result.to(device)

def run_experiment(args):
    """运行插值重建实验"""
    # 加载配置
    task_config = load_yaml(args.task_config)
    
    # 设备配置
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu')
    
    # 准备数据加载器
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop((256, 256)),
        transforms.Resize((256, 256)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = get_dataset(**task_config['data'], transforms=transform)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)
    
    # 初始化LPIPS
    lpips_fn = LPIPS(net='alex').to(device)
    
    # 实验结果存储
    all_results = {}
    
    # 遍历所有参数组合
    for mask_ratio in args.mask_ratios:
        for noise_level in args.noise_levels:
            for method in args.interpolation_methods:
                # 更新配置
                task_config['measurement']['mask_opt']['mask_ratio'] = mask_ratio
                task_config['measurement']['noise']['sigma'] = noise_level
                
                # 创建实验目录
                exp_dir = create_experiment_dir(
                    base_dir=os.path.join(args.save_dir, f'interpolation_{method}'),
                    mask_type=task_config['measurement']['mask_opt']['mask_type'],
                    mask_ratio=mask_ratio,
                    noise_level=noise_level,
                    version=args.version
                )
                
                # 准备操作器和噪声
                operator = get_operator(device=device, **task_config['measurement']['operator'])
                noiser = get_noise(**task_config['measurement']['noise'])
                
                # 运行实验并收集指标
                metrics_list = []
                for i, (ref_img, building_mask, img_path) in enumerate(loader):
                    if i >= args.num_images:
                        break
                    
                    # 从原始图像路径提取文件名
                    filename = os.path.basename(img_path[0])
                    
                    ref_img = ref_img.to(device)
                    building_mask = building_mask.to(device)
                    
                    # 生成掩码和测量值
                    mask = mask_generator(**task_config['measurement']['mask_opt'])(ref_img, building_mask)
                    y = operator.forward(ref_img, mask=mask)
                    y_n = noiser(y)
                    
                    # 使用插值方法重建
                    recon = interpolate_image(y_n, mask, method=method, device=device)
                    
                    # 保存图像
                    os.makedirs(os.path.join(exp_dir, 'input'), exist_ok=True)
                    os.makedirs(os.path.join(exp_dir, 'label'), exist_ok=True)
                    os.makedirs(os.path.join(exp_dir, 'recon'), exist_ok=True)
                    
                    plt.imsave(os.path.join(exp_dir, 'input', filename), clear_color(y_n))
                    plt.imsave(os.path.join(exp_dir, 'label', filename), clear_color(ref_img))
                    plt.imsave(os.path.join(exp_dir, 'recon', filename), clear_color(recon))
                    
                    # 计算指标
                    metrics = evaluate_metrics(recon, ref_img, y_n, device=device, lpips_fn=lpips_fn)
                    metrics_list.append(metrics)
                
                # 计算平均指标
                avg_metrics = {}
                for key in metrics_list[0].keys():
                    avg_metrics[key] = float(np.mean([float(m[key]) for m in metrics_list]))
                
                # 保存结果
                exp_key = f"{method}_mask{mask_ratio:.2f}_noise{noise_level:.3f}"
                all_results[exp_key] = avg_metrics
                
                # 保存当前实验的指标
                with open(os.path.join(exp_dir, 'metrics.json'), 'w') as f:
                    json.dump(avg_metrics, f, indent=4)
    
    # 保存所有实验结果
    save_experiment_results(args.save_dir, all_results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_config', type=str, default='configs/inpainting_config.yaml')
    parser.add_argument('--save_dir', type=str, default='results/interpolation')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--mask_ratios', type=float, nargs='+', 
                       default=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95])
    parser.add_argument('--noise_levels', type=float, nargs='+', 
                       default=[0.01, 0.03, 0.05, 0.07, 0.09])
    parser.add_argument('--interpolation_methods', type=str, nargs='+',
                       default=['nearest', 'linear', 'cubic', 'rbf'],
                       help='Interpolation methods to test')
    parser.add_argument('--num_images', type=int, default=200)
    parser.add_argument('--version', type=str, default='v1')
    
    args = parser.parse_args()
    run_experiment(args) 
