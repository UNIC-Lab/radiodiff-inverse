"""Experimental road-conditioned sampling pipeline.

This script is kept for reference and requires additional dataset/model wiring
before production use.
"""

import os
import json
from functools import partial
import argparse
import yaml
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from piq import psnr, ssim
from piq.perceptual import LPIPS
from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from guided_diffusion.svd_replacement import Deblurring, Deblurring2D
from data.dataloader import get_dataset, get_dataloader
from util.img_utils import clear_color, mask_generator
from util.logger import get_logger
from util.metrics import evaluate_metrics
from util.experiment_utils import create_experiment_dir, save_experiment_results

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def run_experiment(args):
    """运行批量实验"""
    # 加载配置
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)
    
    # 设备配置
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    
    # 加载模型
    # 提醒：您需要确保 model_config.yaml 中的 in_channels 与（图像+条件图）的总通道数匹配
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()
    
    # 准备数据加载器
    data_config = task_config['data']
    # 注意：这里的 Normalize 是针对 Radiomap 的，条件图在 Dataset 中只做了空间变换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256), antialias=True),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = get_dataset(**data_config, transforms=transform)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)
    
    # 实验结果存储
    all_results = {}
    
    # MODIFIED: 移除了 mask_ratio 的循环，只循环噪声水平
    for noise_level in args.noise_levels:
        # 更新配置
        task_config['measurement']['noise']['sigma'] = noise_level
        
        # MODIFIED: 让实验目录名更有意义
        mask_opt = task_config['measurement']['mask_opt']
        mask_name = mask_opt['mask_type']
        if mask_name == 'road_sensors':
             # 如果是道路传感器，则在目录名中包含宽度信息
            width = mask_opt.get('box_size', [0])[0]
            mask_name = f"{mask_name}_width{width}"

        exp_dir = create_experiment_dir(
            base_dir=args.save_dir,
            mask_type=mask_name,
            noise_level=noise_level,
            version=args.version
        )
        
        # 准备操作器和噪声
        operator = get_operator(device=device, **task_config['measurement']['operator'])
        noiser = get_noise(**task_config['measurement']['noise'])
        
        # 准备条件方法
        cond_method = get_conditioning_method(
            task_config['conditioning']['method'],
            operator,
            noiser,
            **task_config['conditioning']['params']
        )
        
        # 准备采样器
        sampler = create_sampler(**diffusion_config, c_rate=args.c_rate, particle_size=args.particle_size)
        sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=cond_method.conditioning)
        
        # 运行实验并收集指标
        metrics_list = []
        # MODIFIED: 解包所有数据项 (ref_img, building_mask, road_map, antenna_map, img_path)
        for i, (ref_img, building_mask, road_map, antenna_map, img_path) in enumerate(loader):
            if i >= args.num_images:
                break
            
            filename = os.path.basename(img_path[0])
            
            # MODIFIED: 将所有张量移动到设备
            ref_img = ref_img.to(device)
            building_mask = building_mask.to(device)
            road_map = road_map.to(device)
            antenna_map = antenna_map.to(device)
            
            # MODIFIED: 调用 mask_generator 时传入 road_map
            mask = mask_generator(**task_config['measurement']['mask_opt'])(img=ref_img, road_map=road_map)
            y = operator.forward(ref_img, mask=mask)
            y_n = noiser(y)
            
            # ADDED: 准备用于模型输入的条件张量
            # 假设所有条件图都是 [B, 1, H, W]，而 ref_img 可能是 [B, 3, H, W]
            # 我们需要将条件图扩展到与 ref_img 相同的通道数，然后再拼接
            # 这里我们假定 ref_img 是单通道的。如果不是，需要调整
            # 另外，我们假设模型输入是 [x_t, building, road, antenna]
            cond_maps = torch.cat([building_mask, road_map, antenna_map], dim=1)

            # 采样
            x_start = torch.randn_like(ref_img).requires_grad_()
            # MODIFIED: 传入 cond_maps
            sample, _ = sample_fn(
                x_start=x_start,
                measurement=y_n,
                cond_maps=cond_maps,
                operator=operator,
                op='inpainting',
                mask=mask,
                record=False,
                save_root=exp_dir
            )
            
            # 保存图像
            # ... (这部分代码不变)
            
            # 计算指标
            metrics = evaluate_metrics(ref_img, sample, y)
            metrics_list.append(metrics)
        
        # 计算平均指标
        avg_metrics = {}
        if metrics_list:
            for key in metrics_list[0].keys():
                avg_metrics[key] = float(np.mean([float(m[key]) for m in metrics_list]))
        
        # MODIFIED: 结果键不再包含 mask_ratio
        exp_key = f"noise{noise_level:.3f}"
        all_results[exp_key] = avg_metrics
        
        with open(os.path.join(exp_dir, 'metrics.json'), 'w') as f:
            json.dump(avg_metrics, f, indent=4)
    
    save_experiment_results(os.path.dirname(exp_dir), all_results) # 保存到父目录

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # ... (parser arguments remain mostly the same, but --mask_ratios is now unused for this setup)
    # MODIFIED: 可以给 mask_ratios 一个无意义的默认值，因为循环已被移除，但参数定义可能仍存在
    parser.add_argument('--mask_ratios', type=float, nargs='+', default=[1.0], help='Unused for road_sensors type, but kept for compatibility.')
    # ... (其他参数)
    args = parser.parse_args()
    run_experiment(args)
