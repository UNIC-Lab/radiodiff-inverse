import os
import json
from functools import partial
import argparse
import yaml

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from data.dataloader import get_dataset, get_dataloader
from util.img_utils import clear_color, mask_generator
from util.logger import get_logger
from util.metrics import evaluate_metrics
from util.experiment_utils import create_experiment_dir, save_experiment_results

def run_experiment(args):
    """运行无条件掩码实验 - ImageNet模型版本"""
    # 加载配置
    model_config = yaml.safe_load(open('configs/model_imagenet_config.yaml'))
    diffusion_config = yaml.safe_load(open(args.diffusion_config))
    task_config = yaml.safe_load(open(args.task_config))
    
    # 设备配置
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()
    
    # 准备数据加载器
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop((256, 256)),
        transforms.Resize((256, 256)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = get_dataset(**task_config['data'], transforms=transform)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)
    
    # 加载建筑物掩码（仅用于评估）
    building_dataset = get_dataset(
        name='building',
        root=task_config['data']['root'],
        image_dir=task_config['data']['image_dir'],
        building_dir=task_config['data']['building_dir'],
        transforms=transform
    )
    building_loader = get_dataloader(building_dataset, batch_size=1, num_workers=0, train=False)
    
    # 实验结果存储
    all_results = {}
    
    # 遍历所有参数组合
    for mask_ratio in args.mask_ratios:
        for noise_level in args.noise_levels:
            # 更新配置
            task_config['measurement']['mask_opt']['mask_ratio'] = mask_ratio
            task_config['measurement']['noise']['sigma'] = noise_level
            
            # 创建实验目录
            exp_dir = create_experiment_dir(
                base_dir=args.save_dir,
                mask_type=task_config['measurement']['mask_opt']['mask_type'],
                mask_ratio=mask_ratio,
                noise_level=noise_level,
                version=f"imagenet_{args.version}"  # 添加模型标识
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
            
            # 运行实验
            metrics_list = []
            for i, ((img, building_mask, img_path), (_, building_mask_2, _)) in enumerate(zip(loader, building_loader)):
                if i >= args.num_images:
                    break
                
                img = img.to(device)
                building_mask = building_mask.to(device)
                
                # 生成掩码和测量值（不使用建筑物信息）
                mask = mask_generator(**task_config['measurement']['mask_opt'])(img)
                y = operator.forward(img, mask=mask)
                y_n = noiser(y)
                
                # 采样（不使用建筑物信息）
                x_start = torch.randn_like(img)
                sample, _ = sample_fn(
                    x_start=x_start,
                    measurement=y_n,
                    operator=operator,
                    op='inpainting',
                    mask=mask,
                    record=False,
                    save_root=exp_dir
                )
                
                # 评估指标（包括建筑物区域的评估）
                metrics = evaluate_metrics(
                    recon_img=sample,
                    gt_img=img,
                    degraded_img=operator.forward(img, mask=mask),
                    device=device
                )
                metrics_list.append(metrics)
                
                # 保存结果
                plt.imsave(os.path.join(exp_dir, 'input', f"{i:04d}_input.png"), clear_color(y_n))
                plt.imsave(os.path.join(exp_dir, 'label', f"{i:04d}_label.png"), clear_color(img))
                plt.imsave(os.path.join(exp_dir, 'recon', f"{i:04d}_recon.png"), clear_color(sample))
                plt.imsave(os.path.join(exp_dir, 'building_mask', f"{i:04d}_mask.png"), clear_color(building_mask))
            
            # 计算平均指标
            avg_metrics = {k: float(np.mean([m[k] for m in metrics_list])) for k in metrics_list[0].keys()}
            
            # 保存结果
            exp_key = f"imagenet_uncond_mask{mask_ratio:.2f}_noise{noise_level:.3f}"
            all_results[exp_key] = avg_metrics
            
            # 保存当前实验的指标
            with open(os.path.join(exp_dir, 'metrics.json'), 'w') as f:
                json.dump(avg_metrics, f, indent=4)
    
    # 保存所有实验结果
    save_experiment_results(exp_dir, all_results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--diffusion_config', type=str, default='configs/diffusion_config.yaml')
    parser.add_argument('--task_config', type=str, default='configs/inpainting_config.yaml')
    parser.add_argument('--save_dir', type=str, default='results/imagenet_uncond')
    parser.add_argument('--c_rate', type=float, default=0.95)
    parser.add_argument('--particle_size', type=int, default=5)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--mask_ratios', type=float, nargs='+', 
                       default=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95])
    parser.add_argument('--noise_levels', type=float, nargs='+', default=[0.05, 0.1, 0.15])
    parser.add_argument('--num_images', type=int, default=200)
    parser.add_argument('--version', type=str, default='v1')
    
    args = parser.parse_args()
    run_experiment(args) 
